"""Attach decision routes to an existing ExLlamaV3 service."""

import asyncio
import secrets

from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from fuzev.api import (APIError, api_error, finished_before_disconnect, models_response, parse_body,
                       request_id, sse_event, stream_error)
from fuzev.systemone import SystemOneAdapter, SystemOneRequest


HEADER = "X-TypeSafe-Request-ID"


class GenerationStream(StreamingResponse):
    def __init__(self, content, cleanup, **kwargs):
        super().__init__(content, **kwargs)
        self.cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self.cleanup()


class ServiceRoutes:
    def __init__(self, get_service, api_key=None):
        self.get_service, self.api_key = get_service, api_key
        self.tasks = {}
        self.closed = False

    def _check_request(self, http, key):
        if self.api_key is not None and not secrets.compare_digest(
            http.headers.get("authorization", "").encode(), ("Bearer " + self.api_key).encode()
        ):
            raise APIError(401, "Missing or invalid bearer token")
        if self.closed:
            raise APIError(503, "Server is closing")
        if key in self.tasks:
            raise APIError(409, "Request ID is already active")

    async def _drain(self, key, task):
        if not task.done():
            task.cancel()
        drain = asyncio.gather(task, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                pass
        if self.tasks.get(key) is task:
            self.tasks.pop(key, None)

    async def respond(self, http, operation, *, status_code=200, schema=None):
        key = request_id({})
        task = None
        try:
            key = getattr(http.state, "request_id", None) or request_id(http.headers)
            self._check_request(http, key)
            if schema is None:
                work = operation(self.get_service(), key)
            else:
                work = operation(self.get_service(), key, await parse_body(http, schema))
            task = asyncio.create_task(work)
            self.tasks[key] = task
            if not await finished_before_disconnect(http, task):
                raise APIError(499, "Client disconnected")
            result = await task
            if isinstance(result, StreamingResponse):
                return result
            return JSONResponse(result, status_code=status_code, headers={HEADER: key})
        except (Exception, asyncio.CancelledError) as error:
            status, body = api_error(error)
            return JSONResponse(body, status_code=status, headers={HEADER: key})
        finally:
            if task is not None:
                await self._drain(key, task)

    async def stream(self, http, iterator, key):
        queue = asyncio.Queue(maxsize=1)
        end = object()

        async def produce():
            try:
                async for chunk in iterator:
                    await queue.put(chunk)
            finally:
                await iterator.aclose()

        producer = asyncio.create_task(produce())
        self.tasks[key] = producer

        async def next_chunk():
            receive = asyncio.create_task(queue.get())
            try:
                await asyncio.wait({receive, producer}, return_when=asyncio.FIRST_COMPLETED)
                if receive.done():
                    return receive.result()
                try:
                    producer.result()
                except (Exception, asyncio.CancelledError) as error:
                    return error
                return end
            finally:
                if not receive.done():
                    receive.cancel()
                await asyncio.gather(receive, return_exceptions=True)

        first_task = asyncio.create_task(next_chunk())
        try:
            if not await finished_before_disconnect(http, first_task):
                raise APIError(499, "Client disconnected")
            first = await first_task
            if isinstance(first, BaseException):
                raise first
        except BaseException:
            first_task.cancel()
            await asyncio.gather(first_task, return_exceptions=True)
            await self._drain(key, producer)
            raise

        async def body():
            chunk = first
            while chunk is not end:
                if isinstance(chunk, BaseException):
                    yield stream_error(chunk)
                    break
                yield sse_event(chunk)
                chunk = await next_chunk()
            yield sse_event("[DONE]")

        return GenerationStream(body(), lambda: self._drain(key, producer), media_type="text/event-stream",
                                headers={HEADER: key, "Cache-Control": "no-cache"})

    async def close(self):
        self.closed = True
        tasks = tuple(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def cancel(self, key):
        task = self.tasks.get(key)
        if task is None or task.done():
            raise APIError(404, "Unknown or completed request")
        if not task.cancelling():
            task.cancel()
        return True


def add_systemone_routes(app, get_service, *, include_models=False, api_key=None):
    """Mount routes without replacing existing routes or closing the service."""
    paths = {route.path for route in app.routes}
    service = get_service()
    wanted = (["/v1/systemone"] if service.head is not None else [])
    if include_models:
        wanted.append("/v1/models")
    if paths.intersection(wanted):
        raise ValueError("A requested route already exists; merge model fields in the existing controller")
    routes = ServiceRoutes(get_service, api_key)

    if service.head is not None:
        adapter = SystemOneAdapter(
            service.decide, model_name=service.model_name,
            token_count=lambda text: service.tokenizer.encode(
                text, encode_special_tokens=False).shape[-1],
            head_id=service.head.head_id, max_options=service.head.max_options,
            primitives=service.head.primitives,
        )
        @app.post("/v1/systemone")
        async def systemone(http: Request):
            async def answer(service, key, request):
                return await adapter.answer(request, prepare=lambda prepared: service.prepare_decisions(
                    prepared, question_keys=tuple(request.questions)))

            return await routes.respond(http, answer, schema=SystemOneRequest)

    if include_models:
        @app.get("/v1/models")
        async def models(http: Request):
            async def result(service, key):
                return models_response(service.model_name, service.head)
            return await routes.respond(http, result)

    return routes
