"""Public generation streams and cancellation over HTTP."""

import asyncio
import json

import pytest

from fuzev_exllamav3.server import create_app
from fuzev_exllamav3.systemone import HEADER
from test_systemone import client


def events(text):
    return [json.loads(line[6:]) if line[6:] != "[DONE]" else "[DONE]"
            for line in text.splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("chat", [False, True])
def test_generation_stream_golden(native, chat):
    service, engine = native
    engine.chunks = ["<thi", "nk>reason", "ing\n</th", "ink>\nAn", "swer."] if chat else ["plain", " answer"]
    engine.completion = "".join(engine.chunks)
    async def run():
        async with client(create_app(service)) as http:
            payload = {"stream": True, "retain_decision_prefix": True}
            payload.update(messages=[{"role": "user", "content": "Question"}] if chat else None,
                           prompt=None if chat else "Question")
            response = await http.post("/v1/chat/completions" if chat else "/v1/completions", json=payload,
                                       headers={HEADER: "stream-golden"})
            assert response.status_code == 200, response.text
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers[HEADER] == "stream-golden"
            chunks = events(response.text)
            assert chunks[-1] == "[DONE]"
            assert all(chunk["id"] == "stream-golden" and chunk["model"] == service.model_name for chunk in chunks[:-1])
            assert chunks[-2]["choices"][0]["finish_reason"] == "length"
            assert chunks[-2]["usage"]["completion_tokens"] == 3
            assert (await http.post("/v1/decisions", json={
                "primitive": "noul", "prefix_id": chunks[-2]["prefix_id"],
                "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}]})).status_code == 200
            if chat:
                assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks[:-1])
                deltas = [chunk["choices"][0]["delta"] for chunk in chunks[:-1]]
                assert deltas[0] == {"role": "assistant"}
                assert "".join(delta.get("reasoning", "") for delta in deltas) == "reasoning\n"
                assert "".join(delta.get("content", "") for delta in deltas) == "\nAnswer."
                assert max(index for index, delta in enumerate(deltas) if "reasoning" in delta) < min(
                    index for index, delta in enumerate(deltas) if "content" in delta)
            else:
                assert all(chunk["object"] == "text_completion" for chunk in chunks[:-1])
                assert "".join(chunk["choices"][0]["text"] for chunk in chunks[:-1]) == "plain answer"
    asyncio.run(run())


def test_nonthinking_chat_stream_content(native):
    service, engine = native
    engine.chunks = ["Plain", " answer"]
    engine.completion = "Plain answer"
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/chat/completions", json={
                "stream": True, "messages": [{"role": "user", "content": "Question"}],
                "chat_template_kwargs": {"enable_thinking": False}})
        deltas = [chunk["choices"][0]["delta"] for chunk in events(response.text)[:-1]]
        assert not any("reasoning" in delta for delta in deltas)
        assert "".join(delta.get("content", "") for delta in deltas) == "Plain answer"
    asyncio.run(run())


@pytest.mark.parametrize("disconnect", [True, False], ids=["disconnect", "cancel-route"])
def test_stream_is_incremental_and_cancellable(native, disconnect):
    service, engine = native
    engine.chunks, engine.completion = ["early output"], "early output"
    async def run():
        engine.block = asyncio.Event()
        queue, output, streamed = asyncio.Queue(), [], asyncio.Event()
        await queue.put({"type": "http.request", "body": json.dumps({"prompt": "Question", "stream": True}).encode(),
                         "more_body": False})
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/v1/completions", "raw_path": b"/v1/completions", "root_path": "",
                 "query_string": b"", "headers": [(b"x-typesafe-request-id", b"stream-cancel")],
                 "client": ("127.0.0.1", 1), "server": ("fixture", 80)}
        async def send(message):
            output.append(message)
            if b"early output" in message.get("body", b""):
                streamed.set()
        app = create_app(service)
        pending = asyncio.create_task(app(scope, queue.get, send))
        await streamed.wait()
        assert output[0]["status"] == 200 and not pending.done()
        async with client(app) as http:
            if disconnect:
                await queue.put({"type": "http.disconnect"})
            else:
                cancelled = await http.post("/v1/requests/stream-cancel/cancel")
                assert cancelled.status_code == 202
            await pending
            assert engine.cancelled > 0
            assert (await http.post("/v1/requests/stream-cancel/cancel")).status_code == 404
            engine.block.set()
            assert (await http.post("/v1/completions", json={"prompt": "Next request"})).status_code == 200
        streamed_text = b"".join(message.get("body", b"") for message in output).decode()
        if not disconnect:
            chunks = events(streamed_text)
            assert chunks[-2]["error"]["code"] == 499 and chunks[-1] == "[DONE]"
    asyncio.run(asyncio.wait_for(run(), 5))


def test_invalid_stream_request_returns_json(native):
    service, engine = native
    service.context_tokens = 2
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/completions", json={"prompt": "Too long", "stream": True})
        assert response.status_code == 422
        assert response.headers["content-type"].startswith("application/json")
        assert not engine.completed
    asyncio.run(run())


def test_stream_connection_lost_while_sending_headers(native):
    service, engine = native
    engine.chunks, engine.completion = ["early output"], "early output"
    async def run():
        engine.block = asyncio.Event()
        queue = asyncio.Queue()
        await queue.put({"type": "http.request", "body": json.dumps({"prompt": "Question", "stream": True}).encode(),
                         "more_body": False})
        scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.4"},
                 "http_version": "1.1", "method": "POST", "scheme": "http", "path": "/v1/completions",
                 "raw_path": b"/v1/completions", "root_path": "", "query_string": b"", "headers": [],
                 "client": ("127.0.0.1", 1), "server": ("fixture", 80)}
        async def send(message):
            raise OSError("Connection closed")
        from starlette.requests import ClientDisconnect
        app = create_app(service)
        with pytest.raises(ClientDisconnect):
            await app(scope, queue.get, send)
        assert engine.cancelled > 0
        engine.block.set()
        async with client(app) as http:
            assert (await http.post("/v1/completions", json={"prompt": "Next request"})).status_code == 200
    asyncio.run(asyncio.wait_for(run(), 5))
