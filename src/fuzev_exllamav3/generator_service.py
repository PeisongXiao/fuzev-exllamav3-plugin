"""Generation and decisions through an existing ExLlamaV3 Generator."""

import asyncio
import inspect
import time
from dataclasses import dataclass
from uuid import uuid4

import torch

from fuzev.api import APIError, decision_overflow, validate_decision, validation_error
from fuzev.encoding import IMAGE_PLACEHOLDER, VERSION, candidate_suffix
from fuzev.images import decode_image
from fuzev.prefix import DEFAULT_PREFIX_CAPACITY, PrefixRegistry
from .factory import LMConditionedHead
from .preparation import ReadoutPreparer
from .reasoning import ReasoningSplitter, chat_message


@dataclass(frozen=True)
class PreparedState:
    tokens: tuple
    embeddings: tuple
    additional_context: str


@dataclass(frozen=True)
class PreparedDecision:
    state: PreparedState
    query: tuple
    tails: tuple
    candidates: tuple
    route: str = "labelled"
    modality: str = "text"


def image_text(rendered, embedding):
    if IMAGE_PLACEHOLDER not in rendered:
        raise validation_error("image", "The selected chat template does not include an image placeholder")
    return rendered.replace(IMAGE_PLACEHOLDER, embedding.text_alias, 1)


def require_fork():
    from exllamav3.generator.job import Job

    if "readout_root_tokens" not in inspect.signature(Job.__init__).parameters:
        raise RuntimeError("This ExLlamaV3 build lacks the FuZev decision readout; "
                           "install exllamav3-fuzev 1.6.1+fuzev.v1.1 "
                           "(see fuzev/docs/install.md#install-the-engine).")


class GeneratorService:
    def __init__(self, generator, tokenizer, head=None, *, model_name,
                 context_tokens=None, chat_template_kwargs=None,
                 image_embeddings=None, prefix_capacity=DEFAULT_PREFIX_CAPACITY):
        require_fork()
        self.async_generator, self.generator = generator, generator.generator
        self.tokenizer, self.model_name = tokenizer, model_name
        limit = min(self.generator.model.config.max_position_embeddings,
                    self.generator.max_total_tokens - 1 - self.generator.num_draft_tokens)
        self.context_tokens = min(context_tokens, limit) if context_tokens is not None else limit
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.image_embeddings = image_embeddings
        self.preparer = ReadoutPreparer(tokenizer, encoding_version=head.encoding_version if head is not None else VERSION)
        self.preparer.render([{"role": "user", "content": ""}])  # load the Hugging Face tokenizer before serving
        if head is not None:
            if head.metadata["hidden_size"] != self.generator.model.config.hidden_size:
                raise ValueError("Decision head feature width differs from the model configuration")
            if not isinstance(head, LMConditionedHead):
                head = LMConditionedHead(self.generator.model, self.preparer.encode, head)
        self.head, self.closed = head, False
        self._close_task = None
        self.jobs = set()
        self.prefixes = PrefixRegistry(capacity=prefix_capacity)

    def _check(self):
        if self.closed:
            raise APIError(503, "Server is closing")
        if self.async_generator._closing or self.async_generator.error is not None:
            raise APIError(503, "Generation engine is unavailable")

    async def _events(self, job):
        self._check()
        from exllamav3.generator.async_generator import AsyncJob

        handle = AsyncJob.from_job(self.async_generator, job, enqueue=False)
        self.jobs.add(handle)
        try:
            await self.async_generator.enqueue_async(handle)
            async for event in handle:
                yield event
                if event.get("eos"):
                    return
            raise asyncio.CancelledError
        finally:
            try:
                await handle.cancel()
            finally:
                self.jobs.discard(handle)

    async def _run(self, job):
        events = self._events(job)
        result = None
        try:
            async for event in events:
                if event.get("eos"):
                    result = event
                    break
        finally:
            try:
                await events.aclose()
            except BaseException:
                if result is not None and (root := result.get("readout_root")) is not None:
                    await self.async_generator.run_on_owner(root.close)
                raise
        return result

    def retain_result(self, result):
        prefix = self.prefixes.retain(
            result["sequence_ids"].reshape(-1).tolist(),
            payload=tuple(result["job"].embeddings or ()))
        return {"prefix_id": prefix.id}

    def release_prefix(self, prefix_id):
        return self.prefixes.release(prefix_id)

    async def _image(self, image):
        if self.image_embeddings is None:
            raise validation_error("image", "Configure the model's vision encoder to use images")
        decoding = asyncio.create_task(asyncio.to_thread(decode_image, image))
        try:
            rgb = await asyncio.shield(decoding)
        except asyncio.CancelledError:
            drain = asyncio.gather(decoding, return_exceptions=True)
            while not drain.done():
                try:
                    await asyncio.shield(drain)
                except asyncio.CancelledError:
                    pass
            decoded = drain.result()[0]
            if not isinstance(decoded, BaseException):
                decoded.close()
            raise
        with rgb:
            self._check()
            embedding = None

            def embed():
                nonlocal embedding
                embedding = self.image_embeddings(self.tokenizer, rgb)

            try:
                await self.async_generator.run_on_owner(embed)
            except BaseException:
                if inspect.isawaitable(embedding):
                    pending = asyncio.ensure_future(embedding)
                    pending.cancel()
                    drain = asyncio.gather(pending, return_exceptions=True)
                    while not drain.done():
                        try:
                            await asyncio.shield(drain)
                        except asyncio.CancelledError:
                            pass
                    drain.result()
                raise
            return await embedding if inspect.isawaitable(embedding) else embedding

    def _generation_prompt(self, request, embeddings):
        options = {**self.chat_template_kwargs, **request.chat_template_kwargs}
        if request.prompt is not None:
            rendered = (embeddings[0].text_alias + "\n" if embeddings else "") + request.prompt
        else:
            messages = [message.model_dump() for message in request.messages]
            if embeddings:
                user = next((row for row in messages if row["role"] == "user"), None)
                if user is None:
                    user = {"role": "user", "content": ""}
                    messages.append(user)
                user["content"] = embeddings[0].text_alias + "\n" + user["content"]
            from jinja2 import TemplateError
            try:
                rendered = self.tokenizer.hf_render_chat_template(
                    messages, **{**options, "add_generation_prompt": True})
            except TemplateError as error:
                raise validation_error("messages", str(error)) from None
        prompt_ids = self.tokenizer.encode(rendered, encode_special_tokens=True, embeddings=embeddings)
        return prompt_ids, rendered.rfind("<think>") > rendered.rfind("</think>")

    async def _generation_job(self, request):
        self._check()
        embeddings = [await self._image(request.image)] if request.image is not None else []
        prompt_ids, thinking = await asyncio.to_thread(self._generation_prompt, request, embeddings)
        prompt_tokens = prompt_ids.shape[-1]
        needed = prompt_tokens + (request.max_tokens if request.max_tokens is not None else 1)
        if needed > self.context_tokens:
            raise validation_error("max_tokens" if request.max_tokens is not None
                                   else "prompt" if request.prompt is not None else "messages",
                f"Prompt plus output need {needed} tokens; the model's context window is {self.context_tokens}")
        from exllamav3.generator.job import Job
        from exllamav3.generator.sampler import ComboSampler

        job_options = {"stop_conditions": list(self.generator.model.config.eos_token_id_list) + request.stop,
                       "max_new_tokens": request.max_tokens if request.max_tokens is not None else self.context_tokens - prompt_tokens}
        defaults = getattr(self.tokenizer, "generation_config_dict", None) or {}
        sampling = {key: defaults[key] for key in ("temperature", "top_p", "top_k", "min_p") if key in defaults}
        sampling.update({key: getattr(request, key) for key in ("temperature", "top_p", "top_k")
                         if getattr(request, key) is not None})
        if sampling:
            job_options["sampler"] = ComboSampler(**sampling)
        if request.seed is not None:
            job_options["seed"] = request.seed
        job = Job(prompt_ids, embeddings=embeddings,
                  max_rq_tokens=self.generator.recurrent_checkpoint_interval,
                  retain_decision_prefix=request.retain_decision_prefix, **job_options)
        return job, prompt_tokens, thinking

    async def generate(self, request, *, request_id=None):
        job, prompt_tokens, thinking = await self._generation_job(request)
        result = await self._run(job)
        completion_tokens = result["new_tokens"]
        choice = {"index": 0, "finish_reason": "length" if result.get("eos_reason") == "max_new_tokens" else "stop"}
        if request.messages is not None:
            choice["message"] = chat_message(result["full_completion"], thinking=thinking)
        else:
            choice["text"] = result["full_completion"]
        return {"id": request_id or "req_" + uuid4().hex, "model": self.model_name,
                "object": "chat.completion" if request.messages is not None else "text_completion",
                "created": int(time.time()), "choices": [choice],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
                **(self.retain_result(result) if request.retain_decision_prefix else {})}

    async def generate_stream(self, request, *, request_id=None):
        job, prompt_tokens, thinking = await self._generation_job(request)
        chat = request.messages is not None
        metadata = {"id": request_id or "req_" + uuid4().hex, "model": self.model_name,
                    "object": "chat.completion.chunk" if chat else "text_completion",
                    "created": int(time.time())}

        def chunk(value, finish_reason=None, **extra):
            choice = {"index": 0, "finish_reason": finish_reason,
                      "delta" if chat else "text": value}
            return {**metadata, "choices": [choice], **extra}

        if chat:
            yield chunk({"role": "assistant"})
        splitter = ReasoningSplitter(thinking=thinking)
        events = self._events(job)
        try:
            async for event in events:
                text = event.get("text", "")
                if chat:
                    for delta in splitter.feed(text, final=bool(event.get("eos"))):
                        yield chunk(delta)
                elif text:
                    yield chunk(text)
                if event.get("eos"):
                    completion_tokens = event["new_tokens"]
                    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                             "total_tokens": prompt_tokens + completion_tokens}
                    yield chunk({} if chat else "",
                                "length" if event.get("eos_reason") == "max_new_tokens" else "stop",
                                usage=usage,
                                **(self.retain_result(event) if request.retain_decision_prefix else {}))
        finally:
            await events.aclose()

    async def _readout(self, ids, embeddings, *, checkpoint=True,
                       readout_root_tokens=None, readout_root=None):
        from exllamav3.generator.job import Job

        return await self._run(Job(ids,
                                   max_new_tokens=0, readout=True, embeddings=list(embeddings),
                                   checkpoint=checkpoint, readout_root_tokens=readout_root_tokens,
                                   readout_root=readout_root))

    async def prepare_state(self, request, *, image=None):
        self._check()
        prefix = None
        if request.prefix_id is not None:
            prefix = self.prefixes.get(request.prefix_id)
        source_image = request.image if image is None else image
        embedding = await self._image(source_image) if source_image is not None else None
        embeddings = list(prefix.payload or ()) if prefix else []
        if embedding is not None:
            embeddings.append(embedding)
        additional = request.additional_context
        if prefix:
            accepted = prefix.token_ids
            if embedding is not None:
                additional = embedding.text_alias + "\n" + additional
        elif embedding is not None:
            def render_image():
                rendered = self.preparer.render(request.messages, image=True)
                return self.preparer.encode(image_text(rendered, embedding), embeddings)
            accepted = await asyncio.to_thread(render_image)
        else:
            accepted = await asyncio.to_thread(self.preparer.base, request.messages)
        return PreparedState(tuple(accepted), tuple(embeddings), additional)

    def _admit_decision(self, request, *, image=None, candidate_field="candidates", primitive_field="primitive"):
        prefix = self.prefixes.get(request.prefix_id) if request.prefix_id is not None else None
        modality = "image_text" if image is not None or request.image is not None or (prefix and prefix.payload) else "text"
        route = validate_decision(request, self.head, modality=modality,
                                  candidate_field=candidate_field, primitive_field=primitive_field)
        return route, modality

    def _prepare_decision(self, request, state, context_field, *, route="labelled", modality="text"):
        candidates = [candidate.model_dump() for candidate in request.candidates]
        query = state.tokens + self.preparer.encode(
            self.preparer.query_suffix(request.question, candidates, state.additional_context, route=route),
            state.embeddings)
        tails = tuple(self.preparer.encode(candidate_suffix(candidate)) for candidate in candidates)
        needed = len(query) + max(map(len, tails))
        decision_overflow(needed, self.context_tokens, context_field)
        return PreparedDecision(state, query, tails, tuple(candidates), route, modality)

    async def prepare_decisions(self, requests, *, image=None, question_keys=None):
        self._check()
        from exllamav3.constants import PAGE_SIZE

        fields = [["questions", key] for key in question_keys] if question_keys is not None else [
            "candidates" for _ in requests]
        admissions = {}
        for request, field in zip(requests, fields):
            question_type = "score" if request.primitive == "ordinal" else request.primitive
            admissions[id(request)] = self._admit_decision(
                request, image=image,
                candidate_field=field + [question_type, "criteria"] if isinstance(field, list) else field,
                primitive_field=field + ["type"] if isinstance(field, list) else "primitive")
        state = await self.prepare_state(requests[0], image=image)
        prepared = {id(request): await asyncio.to_thread(
            self._prepare_decision, request, state, field,
            route=admissions[id(request)][0], modality=admissions[id(request)][1])
                    for request, field in zip(requests, fields)}
        warmed = len(requests) >= 2 and len(state.tokens) > PAGE_SIZE
        warmup = None
        if warmed:
            warmup = await self._readout(torch.tensor([state.tokens], dtype=torch.long), state.embeddings)

        async def decide(request, request_id):
            result = await self.decide(request, request_id=request_id, prepared=prepared[id(request)])
            if request is not requests[0]:
                result["execution"]["input_tokens"] -= len(state.tokens)
            if warmed and request is requests[0]:
                result["execution"]["submitted_tokens"] += len(state.tokens)
                for key in ("cached_tokens", "forwarded_tokens"):
                    result["execution"][key] += warmup[key]
            return result
        return decide

    async def decide(self, request, *, request_id=None, prepared=None, context_field="candidates"):
        self._check()
        if prepared is None:
            route, modality = self._admit_decision(request, candidate_field=context_field)
            state = await self.prepare_state(request)
            prepared = await asyncio.to_thread(self._prepare_decision, request, state, context_field,
                                               route=route, modality=modality)
        query, tails, embeddings = prepared.query, prepared.tails, prepared.state.embeddings
        candidates = prepared.candidates
        query_ids = torch.tensor([query], dtype=torch.long)
        result = await self._readout(query_ids, embeddings, readout_root_tokens=max(map(len, tails)))
        root = result.get("readout_root")
        try:
            query_feature = result["features"]
            execution = {"input_tokens": len(query) + sum(map(len, tails)),
                         "submitted_tokens": len(query) * (len(tails) + 1) + sum(map(len, tails)),
                         "cached_tokens": result["cached_tokens"], "forwarded_tokens": result["forwarded_tokens"],
                         "completed_root_branches": 0}
            keys = []
            window = self.generator.max_batch_size
            for start in range(0, len(tails), window):
                async with asyncio.TaskGroup() as group:
                    tasks = [group.create_task(self._readout(
                                 torch.cat((query_ids, torch.tensor([tail], dtype=torch.long)), dim=-1),
                                 embeddings, checkpoint=False, readout_root=root))
                             for tail in tails[start:start + window]]
                for task in tasks:
                    result = task.result()
                    keys.append(result["features"])
                    for key in ("cached_tokens", "forwarded_tokens"):
                        execution[key] += result[key]
                    if root is not None and result["job"].readout_root is root:
                        execution["completed_root_branches"] += 1
        finally:
            if root is not None:
                await self.async_generator.run_on_owner(root.close)
        self._check()
        logits, raw, calibrated = await self.async_generator.run_on_owner(
            lambda: self.head.score(request.primitive, query_feature, torch.stack(keys),
                                    route=prepared.route, modality=prepared.modality))
        temperature = self.head.temperature_for(request.primitive, modality=prepared.modality, count=len(candidates))
        calibration_id = "raw" if temperature is None else "calibrated"
        return {"id": request_id or "req_" + uuid4().hex, "model": self.model_name,
                "primitive": request.primitive, "head_id": self.head.head_id,
                "calibration_id": calibration_id,
                "selected_id": candidates[max(range(len(logits)), key=logits.__getitem__)]["id"],
                "scores": [{"id": candidate["id"], "logit": logit, "raw_probability": p,
                            "calibrated_probability": q}
                           for candidate, logit, p, q in zip(candidates, logits, raw, calibrated)],
                "execution": execution}

    async def close(self):
        if self._close_task is None:
            self.closed = True
            self._close_task = asyncio.create_task(self._close())
        await self.async_generator._wait(self._close_task)

    async def _close(self):
        try:
            await asyncio.gather(*(handle.cancel() for handle in tuple(self.jobs)), return_exceptions=True)
        finally:
            self.prefixes.clear()
            if self.head is not None:
                retirement = self.async_generator._close_task
                if retirement is not None:
                    await self.async_generator._wait(retirement)
                    self.head.close()
                else:
                    await self.async_generator.run_on_owner(self.head.close)


def build_generator_service(generator, tokenizer, *, model_name, head_directory,
                            calibration="raw", head_device=None, **options):
    from fuzev.head import ServingHead

    head = ServingHead(head_directory, calibration=calibration, device=head_device)
    try:
        return GeneratorService(generator, tokenizer, head, model_name=model_name, **options)
    except BaseException:
        head.close()
        raise
