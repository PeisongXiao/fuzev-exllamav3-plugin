"""Generation and decisions through an existing ExLlamaV3 Generator."""

import asyncio
import inspect
import time
from dataclasses import dataclass
from uuid import uuid4

import torch

from fuzev.api import APIError, decision_overflow, validation_error
from fuzev.encoding import IMAGE_PLACEHOLDER, VERSION, candidate_suffix, query_suffix
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


def image_text(rendered, embedding):
    if IMAGE_PLACEHOLDER not in rendered:
        raise validation_error("image", "The selected chat template does not include an image placeholder")
    return rendered.replace(IMAGE_PLACEHOLDER, embedding.text_alias, 1)


def require_fork():
    from exllamav3.generator.job import Job

    if "readout" not in inspect.signature(Job.__init__).parameters:
        raise RuntimeError("This ExLlamaV3 build lacks FuZev readout; install exllamav3-fuzev "
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
        self.preparer = ReadoutPreparer(tokenizer)
        self.preparer.render([{"role": "user", "content": ""}])  # load the Hugging Face tokenizer before serving
        if head is not None:
            if head.encoding_version != VERSION:
                raise ValueError(f"Decision head requires encoding {head.encoding_version}; installed encoding is {VERSION}")
            if head.metadata["hidden_size"] != self.generator.model.config.hidden_size:
                raise ValueError("Decision head feature width differs from the model configuration")
            if not isinstance(head, LMConditionedHead):
                head = LMConditionedHead(self.generator.model, self.preparer.encode, head)
        self.head, self.closed = head, False
        self.jobs = set()
        self.prefixes = PrefixRegistry(capacity=prefix_capacity)

    def _check(self):
        if self.closed:
            raise APIError(503, "Server is closing")
        if self.async_generator.error is not None:
            raise APIError(503, "Generation engine is unavailable")

    async def _events(self, job):
        self._check()
        from exllamav3.generator.async_generator import AsyncJob

        handle = AsyncJob.from_job(self.async_generator, job)
        self.jobs.add(handle)
        try:
            async for event in handle:
                yield event
                if event.get("eos"):
                    return
            raise asyncio.CancelledError
        finally:
            try:
                if not job.is_finished:
                    cleanup = asyncio.create_task(handle.cancel())
                    while not cleanup.done():
                        try:
                            await asyncio.shield(cleanup)
                        except asyncio.CancelledError:
                            pass
                    cleanup.result()
            finally:
                self.jobs.discard(handle)

    async def _run(self, job):
        events = self._events(job)
        try:
            async for event in events:
                if event.get("eos"):
                    return event
        finally:
            await events.aclose()

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
            embedding = self.image_embeddings(self.tokenizer, rgb)
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

    async def _readout(self, tokens, embeddings, *, checkpoint=True):
        from exllamav3.generator.job import Job

        result = await self._run(Job(torch.tensor([tokens], dtype=torch.long),
                                    max_new_tokens=0, readout=True, embeddings=list(embeddings),
                                    checkpoint=checkpoint))
        return result["features"]

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

    def _prepare_decision(self, request, state, context_field):
        candidates = [candidate.model_dump() for candidate in request.candidates]
        query = state.tokens + self.preparer.encode(
            query_suffix(request.question, candidates, state.additional_context), state.embeddings)
        tails = tuple(self.preparer.encode(candidate_suffix(candidate)) for candidate in candidates)
        needed = len(query) + max(map(len, tails))
        decision_overflow(needed, self.context_tokens, context_field)
        return PreparedDecision(state, query, tails)

    async def prepare_decisions(self, requests, *, image=None, question_keys=None):
        from exllamav3.constants import PAGE_SIZE

        state = await self.prepare_state(requests[0], image=image)
        fields = (["questions", key] for key in question_keys) if question_keys is not None else (
            "candidates" for _ in requests)
        prepared = {id(request): await asyncio.to_thread(self._prepare_decision, request, state, field)
                    for request, field in zip(requests, fields)}
        warmed = len(requests) >= 2 and len(state.tokens) > PAGE_SIZE
        if warmed:
            await self._readout(state.tokens, state.embeddings)

        async def decide(request, request_id):
            result = await self.decide(request, request_id=request_id, prepared=prepared[id(request)])
            if warmed and request is requests[0]:
                result["execution"]["submitted_tokens"] += len(state.tokens)
            return result
        return decide

    async def decide(self, request, *, request_id=None, prepared=None, context_field="candidates"):
        self._check()
        if prepared is None:
            state = await self.prepare_state(request)
            prepared = await asyncio.to_thread(self._prepare_decision, request, state, context_field)
        query, tails, embeddings = prepared.query, prepared.tails, prepared.state.embeddings
        candidates = [candidate.model_dump() for candidate in request.candidates]
        query_feature = await self._readout(query, embeddings)
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(self._readout(query + tail, embeddings, checkpoint=False)) for tail in tails]
        keys = [task.result() for task in tasks]
        logits, raw, calibrated = self.head.score(request.primitive, query_feature, torch.stack(keys))
        return {"id": request_id or "req_" + uuid4().hex, "model": self.model_name,
                "primitive": request.primitive, "head_id": self.head.head_id,
                "calibration_id": self.head.calibration_id,
                "selected_id": candidates[max(range(len(logits)), key=logits.__getitem__)]["id"],
                "scores": [{"id": candidate["id"], "logit": logit, "raw_probability": p,
                            "calibrated_probability": q}
                           for candidate, logit, p, q in zip(candidates, logits, raw, calibrated)],
                "execution": {"submitted_tokens": len(query) * (len(tails) + 1) + sum(map(len, tails))}}

    async def close(self):
        if self.closed:
            return
        self.closed = True
        await asyncio.gather(*(handle.cancel() for handle in tuple(self.jobs)), return_exceptions=True)
        self.prefixes.clear()
        if self.head is not None:
            self.head.close()


def build_generator_service(generator, tokenizer, *, model_name, head_directory,
                            calibration="raw", head_device=None, **options):
    from fuzev.head import ServingHead

    head = ServingHead(head_directory, calibration=calibration, device=head_device)
    try:
        return GeneratorService(generator, tokenizer, head, model_name=model_name, **options)
    except BaseException:
        head.close()
        raise
