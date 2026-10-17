"""CPU engine boundary, real native service, shared renderer and decision head."""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import get_ident
from types import ModuleType, SimpleNamespace

import jinja2
import pytest
import torch
from safetensors.torch import save_file

from fuzev._scoring import SharedDecisionHead
from fuzev.contracts import Primitive
from fuzev.encoding import VERSION
from fuzev.head import ServingHead
from fuzev_exllamav3.generator_service import GeneratorService, IMAGE_PLACEHOLDER


class Tokenizer:
    def encode(self, text, **kwargs):
        if len(text) == 2 and text[0] == " " and "A" <= text[1] <= "Z":
            return torch.tensor([[ord(text[1]) - ord("A")]])
        return torch.tensor([[ord(char) + 64 for char in text]], dtype=torch.long)

    def hf_render_chat_template(self, messages, **kwargs):
        if "chat_template" in kwargs:
            return jinja2.Environment().from_string(kwargs["chat_template"]).render(messages=messages, **kwargs)
        rendered = ""
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(IMAGE_PLACEHOLDER if part["type"] == "image" else part["text"] for part in content)
            rendered += f"<|im_start|>{message['role']}\n{content}<|im_end|>\n"
        return rendered + "<|im_start|>assistant\n" + ("<think>\n" if kwargs.get("enable_thinking", True) else "")


class Job:
    def __init__(self, input_ids, *, readout=False, checkpoint=True,
                 readout_root_tokens=None, readout_root=None, **kwargs):
        kwargs.update(readout=readout, checkpoint=checkpoint,
                      readout_root_tokens=readout_root_tokens, readout_root=readout_root)
        self.input_ids, self.options = input_ids, kwargs
        self.embeddings = kwargs.get("embeddings", [])
        self.readout_root = readout_root
        self.cached_tokens = self.forwarded_tokens = 0
        self.is_finished = False
        self.sequences = [SimpleNamespace(kv_position=input_ids.shape[-1])]


class Root:
    def __init__(self, owner, job):
        self.owner, self.ids = owner, job.input_ids.clone()
        self.bound = job.options["readout_root_tokens"]
        self.children = set()
        self.closed = False
        owner.roots.add(self)

    def attach(self, job):
        if self.closed or self not in self.owner.roots:
            raise ValueError("Closed root")
        length = self.ids.shape[-1]
        if (not torch.equal(job.input_ids[:, :length], self.ids)
                or not 0 < job.input_ids.shape[-1] - length <= self.bound):
            raise ValueError("Mismatched branch")
        self.children.add(job)

    def close(self):
        self.closed = True
        if not self.children:
            self.owner.roots.discard(self)


class AsyncJob:
    @classmethod
    def from_job(cls, owner, job, *, enqueue=True):
        handle = cls()
        handle.owner, handle.job = owner, job
        handle.delivered = False
        handle.task = None
        if enqueue:
            owner.enqueue(handle)
        return handle

    async def __aiter__(self):
        try:
            if not self.job.options["readout"] and self.owner.chunks is not None:
                for text in self.owner.chunks:
                    yield {"text": text, "eos": False}
                    await asyncio.sleep(0)
            result = await asyncio.shield(self.task)
            self.delivered = True
            yield result
        finally:
            if self.task.done():
                self.owner.jobs.discard(self.job)

    async def cancel(self):
        if self.task is None:
            return
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        if not self.delivered and not self.task.cancelled() and self.task.exception() is None:
            root = self.task.result().get("readout_root")
            if root is not None:
                await self.owner.run_on_owner(root.close)
        self.owner.jobs.discard(self.job)


class Engine:
    def __init__(self):
        self.jobs, self.completed = {"external-generation"}, []
        self.closed, self.error, self.failure = False, None, None
        self._closing, self._close_task = False, None
        self.block = None
        self.cancelled = 0
        self.started = asyncio.Event()
        self.branch_started = asyncio.Event()
        self.branch_block = None
        self.completion = "analysis\n</think>\n\nAnswer."
        self.chunks = None
        self.roots = set()
        self.capture_roots = True
        self.feature = lambda job: torch.arange(8).float()
        self.delay = lambda job: 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.owner_thread = self.executor.submit(get_ident).result()
        lm = SimpleNamespace(caps={"logits_output": True}, lora_a_tensors={}, lora_b_tensors={},
            select_outputs=lambda ids: None,
            forward=lambda query, params: torch.tensor([[[2.0, 0.0] + [-1.0] * 24]]))
        self.generator = SimpleNamespace(model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=8, max_position_embeddings=65536, eos_token_id_list=[99]),
            fwd_modules=[(lm, None, None)]), max_total_tokens=65536, num_draft_tokens=0,
            max_batch_size=4, recurrent_checkpoint_interval=256)

    def enqueue(self, handle):
        if handle.job.readout_root is not None:
            handle.job.readout_root.attach(handle.job)
        handle.task = asyncio.create_task(self.process(handle.job))
        self.jobs.add(handle.job)

    async def enqueue_async(self, handle):
        self.enqueue(handle)

    async def run_on_owner(self, function, *args, **kwargs):
        if self._close_task is not None:
            raise RuntimeError("Engine is closing")
        future = asyncio.wrap_future(self.executor.submit(function, *args, **kwargs))
        return await self._wait(future)

    async def _wait(self, future):
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            while not future.done():
                try:
                    await asyncio.shield(future)
                except (asyncio.CancelledError, Exception):
                    pass
            if not future.cancelled():
                future.exception()
            raise

    async def process(self, job):
        self.started.set()
        try:
            await asyncio.sleep(self.delay(job))
            if job.readout_root is not None:
                self.branch_started.set()
                if self.branch_block is not None:
                    await self.branch_block.wait()
            if self.block is not None:
                await self.block.wait()
            if self.failure is not None:
                raise self.failure
            job.is_finished = True
            self.completed.append(job)
            job.cached_tokens = job.readout_root.ids.shape[-1] if job.readout_root is not None else 0
            job.forwarded_tokens = job.input_ids.numel() - job.cached_tokens
            result = {"eos": True, "job": job, "features": self.feature(job),
                    "stage": "readout" if job.options["readout"] else "streaming",
                    "cached_tokens": job.cached_tokens, "forwarded_tokens": job.forwarded_tokens,
                    "new_tokens": 3, "full_completion": self.completion,
                    "text": self.completion if self.chunks is None else "",
                    "eos_reason": "max_new_tokens", "sequence_ids": job.input_ids.clone() if job.options["readout"]
                    else torch.cat((job.input_ids, torch.tensor([[165]])), dim=-1)}
            if job.options["readout_root_tokens"] is not None:
                result["readout_root"] = Root(self, job) if self.capture_roots else None
                if not self.capture_roots:
                    result["readout_fallback"] = "Fixture root capacity exhausted"
            return result
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            if job.readout_root is not None:
                root = job.readout_root
                root.children.discard(job)
                if root.closed and not root.children:
                    self.roots.discard(root)
            self.jobs.discard(job)

    async def close(self):
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await self._wait(self._close_task)

    async def _close(self):
        await self._wait(asyncio.wrap_future(self.executor.submit(lambda: None)))
        self.closed = True
        self.executor.shutdown(wait=True)


@pytest.fixture
def native(tmp_path, monkeypatch):
    for name, members in {
        "exllamav3": {}, "exllamav3.generator": {},
        "exllamav3.generator.job": {"Job": Job},
        "exllamav3.generator.async_generator": {"AsyncJob": AsyncJob},
        "exllamav3.generator.sampler": {"ComboSampler": lambda **kwargs: kwargs},
        "exllamav3.constants": {"PAGE_SIZE": 256},
    }.items():
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    geometry = dict(hidden_size=8, projection_size=4, query_adapter_size=3, key_adapter_size=2)
    bundle = SharedDecisionHead(**geometry)
    save_file({key: torch.zeros_like(value) for key, value in bundle.state_dict().items()},
              tmp_path / "head.safetensors")
    (tmp_path / "config.json").write_text(json.dumps(dict(
        geometry, max_options=20, primitives=list(Primitive), encoding_version=VERSION,
        head_id="test-head", release_date="2026-10-12")))
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {p: 2.0 for p in Primitive}}))
    head = ServingHead(tmp_path, device="cpu", calibration="calibrated")
    owner = Engine()
    service = GeneratorService(owner, Tokenizer(), head, model_name="test-model")
    service.bundle_directory = tmp_path
    try:
        yield service, owner
    finally:
        owner.executor.shutdown(wait=True)


def body(**overrides):
    return {"state": {"message": "Charged twice", "history": ["paid", "paid"]},
            "questions": {
                "pick": {"type": "choice", "criteria": {"refund": "Return funds", "wait": "Wait"}},
                "check": {"type": "noul", "criteria": {"true": "Duplicate", "false": "Not duplicate"}},
                "rank": {"type": "score", "criteria": [{"level": "low"}, {"level": "high"}]}},
            **overrides}


def decision(**overrides):
    return {"model": "test-model", "primitive": "noul", "messages": [{"role": "user", "content": "Charged twice"}],
            "candidates": [{"id": "yes", "text": "Yes"}, {"id": "no", "text": "No"}], **overrides}


def job_text(job):
    return "".join(chr(value - 64) for value in job.input_ids[0].tolist())
