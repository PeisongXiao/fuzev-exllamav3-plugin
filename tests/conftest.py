"""CPU engine boundary, real native service, shared renderer and decision head."""

import asyncio
import json
import sys
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
    def __init__(self, input_ids, *, readout=False, checkpoint=True, **kwargs):
        kwargs.update(readout=readout, checkpoint=checkpoint)
        self.input_ids, self.options = input_ids, kwargs
        self.embeddings = kwargs.get("embeddings", [])
        self.is_finished = False
        self.sequences = [SimpleNamespace(kv_position=input_ids.shape[-1])]


class AsyncJob:
    @classmethod
    def from_job(cls, owner, job):
        handle = cls()
        handle.owner, handle.job = owner, job
        handle.task = asyncio.create_task(owner.process(job))
        owner.jobs.add(job)
        return handle

    async def __aiter__(self):
        try:
            if not self.job.options["readout"] and self.owner.chunks is not None:
                for text in self.owner.chunks:
                    yield {"text": text, "eos": False}
                    await asyncio.sleep(0)
            yield await asyncio.shield(self.task)
        finally:
            if self.task.done():
                self.owner.jobs.discard(self.job)

    async def cancel(self):
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        self.owner.jobs.discard(self.job)


class Engine:
    def __init__(self):
        self.jobs, self.completed = {"external-generation"}, []
        self.closed, self.error, self.failure = False, None, None
        self.block = None
        self.cancelled = 0
        self.started = asyncio.Event()
        self.completion = "analysis\n</think>\n\nAnswer."
        self.chunks = None
        lm = SimpleNamespace(caps={"logits_output": True},
            forward=lambda query, params: torch.tensor([[[2.0, 0.0] + [-1.0] * 24]]))
        self.generator = SimpleNamespace(model=SimpleNamespace(
            config=SimpleNamespace(hidden_size=8, max_position_embeddings=65536, eos_token_id_list=[99]),
            fwd_modules=[(lm, None, None)]), max_total_tokens=65536, num_draft_tokens=0)

    async def process(self, job):
        self.started.set()
        try:
            await asyncio.sleep(0)
            if self.block is not None:
                await self.block.wait()
            if self.failure is not None:
                raise self.failure
            job.is_finished = True
            self.completed.append(job)
            return {"eos": True, "job": job, "features": torch.arange(8).float(),
                    "new_tokens": 3, "full_completion": self.completion,
                    "text": self.completion if self.chunks is None else "",
                    "eos_reason": "max_new_tokens", "sequence_ids": torch.cat(
                        (job.input_ids, torch.tensor([[165]])), dim=-1)}
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.jobs.discard(job)

    async def close(self):
        self.closed = True


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
        geometry, max_options=20, primitives=list(Primitive), encoding_version=VERSION, release_date="2026-10-12")))
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {p: 2.0 for p in Primitive}}))
    head = ServingHead(tmp_path, device="cpu", calibration="calibrated")
    owner = Engine()
    service = GeneratorService(owner, Tokenizer(), head, model_name="test-model")
    service.bundle_directory = tmp_path
    return service, owner


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
