"""Real ExLlamaV3 Job interface used by the plugin."""

import asyncio
from types import SimpleNamespace

import pytest
import torch

from fuzev.service_contracts import GenerationRequest


def test_service_builds_readout_and_retained_generation_jobs():
    pytest.importorskip("exllamav3")
    from fuzev_exllamav3.generator_service import GeneratorService

    config = SimpleNamespace(max_position_embeddings=64, eos_token_id_list=[0])
    generator = SimpleNamespace(error=None, generator=SimpleNamespace(
        model=SimpleNamespace(config=config), max_total_tokens=64, num_draft_tokens=0))
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: torch.tensor([[1, 2]]),
                                hf_render_chat_template=lambda *args, **kwargs: "x",
                                generation_config_dict={})
    service = GeneratorService(generator, tokenizer, model_name="m")
    jobs = []

    async def run(job):
        jobs.append(job)
        return {"features": torch.zeros(1)}

    service._run = run

    async def case():
        await service._readout([1, 2, 3], [], checkpoint=False)
        job, _, _ = await service._generation_job(
            GenerationRequest(prompt="x", max_tokens=1, retain_decision_prefix=True))
        return job

    generation = asyncio.run(case())
    readout, = jobs
    assert readout.readout is True and readout.max_new_tokens == 0
    assert readout.checkpoint_enabled is False
    assert generation.retain_decision_prefix is True
