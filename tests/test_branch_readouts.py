import asyncio
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from fuzev.api import APIError
from fuzev.service_contracts import GenerationRequest, ServiceDecisionRequest
from conftest import AsyncJob, Root, decision
from test_generator_service import image_input


def choice_request():
    return ServiceDecisionRequest(**decision(primitive="choice", candidates=[
        {"id": str(index), "text": char * (index + 1)} for index, char in enumerate("abcdefg")]))


def assert_idle(service, owner):
    assert not owner.roots and not service.jobs
    assert owner.jobs == {"external-generation"} and not owner.closed


@pytest.mark.parametrize("window", [1, 2, 4, 8])
@pytest.mark.parametrize("capture", [True, False])
def test_ordered_whole_slate_scores_and_native_work_totals(native, window, capture):
    service, owner = native
    owner.generator.max_batch_size = window
    owner.capture_roots = capture
    owner.delay = lambda job: (job.input_ids[0, -1].item() % 3) * 0.001 if job.readout_root else 0
    owner.feature = lambda job: torch.tensor([job.input_ids[0, -1].item() % 16] + [0.] * 7)
    with torch.no_grad():
        service.head.bundle.scorer.query.bias.fill_(1)
        service.head.bundle.scorer.key.weight[0, 0] = 2
    request = choice_request()
    result = asyncio.run(service.decide(request))
    values = torch.arange(1, 8).float()
    logits = torch.tensor([2., 0.] + [-1.] * 5) + (values - values.mean())
    assert [row["id"] for row in result["scores"]] == [candidate.id for candidate in request.candidates]
    assert [row["logit"] for row in result["scores"]] == pytest.approx(logits.tolist())
    assert [row["raw_probability"] for row in result["scores"]] == pytest.approx(logits.softmax(0).tolist())
    assert [row["calibrated_probability"] for row in result["scores"]] == pytest.approx((logits / 2).softmax(0).tolist())
    assert result["selected_id"] == "6"
    logical = sum(job.input_ids.numel() for job in owner.completed)
    reused = len(request.candidates) * owner.completed[0].input_ids.numel() if capture else 0
    assert result["execution"] == {"input_tokens": logical - len(request.candidates) * owner.completed[0].input_ids.numel(),
                                   "submitted_tokens": logical, "cached_tokens": reused,
                                   "forwarded_tokens": logical - reused,
                                   "completed_root_branches": sum(job.readout_root is not None
                                                                  for job in owner.completed)}
    assert_idle(service, owner)


def test_cancelled_branch_window_drains_root_and_continues_serving(native):
    service, owner = native

    async def run():
        owner.branch_block = asyncio.Event()
        pending = asyncio.create_task(service.decide(choice_request()))
        await owner.branch_started.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert_idle(service, owner)
        owner.branch_block.set()
        assert (await service.decide(ServiceDecisionRequest(**decision())))["selected_id"] == "yes"
        assert_idle(service, owner)

    asyncio.run(asyncio.wait_for(run(), 5))


def test_failed_branch_window_releases_root_and_other_jobs_survive(native):
    service, owner = native
    feature = owner.feature

    def fail(job):
        if job.readout_root is not None:
            raise RuntimeError("Tail readout failed")
        return feature(job)

    async def run():
        owner.feature = fail
        with pytest.raises(ExceptionGroup) as error:
            await service.decide(choice_request())
        assert any(str(item) == "Tail readout failed" for item in error.value.exceptions)
        assert_idle(service, owner)
        owner.feature = feature
        assert (await service.decide(ServiceDecisionRequest(**decision())))["selected_id"] == "yes"
        assert_idle(service, owner)

    asyncio.run(run())


def test_root_handoff_cleanup_failure_disposes_owned_terminal_result(native, monkeypatch):
    service, owner = native
    cleanup = AsyncJob.cancel

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()

        async def interrupt(handle):
            await cleanup(handle)
            entered.set()
            await release.wait()

        monkeypatch.setattr(AsyncJob, "cancel", interrupt)
        pending = asyncio.create_task(service._readout(torch.tensor([[1, 2]]), (), readout_root_tokens=2))
        await entered.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert_idle(service, owner)

    asyncio.run(asyncio.wait_for(run(), 5))


def test_vision_cancellation_keeps_pixels_alive_until_owner_finishes(native):
    service, owner = native
    entered, release = Event(), Event()
    pixels = []

    def embed(tokenizer, rgb):
        entered.set()
        assert release.wait(5)
        pixels.append(rgb.getpixel((0, 0)))
        return object()

    service.image_embeddings = embed

    async def run():
        pending = asyncio.create_task(service.generate(GenerationRequest(prompt="Describe", image=image_input())))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert pixels == [(80, 120, 160)]
        assert_idle(service, owner)

    asyncio.run(asyncio.wait_for(run(), 10))


def test_async_vision_adapter_keeps_loop_affinity_and_can_dispatch_model_work(native):
    service, owner = native

    async def run():
        ready = asyncio.Event()

        async def embed(tokenizer, rgb):
            ready.set()
            await ready.wait()
            return await owner.run_on_owner(lambda: SimpleNamespace(text_alias=str(rgb.getpixel((0, 0)))))

        service.image_embeddings = embed
        result = await service._image(image_input())
        assert result.text_alias == "(80, 120, 160)"
        assert_idle(service, owner)

    asyncio.run(run())


def test_score_cancellation_drains_owner_before_head_shutdown(native, monkeypatch):
    service, owner = native
    entered, release = Event(), Event()
    score = service.head.score

    def blocking(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return score(*args, **kwargs)

    monkeypatch.setattr(service.head, "score", blocking)

    async def run():
        pending = asyncio.create_task(service.decide(choice_request()))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            closing = asyncio.create_task(service.close())
            await asyncio.sleep(0)
            assert not closing.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await closing
        assert service.head.bundle is None
        assert_idle(service, owner)

    asyncio.run(asyncio.wait_for(run(), 10))


def test_head_releases_after_attached_engine_already_retired(native):
    service, owner = native

    async def run():
        await owner.close()
        await service.close()
        await service.close()
        assert service.head.bundle is None
        assert owner.closed and not service.jobs and not owner.roots

    asyncio.run(run())


def test_shutdown_before_score_admission_returns_unavailable(native, monkeypatch):
    service, owner = native
    entered, release = Event(), Event()
    close = Root.close

    def pause(root):
        close(root)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(Root, "close", pause)

    async def run():
        pending = asyncio.create_task(service.decide(choice_request()))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            closing = asyncio.create_task(service.close())
            await asyncio.sleep(0)
        finally:
            release.set()
        with pytest.raises(APIError) as error:
            await pending
        assert error.value.status == 503
        await closing
        assert_idle(service, owner)

    asyncio.run(asyncio.wait_for(run(), 10))


def test_owner_score_failure_is_local_to_request(native, monkeypatch):
    service, owner = native
    score = service.head.score

    def fail(*args, **kwargs):
        raise RuntimeError("Head failed")

    async def run():
        monkeypatch.setattr(service.head, "score", fail)
        with pytest.raises(RuntimeError, match="Head failed"):
            await service.decide(choice_request())
        assert owner.error is None
        assert_idle(service, owner)
        monkeypatch.setattr(service.head, "score", score)
        assert (await service.decide(ServiceDecisionRequest(**decision())))["selected_id"] == "yes"

    asyncio.run(run())


def test_shared_warmup_native_totals_are_attributed_once(native):
    service, owner = native
    requests = [ServiceDecisionRequest(**decision(messages=[{"role": "user", "content": "Details " * 100}]))
                for _ in range(2)]

    async def run():
        submit = await service.prepare_decisions(requests, question_keys=("a", "b"))
        results = await asyncio.gather(*(submit(request, str(index)) for index, request in enumerate(requests)))
        assert sum(result["execution"]["submitted_tokens"] for result in results) == sum(
            job.input_ids.numel() for job in owner.completed)
        for key in ("cached_tokens", "forwarded_tokens"):
            assert sum(result["execution"][key] for result in results) == sum(
                getattr(job, key) for job in owner.completed)
        state_tokens = len(service.preparer.base(requests[0].messages))
        query_tokens = [job.input_ids.numel() for job in owner.completed
                        if job.options["readout_root_tokens"] is not None]
        assert sum(result["execution"]["input_tokens"] for result in results) == (
            sum(job.input_ids.numel() for job in owner.completed)
            - 2 * sum(query_tokens) - 2 * state_tokens)
        assert_idle(service, owner)

    asyncio.run(run())
