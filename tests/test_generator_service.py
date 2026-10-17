import asyncio
import base64
import io

from PIL import Image
import pytest

from fuzev.service_contracts import GenerationRequest, ServiceDecisionRequest
from fuzev.api import APIError
from fuzev.encoding import candidate_suffix, query_suffix
from fuzev_exllamav3.generator_service import GeneratorService, build_generator_service, chat_message
from fuzev_exllamav3.reasoning import ReasoningSplitter
from conftest import decision, job_text


@pytest.mark.parametrize("text,thinking,expected", [
    ("plain", False, {"role": "assistant", "content": "plain", "reasoning": None}),
    ("<think>analysis</think> answer", False, {"role": "assistant", "content": " answer", "reasoning": "analysis"}),
    ("analysis</think>answer", True, {"role": "assistant", "content": "answer", "reasoning": "analysis"}),
    ("unfinished", True, {"role": "assistant", "content": None, "reasoning": "unfinished"}),
    ("<think>unfinished", False, {"role": "assistant", "content": None, "reasoning": "unfinished"}),
    (" analysis \n</think>\n answer ", True,
     {"role": "assistant", "content": "\n answer ", "reasoning": " analysis \n"}),
    ("<think></think>", True, {"role": "assistant", "content": None, "reasoning": None}),
    ("a</think>b", False, {"role": "assistant", "content": "ab", "reasoning": None}),
])
def test_chat_reasoning(text, thinking, expected):
    assert chat_message(text, thinking=thinking) == expected
    splitter = ReasoningSplitter(thinking=thinking)
    parts = [part for char in text for part in splitter.feed(char)] + splitter.feed("", final=True)
    assert ("".join(part.get("reasoning", "") for part in parts) or None) == expected["reasoning"]
    assert ("".join(part.get("content", "") for part in parts) or None) == expected["content"]


def test_real_head_golden_and_compact_decision(native):
    service, owner = native
    result = asyncio.run(service.decide(ServiceDecisionRequest(**decision()), request_id="direct"))
    assert result["head_id"] == "test-head"
    assert result["selected_id"] == "yes"
    assert result["calibration_id"] == "calibrated"
    assert [row["logit"] for row in result["scores"]] == [2.0, 0.0]
    assert [row["raw_probability"] for row in result["scores"]] == pytest.approx([0.880797078, 0.119202922])
    assert [row["calibrated_probability"] for row in result["scores"]] == pytest.approx([0.731058579, 0.268941421])
    assert set(result) == {"id", "model", "primitive", "head_id", "calibration_id", "selected_id", "scores", "execution"}
    assert result["execution"] == {
        "input_tokens": sum(job.input_ids.numel() for job in owner.completed)
                        - 2 * owner.completed[0].input_ids.numel(),
        "submitted_tokens": sum(job.input_ids.numel() for job in owner.completed),
        "cached_tokens": sum(job.cached_tokens for job in owner.completed),
        "forwarded_tokens": sum(job.forwarded_tokens for job in owner.completed),
        "completed_root_branches": sum(job.readout_root is not None for job in owner.completed)}
    assert owner.jobs == {"external-generation"}
    repeated = asyncio.run(service.decide(ServiceDecisionRequest(**decision())))
    assert repeated["scores"] == result["scores"]
    assert [row["id"] for row in result["scores"]] == ["yes", "no"]


def test_generation_defaults_template_overrides_and_prefix(native):
    service, owner = native
    async def run():
        request = GenerationRequest(model="test-model", messages=[{"role": "user", "content": "Hello"}])
        result = await service.generate(request)
        assert result["choices"][0]["message"] == {"role": "assistant", "content": "\n\nAnswer.", "reasoning": "analysis\n"}
        assert set(result) == {"id", "model", "object", "created", "choices", "usage"}
        options = owner.completed[-1].options
        assert "seed" not in options and "sampler" not in options
        owner.completion = "Plain answer"
        result = await service.generate(request.model_copy(update={
            "max_tokens": 10, "seed": 123, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}, "retain_decision_prefix": True}))
        assert result["choices"][0]["message"] == {"role": "assistant", "content": "Plain answer", "reasoning": None}
        assert "<think>" not in job_text(owner.completed[-1])
        assert owner.completed[-1].options["sampler"] == {"temperature": 0.0}
        prefix = service.prefixes.get(result["prefix_id"])
        owner.completed.clear()
        await service.decide(ServiceDecisionRequest(**decision(
            messages=None, prefix_id=prefix.id, additional_context="New evidence")))
        candidates = decision()["candidates"]
        query = prefix.token_ids + service.preparer.encode(query_suffix("", candidates, "New evidence"))
        assert {tuple(job.input_ids[0].tolist()) for job in owner.completed} == {
            query, *(query + service.preparer.encode(candidate_suffix(candidate)) for candidate in candidates)}
        assert service.release_prefix(prefix.id)
        with pytest.raises(APIError) as error:
            service.release_prefix(prefix.id)
        assert error.value.status == 404
        await service.close()
        await service.close()
        assert not owner.closed and owner.jobs == {"external-generation"}
    asyncio.run(run())


def image_input(data_url=True):
    buffer = io.BytesIO()
    Image.new("RGBA", (2, 3), (80, 120, 160, 255)).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + payload if data_url else payload


def test_decision_on_retained_image_prefix(native):
    from types import SimpleNamespace
    service, owner = native
    images = []
    async def embed(tokenizer, rgb):
        images.append((rgb.mode, rgb.size))
        return SimpleNamespace(text_alias=f"<image:{len(images)}>")
    service.image_embeddings = embed
    async def run():
        generated = await service.generate(GenerationRequest(
            model="test-model", prompt="Describe", image=image_input(), retain_decision_prefix=True))
        result = await service.decide(ServiceDecisionRequest(**decision(messages=None, prefix_id=generated["prefix_id"],
                image=image_input(), additional_context="Compare")))
        assert images == [("RGB", (2, 3)), ("RGB", (2, 3))]
        assert result["selected_id"] == "yes"
        assert [row["logit"] for row in result["scores"]] == [2.0, 0.0]
        await service.close()
        assert not owner.closed
    asyncio.run(run())


def test_service_close_drains_only_service_jobs(native):
    service, owner = native
    async def run():
        owner.block = asyncio.Event()
        request = asyncio.create_task(service.generate(GenerationRequest(model="test-model", prompt="Hello")))
        await owner.started.wait()
        await service.close()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert owner.jobs == {"external-generation"} and not owner.closed
    asyncio.run(asyncio.wait_for(run(), 3))


def test_generation_does_not_use_decision_head(native, monkeypatch):
    service, engine = native
    def score(*args, **kwargs):
        raise AssertionError("Generation used the decision head")
    monkeypatch.setattr(service.head, "score", score)
    result = asyncio.run(service.generate(GenerationRequest(prompt="Hello")))
    assert result["choices"][0]["text"] == engine.completion
    assert all(not job.options["readout"] for job in engine.completed)


def test_operator_chat_defaults_do_not_change_decision_encoding(native):
    service, engine = native
    configured = GeneratorService(engine, service.tokenizer, service.head, model_name=service.model_name,
                                  chat_template_kwargs={"chat_template": "{{ messages[0].content }}",
                                                        "reasoning_effort": "high", "enable_thinking": False})
    async def run():
        request = ServiceDecisionRequest(**decision())
        plain = await service.decide(request)
        original = {tuple(job.input_ids[0].tolist()) for job in engine.completed}
        engine.completed.clear()
        customized = await configured.decide(request)
        assert customized["scores"] == plain["scores"]
        assert {tuple(job.input_ids[0].tolist()) for job in engine.completed} == original
        generated = await configured.generate(GenerationRequest(messages=[{"role": "user", "content": "Custom"}]))
        assert generated["usage"]["prompt_tokens"] == len("Custom")
    asyncio.run(run())


def test_generation_image_uses_model_template(native):
    from types import SimpleNamespace
    service, engine = native
    service.chat_template_kwargs = {"chat_template": "{{ messages[0].content }}<model-assistant>"}
    service.image_embeddings = lambda tokenizer, image: SimpleNamespace(text_alias="<start_of_image>")
    async def run():
        generated = await service.generate(GenerationRequest(
            messages=[{"role": "user", "content": "Describe"}], image=image_input()))
        assert generated["usage"]["prompt_tokens"] == len("<start_of_image>\nDescribe<model-assistant>")
        assert job_text(engine.completed[0]) == "<start_of_image>\nDescribe<model-assistant>"
    asyncio.run(run())


def test_cache_budget_limits_context(native):
    service, engine = native
    engine.generator.max_total_tokens = 300
    engine.generator.num_draft_tokens = 3
    bounded = GeneratorService(engine, service.tokenizer, service.head, model_name=service.model_name)
    assert bounded.context_tokens == 296
    with pytest.raises(APIError) as error:
        asyncio.run(bounded.generate(GenerationRequest(prompt="x" * 297)))
    assert error.value.status == 422 and "296" in str(error.value)
    assert not engine.completed


def test_encoding_mismatch_names_both_versions(native):
    import json
    service, engine = native
    config = service.bundle_directory / "config.json"
    metadata = json.loads(config.read_text())
    metadata["encoding_version"] = "another-encoding-v2"
    config.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="another-encoding-v2.*fuzev-candidate-branches-v1"):
        build_generator_service(engine, service.tokenizer, model_name=service.model_name,
                                head_directory=service.bundle_directory, head_device="cpu")
