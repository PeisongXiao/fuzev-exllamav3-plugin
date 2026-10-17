import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
import torch

from fuzev.encoding import LIST_AWARE_VERSION
from fuzev_exllamav3.generator_service import build_generator_service
from fuzev_exllamav3.server import create_app
from conftest import body, decision, job_text
from test_generator_service import image_input


@pytest.fixture
def v2_native(native, request):
    old, engine = native
    directory = old.bundle_directory
    metadata = json.loads((directory / "config.json").read_text())
    del metadata["primitives"], metadata["max_options"]
    maximum = {"text": 255, "image_text": 26, **getattr(request, "param", {})}
    metadata.update(
        schema="fuzev-shared-head-package-v2", architecture_version="fuzev-shared-decision-head-v1",
        encoding_version=LIST_AWARE_VERSION,
        baseline={"labelled": "query-label-logits", "label_free": "zero"},
        features={name: {"tap": "final-pre-lm", "pooling": "last-token"} for name in ("query", "candidate")},
        capacities={
            "choice": {modality: {"implemented_max": count, "validated_max": None}
                       for modality, count in maximum.items()},
            "ordinal": {"text": {"implemented_max": 41, "validated_max": None}},
            "noul": {"text": {"implemented_max": 2, "validated_max": None},
                     "image_text": {"implemented_max": 2, "validated_max": None}},
        }, residual_strength={"text": 1.0, "image_text": 1.0},
    )
    strata = []
    for primitive, supported in metadata["capacities"].items():
        for modality, capacity in supported.items():
            maximum = capacity["implemented_max"]
            strata.append(dict(primitive=primitive, modality=modality, k_min=2,
                               k_max=min(maximum, 26), temperature=2.0))
            if maximum > 26:
                strata.append(dict(primitive=primitive, modality=modality, k_min=27,
                                   k_max=maximum, temperature=3.0))
    (directory / "config.json").write_text(json.dumps(metadata))
    (directory / "calibration.json").write_text(json.dumps({"strata": strata}))
    old.head.close()
    service = build_generator_service(engine, old.tokenizer, model_name=old.model_name,
                                      head_directory=directory, head_device="cpu", calibration="calibrated")
    return service, engine


def client(service):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(service)), base_url="http://fixture")


def options(count):
    return [{"id": str(index), "text": f"Option {index}"} for index in range(count)]


@pytest.fixture
def partial_v2_native(v2_native, native, request):
    old, engine = v2_native
    directory = native[0].bundle_directory
    strata = [dict(primitive="choice", modality="text", k_min=count, k_max=count, temperature=temperature)
              for count, temperature in ((2, 2.0), (4, 1.0))]
    (directory / "calibration.json").write_text(json.dumps({"strata": strata}))
    old.head.close()
    service = build_generator_service(engine, old.tokenizer, model_name=old.model_name,
        head_directory=directory, head_device="cpu", calibration=request.param)
    service.image_embeddings = lambda tokenizer, rgb: SimpleNamespace(text_alias="<image:1>")
    return service, engine


@pytest.mark.parametrize("partial_v2_native", ["calibrated", "raw"], indirect=True)
def test_sparse_calibration_responses_across_mixed_requests(partial_v2_native):
    service, engine = partial_v2_native
    requested = service.head.calibration_id
    cases = [("choice", 2, None, "calibrated", 2.0),
             ("choice", 3, None, "raw", 1.0),
             ("choice", 4, None, "calibrated", 1.0),
             ("choice", 27, None, "raw", 1.0),
             ("ordinal", 2, None, "raw", 1.0),
             ("choice", 2, image_input(), "raw", 1.0)]

    async def run():
        async with client(service) as http:
            responses = await asyncio.gather(*(http.post("/v1/decisions", json=decision(
                primitive=primitive, candidates=options(count), image=image))
                for primitive, count, image, _, _ in cases))
            repeated = await http.post("/v1/decisions", json=decision(primitive="choice", candidates=options(2)))
        for response, (primitive, count, _, calibration_id, temperature) in zip(responses, cases):
            assert response.status_code == 200, response.text
            result = response.json()
            if requested == "raw":
                calibration_id, temperature = "raw", 1.0
            baseline = torch.tensor([2.0, 0.0] + [-1.0] * (count - 2)) if count <= 26 else torch.zeros(count)
            assert result["primitive"] == primitive and result["calibration_id"] == calibration_id
            assert result["selected_id"] == "0"
            assert [row["id"] for row in result["scores"]] == [str(index) for index in range(count)]
            assert [row["logit"] for row in result["scores"]] == baseline.tolist()
            assert [row["raw_probability"] for row in result["scores"]] == pytest.approx(baseline.softmax(0).tolist())
            assert [row["calibrated_probability"] for row in result["scores"]] == pytest.approx(
                (baseline / temperature).softmax(0).tolist())
        assert repeated.status_code == 200, repeated.text
        assert repeated.json()["calibration_id"] == requested
        assert repeated.json()["scores"] == responses[0].json()["scores"]
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("count", [2, 26, 27, 255])
def test_v2_decision_response_golden_at_route_boundaries(v2_native, count):
    service, engine = v2_native

    async def run():
        async with client(service) as http:
            response = await http.post("/v1/decisions", json=decision(primitive="choice", candidates=options(count)))
        assert response.status_code == 200, response.text
        result = response.json()
        baseline = torch.tensor([2.0, 0.0] + [-1.0] * (count - 2)) if count <= 26 else torch.zeros(count)
        assert result["selected_id"] == "0"
        assert result["primitive"] == "choice" and result["calibration_id"] == "calibrated"
        assert [row["id"] for row in result["scores"]] == [str(index) for index in range(count)]
        assert [row["logit"] for row in result["scores"]] == baseline.tolist()
        assert [row["raw_probability"] for row in result["scores"]] == pytest.approx(baseline.softmax(0).tolist())
        temperature = 2 if count <= 26 else 3
        assert [row["calibrated_probability"] for row in result["scores"]] == pytest.approx(
            (baseline / temperature).softmax(0).tolist())
        assert result["execution"]["submitted_tokens"] == sum(job.input_ids.numel() for job in engine.completed)
        query = job_text(next(job for job in engine.completed if "Candidate under consideration" not in job_text(job)))
        for index in range(count):
            assert f"Option {index}" in query
        if count > 26:
            assert "\n27. Option 26" in query
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("primitive,count,status", [("choice", 256, 422), ("ordinal", 41, 200), ("ordinal", 42, 422)])
def test_v2_api_uses_each_primitive_capacity(v2_native, primitive, count, status):
    service, engine = v2_native

    async def run():
        async with client(service) as http:
            direct = await http.post("/v1/decisions", json=decision(primitive=primitive, candidates=options(count)))
            kind = "score" if primitive == "ordinal" else primitive
            criteria = list(range(count)) if kind == "score" else {str(index): None for index in range(count)}
            mapped = await http.post("/v1/systemone", json=body(questions={"q": {"type": kind, "criteria": criteria}}))
        assert direct.status_code == mapped.status_code == status
        if status == 422:
            assert direct.json()["detail"][0]["loc"] == ["body", "candidates"]
            assert mapped.json()["detail"][0]["loc"] == ["body", "questions", "q", kind, "criteria"]
            assert not engine.completed
        else:
            assert len(mapped.json()["answers"]["q"]["probabilities"]) == count
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("v2_native", [{"image_text": 26}, {"image_text": 31}], indirect=True)
def test_v2_image_capacity_includes_retained_payload(v2_native, retained):
    service, engine = v2_native
    service.image_embeddings = lambda tokenizer, rgb: SimpleNamespace(text_alias="<image:1>")

    async def run():
        maximum = service.head.capacities["choice"]["image_text"]["implemented_max"]
        if retained:
            prefix = service.prefixes.retain(service.preparer.base([{"role": "user", "content": "Image"}]),
                                             [SimpleNamespace(text_alias="<image:1>")])
            context = dict(messages=None, prefix_id=prefix.id)
        else:
            context = dict(image=image_input())
        async with client(service) as http:
            if maximum == 26:
                rejected = await http.post("/v1/decisions", json=decision(
                    primitive="choice", candidates=options(27), **context))
                assert rejected.status_code == 422
                assert rejected.json()["detail"][0]["loc"] == ["body", "candidates"]
                assert not engine.completed
            accepted = await http.post("/v1/decisions", json=decision(
                primitive="choice", candidates=options(maximum), **context))
        assert accepted.status_code == 200, accepted.text
        baseline = [2.0, 0.0] + [-1.0] * (maximum - 2) if maximum <= 26 else [0.0] * maximum
        assert [row["logit"] for row in accepted.json()["scores"]] == baseline
        if maximum > 26:
            assert any("\n27. Option 26" in job_text(job) for job in engine.completed)
        await service.close()

    asyncio.run(run())


def test_v2_systemone_preflights_later_questions_before_images(v2_native):
    service, engine = v2_native
    images = []
    service.image_embeddings = lambda *args: images.append(args)

    async def run():
        async with client(service) as http:
            response = await http.post("/v1/systemone", json=body(image=image_input(), questions={
                "first": {"type": "noul"},
                "later": {"type": "choice", "criteria": {str(index): None for index in range(27)}}}))
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "questions", "later", "choice", "criteria"]
        assert not images and not engine.completed
        await service.close()

    asyncio.run(run())


def test_v2_systemone_whole_slate_probabilities_match_direct(v2_native):
    service, engine = v2_native

    async def run():
        async with client(service) as http:
            direct = await http.post("/v1/decisions", json=decision(primitive="choice", candidates=options(255)))
            mapped = await http.post("/v1/systemone", json=body(questions={
                "q": {"type": "choice", "criteria": {str(index): f"Option {index}" for index in range(255)}}}))
        assert direct.status_code == mapped.status_code == 200
        probabilities = mapped.json()["answers"]["q"]["probabilities"]
        assert list(probabilities.values()) == [row["calibrated_probability"] for row in direct.json()["scores"]]
        assert sum(probabilities.values()) == pytest.approx(1)
        assert mapped.json()["answers"]["q"]["choice"] == direct.json()["selected_id"]
        await service.close()

    asyncio.run(run())


@pytest.mark.parametrize("v2_native", [{"text": 300}], indirect=True)
def test_candidate_limit_comes_from_loaded_head(v2_native):
    service, engine = v2_native

    async def run():
        async with client(service) as http:
            accepted = await http.post("/v1/decisions", json=decision(primitive="choice", candidates=options(300)))
            rejected = await http.post("/v1/decisions", json=decision(primitive="choice", candidates=options(301)))
        assert accepted.status_code == 200, accepted.text
        scores = accepted.json()["scores"]
        assert len(scores) == 300
        assert [row["calibrated_probability"] for row in scores] == pytest.approx([1 / 300] * 300)
        assert rejected.status_code == 422
        assert rejected.json()["detail"] == [{"loc": ["body", "candidates"],
                                              "msg": "This head supports at most 300 options", "type": "value_error"}]
        await service.close()

    asyncio.run(run())
