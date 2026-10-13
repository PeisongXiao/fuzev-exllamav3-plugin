import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
import httpx
import pytest

from fuzev.api import models_response
from fuzev.service_contracts import ServiceDecisionRequest
from fuzev_exllamav3.server import create_app
from fuzev_exllamav3.generator_service import build_generator_service
from fuzev_exllamav3.systemone import HEADER, add_systemone_routes
from conftest import body, decision, job_text
from test_generator_service import image_input


def client(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture")


def test_three_primitives_match_direct_service_and_evaluated_rendering(native):
    service, owner = native
    async def run():
        direct = await service.decide(ServiceDecisionRequest(**decision()))
        owner.completed.clear()
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", json=body(model="any-model", ignored="accepted"),
                                       headers={HEADER: "sdk.1:_-"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert response.headers[HEADER] == "sdk.1:_-" and result["model"] == service.model_name
        p = direct["scores"][0]["calibrated_probability"]
        assert result["answers"]["check"]["noul"] == p
        assert result["answers"]["pick"]["probabilities"] == {"refund": p, "wait": direct["scores"][1]["calibrated_probability"]}
        assert result["answers"]["rank"]["legend"] == {"0": {"level": "low"}, "1": {"level": "high"}}
        assert result["answers"]["rank"]["score"] == direct["scores"][1]["calibrated_probability"]
        queries = [job_text(job) for job in owner.completed
                   if "Select one candidate." in job_text(job) and "Candidate under consideration" not in job_text(job)]
        assert any("(A) Yes: Duplicate\n(B) No: Not duplicate" in text for text in queries)
        assert all("Question: \nCandidates:" in text for text in queries)
        assert all("message: Charged twice\nhistory:\n  - paid\n  - paid" in text for text in queries)
        assert result["usage"]["input_tokens"] == sum(job.input_ids.numel() for job in owner.completed)
        assert owner.jobs == {"external-generation"}
    asyncio.run(run())


@pytest.mark.parametrize("count,status", [(1, 422), (2, 200), (20, 200), (21, 422)])
def test_limits_come_from_loaded_head(native, count, status):
    service, engine = native
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", json=body(questions={
                "c": {"type": "choice", "criteria": {str(i): "option" for i in range(count)}},
                "s": {"type": "score", "criteria": list(range(count))}}))
        assert response.status_code == status, response.text
        if status == 422:
            assert not engine.completed
        if count > service.head.max_options:
            assert f"at most {service.head.max_options} options" in response.text
    asyncio.run(run())


@pytest.mark.parametrize("payload", [b"{", b"\xff", b"[]", b"{}", b'{"state":NaN,"questions":{"n":{"type":"noul"}}}'])
def test_invalid_json_and_known_fields_have_shared_errors(native, payload):
    service, owner = native
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", content=payload, headers={HEADER: "invalid"})
        assert response.status_code == 422, response.text
        assert isinstance(response.json()["detail"], list)
        assert response.headers[HEADER] == "invalid" and not owner.completed
    asyncio.run(run())


def test_internal_errors_are_not_misclassified_as_client_errors(native):
    service, owner = native
    owner.failure = RuntimeError("bad backend")
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", json=body(), headers={HEADER: "failure"})
        assert response.status_code == 500 and response.headers[HEADER] == "failure"
        assert response.json() == {"detail": "Internal server error"}
        assert owner.jobs == {"external-generation"}
    asyncio.run(run())


def test_standard_images_are_decoded_once_for_concurrent_questions(native):
    service, owner = native
    images = []
    def embed(tokenizer, rgb):
        images.append(rgb.mode)
        return SimpleNamespace(text_alias="<image:1>")
    service.image_embeddings = embed
    async def run():
        async with client(create_app(service)) as http:
            invalid = await http.post("/v1/systemone", json=body(image="not an image"))
            assert invalid.status_code == 422 and not owner.completed
            response = await http.post("/v1/systemone", json=body(image=image_input()))
        assert response.status_code == 200, response.text
        assert invalid.status_code == 422
        assert images == ["RGB"]
        assert all("<image:1>" in job_text(job) for job in owner.completed)
    asyncio.run(run())


def test_cancel_parent_and_continue_serving(native):
    service, owner = native
    async def run():
        owner.block = asyncio.Event()
        app = create_app(service)
        async with client(app) as http:
            pending = asyncio.create_task(http.post("/v1/systemone", json=body(), headers={HEADER: "cancel-me"}))
            await owner.started.wait()
            duplicate = await http.post("/v1/systemone", json=body(), headers={HEADER: "cancel-me"})
            assert duplicate.status_code == 409
            cancelled = await http.post("/v1/requests/cancel-me/cancel")
            assert cancelled.json() == {"id": "cancel-me", "cancel_requested": True}
            response = await pending
            assert response.status_code == 499 and response.headers[HEADER] == "cancel-me"
            assert owner.cancelled > 0 and owner.jobs == {"external-generation"}
            assert (await http.post("/v1/requests/cancel-me/cancel")).status_code == 404
            owner.block.set()
            assert (await http.post("/v1/systemone", json=body())).status_code == 200
            await app.state.fuzev_routes.close()
            assert not service.closed and not owner.closed
    asyncio.run(asyncio.wait_for(run(), 5))


def test_disconnect_drains_jobs(native):
    service, owner = native
    async def run():
        owner.block = asyncio.Event()
        queue, output = asyncio.Queue(), []
        await queue.put({"type": "http.request", "body": json.dumps(body()).encode(), "more_body": False})
        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/v1/systemone", "raw_path": b"/v1/systemone", "root_path": "",
                 "query_string": b"", "headers": [(b"x-typesafe-request-id", b"disconnect")],
                 "client": ("127.0.0.1", 1), "server": ("fixture", 80)}
        async def send(message):
            output.append(message)
        pending = asyncio.create_task(create_app(service)(scope, queue.get, send))
        await owner.started.wait()
        await queue.put({"type": "http.disconnect"})
        await pending
        assert output[0]["status"] == 499
        assert owner.cancelled > 0 and owner.jobs == {"external-generation"}
    asyncio.run(asyncio.wait_for(run(), 3))


def test_existing_application_routes_remain_unchanged(native):
    service, owner = native
    async def run():
        app = FastAPI()
        @app.get("/v1/models")
        async def models():
            return models_response(service.model_name, service.head)
        @app.post("/v1/completions")
        async def ordinary_generation():
            return {"original": True}
        with pytest.raises(ValueError, match="already exists"):
            add_systemone_routes(app, lambda: service, include_models=True)
        routes = add_systemone_routes(app, lambda: service)
        await routes.close()
        async with client(app) as http:
            assert (await http.post("/v1/completions", headers={HEADER: "invalid id"})).json() == {"original": True}
            assert (await http.get("/v1/models")).json()["models"][0]["release_date"] == "2026-10-12"
        assert not service.closed and not owner.closed
    asyncio.run(run())


def test_eight_concurrent_systemone_requests(native):
    service, _ = native
    async def run():
        async with client(create_app(service)) as http:
            responses = await asyncio.gather(*(http.post("/v1/systemone", json=body()) for _ in range(8)))
        assert all(response.status_code == 200 for response in responses)
        assert all(list(response.json()["answers"]) == list(body()["questions"]) for response in responses)
    asyncio.run(run())


def test_known_question_type_is_validated_and_unknown_fields_are_ignored(native):
    service, engine = native
    async def run():
        async with client(create_app(service)) as http:
            invalid = await http.post("/v1/systemone", json=body(questions={
                "rank": {"type": "score", "criteria": "x"}}))
            assert invalid.status_code == 422 and not engine.completed
            response = await http.post("/v1/systemone", json=body(questions={
                "check": {"type": "noul", "future_question_field": True}}, future_field=True))
            assert response.status_code == 200
    asyncio.run(run())


def test_all_question_primitives_validated_before_engine_work(native):
    service, engine = native
    config = service.bundle_directory / "config.json"
    metadata = json.loads(config.read_text())
    metadata["primitives"] = ["choice", "noul"]
    config.write_text(json.dumps(metadata))
    calibration = service.bundle_directory / "calibration.json"
    calibration.write_text(json.dumps({"temperatures": {"choice": 2.0, "noul": 2.0}}))
    service = build_generator_service(engine, service.tokenizer, model_name=service.model_name,
                                      head_directory=service.bundle_directory, head_device="cpu")
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", json=body(image=image_input()))
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "questions", "rank", "type"]
        assert not engine.completed
    asyncio.run(run())


def test_shared_state_preserves_long_state_golden_answers(native):
    from fuzev.systemone import SystemOneRequest, decision_request, state_text
    service, engine = native
    payload = body(state="shared ticket details " * 100)
    async def run():
        request = SystemOneRequest.model_validate(payload)
        direct = {}
        for key, question in request.questions.items():
            direct[key] = await service.decide(decision_request(
                request, question, service.model_name, service.head.head_id, state_text(request.state)))
        engine.completed.clear()
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", json=payload)
        assert response.status_code == 200, response.text
        result = response.json()
        for key, answer in result["answers"].items():
            if answer["type"] == "noul":
                assert answer["noul"] == direct[key]["scores"][0]["calibrated_probability"]
            else:
                assert list(answer["probabilities"].values()) == [
                    row["calibrated_probability"] for row in direct[key]["scores"]]
        assert result["usage"]["input_tokens"] == sum(job.input_ids.numel() for job in engine.completed)
    asyncio.run(run())


def test_real_sdk_cpu_integration(native):
    import typesafe_sdk as sdk
    import httpx2
    service, owner = native
    async def run():
        async with sdk.AsyncTypeSafeClient(api_key="test", base_url="http://fixture",
                transport=httpx2.ASGITransport(app=create_app(service, api_key="test")),
                retry=sdk.RetryPolicy(max_retries=0)) as http:
            models = await http.models.list()
            assert models.models[0].name == service.model_name
            result = await http.system_one(state=body()["state"], questions={
                "pick": sdk.Choice(criteria={"refund": "Return funds", "wait": "Wait"}),
                "check": sdk.Noul(instructions="Was the customer charged twice?"),
                "rank": sdk.Score(criteria=[{"level": "low"}, {"level": "high"}])},
                model="jev-arbitrary", extra_headers={HEADER: "real-sdk"})
            assert result.request_id == "real-sdk"
            assert result.model == service.model_name
            assert result.choices["pick"].choice == "refund"
            assert result.nouls["check"].noul == pytest.approx(0.731058579)
            assert result.scores["rank"].legend == {0: {"level": "low"}, 1: {"level": "high"}}
            assert result.usage.input_tokens == sum(job.input_ids.numel() for job in owner.completed)
        assert not owner.closed and owner.jobs == {"external-generation"}
    asyncio.run(run())


def test_attached_routes_check_the_key_before_the_body(native):
    service, _ = native
    app = FastAPI()
    add_systemone_routes(app, lambda: service, api_key="secret")

    async def run():
        async with client(app) as http:
            bad = b"{not json"
            assert (await http.post("/v1/systemone", content=bad)).status_code == 401
            response = await http.post("/v1/systemone", content=bad, headers={"Authorization": "Bearer secret"})
            assert response.status_code == 422
            assert (await http.post("/v1/systemone", json=body(),
                                    headers={"Authorization": "Bearer secret"})).status_code == 200
    asyncio.run(run())
