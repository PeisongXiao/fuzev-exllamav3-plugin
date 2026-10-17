import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from fuzev_exllamav3.server import create_app, parser, serve
from fuzev_exllamav3.systemone import HEADER
from conftest import body, decision
from test_systemone import client


@pytest.mark.parametrize("path,payload,status", [
    ("/v1/completions", {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}]}, 422),
    ("/v1/chat/completions", {"model": "test-model", "prompt": "Hello"}, 422),
    ("/v1/decisions", decision(messages=None, prefix_id="expired"), 404),
    ("/v1/decisions", decision(candidates=[{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]), 422),
    ("/v1/decisions", decision(candidates=[{"id": "yes", "text": "Yes"}]), 422),
])
def test_shared_error_shapes_and_ids(native, path, payload, status):
    service, owner = native
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post(path, json=payload, headers={HEADER: "route.1:-_"})
        assert response.status_code == status, response.text
        assert response.headers[HEADER] == "route.1:-_"
        if status == 404:
            assert response.json()["error"]["type"] == "invalid_request_error"
        else:
            assert isinstance(response.json()["detail"], list)
        assert not owner.completed
    asyncio.run(run())


@pytest.mark.parametrize("header,status", [(None, 401), ("Bearer wrong", 401), ("Bearer secret", 200)])
def test_optional_bearer_auth(native, header, status):
    service, _ = native
    async def run():
        headers = {HEADER: "auth"}
        if header is not None:
            headers["Authorization"] = header
        async with client(create_app(service, api_key="secret")) as http:
            for path in ("/v1/models",):
                response = await http.get(path, headers=headers)
                assert response.status_code == status
                assert response.headers[HEADER] == "auth"
            assert (await http.get("/health")).status_code == 200
            response = await http.post("/v1/systemone", json=body(), headers=headers)
            assert response.status_code == status
    asyncio.run(run())


def test_ids_compact_health_and_generation_only(native):
    service, owner = native
    service.head = None
    async def run():
        app = create_app(service)
        async with client(app) as http:
            health = await http.get("/health")
            assert set(health.json()) == {"status", "model", "context_window", "capabilities"}
            assert set(health.json()["capabilities"]) == {"generation", "decisions", "images", "streaming"}
            assert health.json()["capabilities"]["streaming"] is True
            assert health.headers[HEADER].startswith("req_")
            assert (await http.get("/v1/models")).json()["models"] == []
            assert (await http.post("/v1/systemone", json=body())).status_code == 404
            assert (await http.post("/v1/decisions", json=decision())).status_code == 404
            invalid = await http.get("/health", headers={HEADER: "invalid id"})
            assert invalid.status_code == 422 and invalid.headers[HEADER].startswith("req_")
            generated = await http.post("/v1/completions", json={"model": "test-model", "prompt": "Hello"})
            assert generated.status_code == 200
            assert generated.json()["id"] == generated.headers[HEADER]
            owner.error = RuntimeError("engine failed")
            assert (await http.get("/health")).status_code == 503
    asyncio.run(run())


def test_context_errors_and_prefix_release(native):
    service, _ = native
    async def run():
        async with client(create_app(service)) as http:
            generated = await http.post("/v1/completions", json={
                "model": "test-model", "prompt": "Hello", "retain_decision_prefix": True})
            key = generated.json()["prefix_id"]
            response = await http.delete("/v1/prefixes/" + key)
            assert response.json() == {"id": key, "released": True}
            service.context_tokens = 2
            response = await http.post("/v1/completions", json={"model": "test-model", "prompt": "Hello"})
            assert response.status_code == 422 and "2" in response.json()["detail"][0]["msg"]
            response = await http.post("/v1/decisions", json=decision())
            assert response.status_code == 422 and response.json()["detail"][0]["loc"] == ["body", "candidates"]
            response = await http.post("/v1/systemone", json=body())
            assert response.status_code == 422
            assert response.json()["detail"][0]["loc"] == ["body", "questions", "pick"]
    asyncio.run(run())


@pytest.mark.parametrize("stream", [False, True])
def test_omitted_generation_limit_respects_advertised_context(native, monkeypatch, stream):
    service, engine = native
    service.context_tokens = 12
    process = engine.process
    async def finish_at_limit(job):
        engine.completion = "x" * job.options["max_new_tokens"]
        result = await process(job)
        result["new_tokens"] = len(engine.completion)
        return result
    monkeypatch.setattr(engine, "process", finish_at_limit)
    async def run():
        async with client(create_app(service)) as http:
            exhausted = await http.post("/v1/completions", json={"prompt": "x" * 12, "stream": stream})
            assert exhausted.status_code == 422 and not engine.completed
            response = await http.post("/v1/completions", json={"prompt": "Hello", "stream": stream})
        assert response.status_code == 200, response.text
        if stream:
            import json
            chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            text = "".join(chunk["choices"][0]["text"] for chunk in chunks)
            usage = chunks[-1]["usage"]
        else:
            text, usage = response.json()["choices"][0]["text"], response.json()["usage"]
        assert text == "x" * 7
        assert usage == {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    asyncio.run(run())


@pytest.mark.parametrize("profile_arg,chunk_args,expected_chunk,extra", [
    (None, [], 512, []),
    ('{"prefill_chunk": 1024}', ["--max-chunk-size", "256"], 256, []),
    ('{"prefill_chunk": 1024}', [], 1024, []),
    ("file", [], 1024, []),
    (None, [], 512, ["--mtp-tokens", "3", "--cache-bits", "8", "8", "--max-batch-size", "7"]),
    ('{"prefill_chunk": 4096}', [], 1536, []),
])
def test_launcher_uses_one_generator_and_closes_its_models(native, monkeypatch, tmp_path, extra,
                                                          profile_arg, chunk_args, expected_chunk):
    service, owner = native
    profile = {"prefill_chunk": 512}
    if profile_arg == "file":
        profile_path = tmp_path / "prefill.json"
        profile_path.write_text(json.dumps({"prefill_chunk": 1024}))
        profile_arg = str(profile_path)
    profile_args = ["--prefill-profile", profile_arg] if profile_arg is not None else []
    args = parser().parse_args([
        "--model", "local-model", "--model-name", "test-model", "--head", str(service.bundle_directory),
        "--head-device", "cpu", "--cache-tokens", "1300", "--chat-template-kwargs", '{"enable_thinking": false}',
        *extra, *profile_args, *chunk_args])
    models = []
    class Model:
        def __init__(self):
            self.loaded = self.unloaded = False
        @classmethod
        def from_config(cls, config, **kwargs):
            model = cls()
            models.append(model)
            return model
        def load(self, *, max_chunk_size, **kwargs):
            self.prefill_capacity = max_chunk_size
            self.loaded = True
        def unload(self):
            self.unloaded = True
    config = owner.generator.model.config
    monkeypatch.setattr(config, "infer_params", SimpleNamespace(prefill_optimizations=True), raising=False)
    engine = sys.modules["exllamav3"]
    engine.Config = SimpleNamespace(from_directory=lambda path: config)
    engine.Model = Model
    engine.Tokenizer = SimpleNamespace(from_config=lambda config: service.tokenizer)
    caches = []
    def cache(model, **kwargs):
        value = SimpleNamespace(num_slots=kwargs.get("max_batch_size", 5), **kwargs)
        caches.append(value)
        return value
    engine.Cache = cache
    created = []
    def generator(max_chunk_size=128, **kwargs):
        created.append(kwargs)
        owner.generator.max_chunk_size = max_chunk_size
        owner.generator.max_total_tokens = kwargs["cache"].max_num_tokens
        owner.generator.num_draft_tokens = kwargs.get("num_draft_tokens", 0)
        return owner
    engine.Generator = generator
    monkeypatch.setattr(sys.modules["exllamav3.generator.async_generator"], "AsyncGenerator", generator, raising=False)
    for name, members in {"exllamav3.cache": {"CacheLayer_quant": object()},
                          "exllamav3.constants": {"PAGE_SIZE": 256},
                          "exllamav3.optimized": {"settings": SimpleNamespace(CONFIG=profile),
                                                 "enable_prefill_optimizations": profile.update}}.items():
        module = ModuleType(name)
        module.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, module)
    uvicorn = ModuleType("uvicorn")
    uvicorn.Config = lambda app, **kwargs: SimpleNamespace(app=app, **kwargs)
    class Server:
        def __init__(self, config):
            self.config = config
        async def serve(self):
            assert self.config.host == "127.0.0.1" and self.config.port == 8000
            async with client(self.config.app) as http:
                response = await http.post("/v1/completions", json={"prompt": "Hello"})
                assert response.status_code == 200
                assert (await http.post("/v1/decisions", json=decision())).status_code == 200
                assert (await http.post("/v1/systemone", json=body())).status_code == 200
                assert (await http.get("/v1/models")).json()["data"][0]["id"] == "test-model"
                health = (await http.get("/health")).json()
                assert health["context_window"] == owner.generator.max_total_tokens - 1 - owner.generator.num_draft_tokens
                owner.block = asyncio.Event()
                owner.started.clear()
                pending = asyncio.create_task(http.post("/v1/completions", json={"prompt": "Hello"},
                                                      headers={HEADER: "launcher-cancel"}))
                await owner.started.wait()
                assert (await http.post("/v1/requests/launcher-cancel/cancel")).status_code == 202
                assert (await pending).status_code == 499
    uvicorn.Server = Server
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    asyncio.run(serve(args))
    assert len(created) == 1
    assert all(cache.max_num_tokens % 256 == 0 and cache.max_num_tokens >= 1300 for cache in caches)
    assert owner.closed and all(model.loaded and model.unloaded for model in models)
    assert all(model.prefill_capacity == expected_chunk for model in models)
    assert owner.generator.max_chunk_size == expected_chunk


def test_generation_accepts_common_openai_fields_and_model_alias(native):
    service, _ = native
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/chat/completions", json={
                "model": "client-alias", "n": 1, "user": "client-user", "stop": "stop here",
                "logprobs": False, "stream_options": None, "max_completion_tokens": 10,
                "messages": [{"role": "developer", "name": "policy", "content": [{"type": "text", "text": "Answer"}]},
                             {"role": "user", "content": [{"type": "text", "text": "Hello"}]}]})
            assert response.status_code == 200, response.text
            assert response.json()["model"] == service.model_name
            assert (await http.post("/v1/decisions", json=decision(model="client-alias"))).status_code == 200
    asyncio.run(run())


def test_reserved_request_chat_template_is_rejected(native):
    service, engine = native
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_kwargs": {"chat_template": "{{ messages }}"}})
        assert response.status_code == 422
        assert not engine.completed
    asyncio.run(run())


def test_nested_json_and_body_limit_share_boundary_errors(native):
    service, _ = native
    async def run():
        async with client(create_app(service)) as http:
            response = await http.post("/v1/systemone", content=b"[" * 100000 + b"]" * 100000,
                                       headers={HEADER: "nested"})
            assert response.status_code == 422
            assert response.json()["detail"][0]["msg"] == "JSON nesting is too deep"
            assert response.headers[HEADER] == "nested"
            missing = await http.get("/absent", headers={HEADER: "absent"})
            assert missing.status_code == 404 and missing.headers[HEADER] == "absent"
        async with client(create_app(service, max_request_bytes=1024)) as http:
            response = await http.post("/v1/completions", json={"prompt": "x" * 1025})
            assert response.status_code == 413
    asyncio.run(run())
