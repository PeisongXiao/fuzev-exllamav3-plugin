"""Serve generation and FuZev decisions with one ExLlamaV3 Generator."""

import argparse
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request

from fuzev.api import APIBoundary, add_decision_route, health_response, parse_body, validation_error
from fuzev.prefix import DEFAULT_PREFIX_CAPACITY
from fuzev.service_contracts import GenerationRequest
from .systemone import add_systemone_routes


def create_app(service, *, api_key=None, max_request_bytes=None):
    """Mount the public API. The caller remains responsible for service shutdown."""
    app = FastAPI(title="FuZev ExLlamaV3 server")
    boundary = {"api_key": api_key}
    if max_request_bytes is not None:
        boundary["max_request_bytes"] = max_request_bytes
    app.add_middleware(APIBoundary, **boundary)
    routes = add_systemone_routes(app, lambda: service, include_models=True)
    app.state.fuzev_routes = routes

    async def generate(http, chat):
        request = await parse_body(http, GenerationRequest)
        if (request.messages is not None) != chat:
            raise validation_error("messages" if chat else "prompt",
                                   "Use messages for chat completions and prompt for completions")

        async def result(service, key):
            if request.stream:
                return await routes.stream(http, service.generate_stream(request, request_id=key), key)
            return await service.generate(request, request_id=key)
        return await routes.respond(http, result)

    @app.post("/v1/completions")
    async def completions(http: Request):
        return await generate(http, False)

    @app.post("/v1/chat/completions")
    async def chat_completions(http: Request):
        return await generate(http, True)

    if service.head is not None:
        async def decide(service, request, key):
            return await service.decide(request, request_id=key)
        add_decision_route(app, routes.respond, decide)

    @app.post("/v1/requests/{key}/cancel", status_code=202)
    async def cancel(key: str, http: Request):
        async def result(service, request_key):
            return {"id": key, "cancel_requested": routes.cancel(key)}
        return await routes.respond(http, result, status_code=202)

    @app.delete("/v1/prefixes/{key}")
    async def release(key: str, http: Request):
        async def result(service, request_key):
            return {"id": key, "released": service.release_prefix(key)}
        return await routes.respond(http, result)

    @app.get("/health")
    async def health(http: Request):
        async def result(service, key):
            service._check()
            return health_response(service.model_name, service.context_tokens,
                                   decisions=service.head is not None,
                                   images=service.image_embeddings is not None)
        return await routes.respond(http, result)

    return app


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--model", required=True, help="ExLlamaV3 model directory")
    cli.add_argument("--model-name", required=True, help="Model name exposed by the API")
    cli.add_argument("--head", help="Decision head bundle; omit for generation only")
    cli.add_argument("--head-device", help="Device for the decision head; default: available CUDA device or CPU")
    cli.add_argument("--chat-template-kwargs", type=json.loads, default={}, help="JSON object of chat template defaults")
    cli.add_argument("--calibration", choices=("raw", "calibrated"), default="raw",
                     help="Probability mode; requests without matching calibration use raw probabilities (default: raw)")
    cli.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    cli.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000)")
    cli.add_argument("--api-key", help="Require this bearer token on /v1 routes")
    cli.add_argument("--max-request-bytes", type=int, help="Maximum request body size in bytes (default 8 MiB)")
    cli.add_argument("--context-length", type=int, help="Context window; default: model and cache budget")
    cli.add_argument("--cache-tokens", type=int, help="Total cache tokens, rounded up to a page multiple; default: one context")
    cli.add_argument("--cache-bits", nargs=2, type=int, metavar=("K", "V"), help="Quantized KV cache bit widths")
    cli.add_argument("--max-batch-size", type=int, help="Engine batch size (default 16)")
    cli.add_argument("--prefill-profile", help="Prefill profile keys as a JSON object or JSON file path")
    cli.add_argument("--max-chunk-size", type=int,
                     help="Prefill chunk tokens, a multiple of 256 and at most the cache tokens; default: the "
                          "prefill profile's prefill_chunk (32,768), capped at the cache tokens. "
                          "Lower it if loading or prefill runs out of memory.")
    cli.add_argument("--use-vram", nargs="+", type=float, help="VRAM in GiB per device; default: engine allocation")
    cli.add_argument("--mtp-tokens", type=int, help="Enable the model's MTP component with this draft length")
    cli.add_argument("--vision", action="store_true",
                     help="Load the model's vision component for experimental image inputs")
    cli.add_argument("--prefix-capacity", type=int, default=DEFAULT_PREFIX_CAPACITY,
                     help=f"Retained prefixes with LRU eviction (default: {DEFAULT_PREFIX_CAPACITY})")
    return cli


async def serve(args):
    # Engine imports and CUDA allocation belong only to the launcher.
    import uvicorn
    from exllamav3 import Cache, Config, Model, Tokenizer
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.constants import PAGE_SIZE
    from exllamav3.generator.async_generator import AsyncGenerator
    from .generator_service import GeneratorService, build_generator_service, require_fork

    require_fork()

    from exllamav3.optimized import settings

    if args.prefill_profile is not None:
        from exllamav3.optimized import enable_prefill_optimizations

        profile = args.prefill_profile
        if not profile.lstrip().startswith("{"):
            profile = Path(profile).read_text()
        enable_prefill_optimizations(json.loads(profile))
    config = Config.from_directory(args.model)
    context = args.context_length or config.max_position_embeddings
    cache_tokens = ((args.cache_tokens or context) + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE
    max_chunk_size = min(settings.CONFIG["prefill_chunk"], cache_tokens) if args.max_chunk_size is None else args.max_chunk_size
    cache_options, load_options, generator_options = {}, {}, {}
    if args.cache_bits is not None:
        cache_options.update(layer_type=CacheLayer_quant, k_bits=args.cache_bits[0], v_bits=args.cache_bits[1])
    if args.max_batch_size is not None:
        cache_options["max_batch_size"] = args.max_batch_size
        generator_options["max_batch_size"] = args.max_batch_size
    load_options["max_chunk_size"] = generator_options["max_chunk_size"] = max_chunk_size
    if args.use_vram is not None:
        load_options["use_per_device"] = args.use_vram
    models, engine, service, app = [], None, None, None
    try:
        model = Model.from_config(config)
        models.append(model)
        print(f"Prefill optimizations {'enabled' if config.infer_params.prefill_optimizations else 'disabled'}; "
              f"prefill chunk {max_chunk_size} tokens"
              + (f"; profile {json.dumps(settings.CONFIG)}" if config.infer_params.prefill_optimizations else ""),
              flush=True)
        cache = Cache(model, max_num_tokens=cache_tokens, max_history=args.mtp_tokens or 0, **cache_options)
        generator_options["max_batch_size"] = cache.num_slots
        load_options["max_batch_size"] = cache.num_slots
        load_options["max_output_size"] = cache.num_slots * ((args.mtp_tokens or 0) + 1)
        if args.mtp_tokens:
            draft = Model.from_config(config, component="mtp")
            models.append(draft)
            draft_cache = Cache(draft, max_num_tokens=cache_tokens, **cache_options)
            draft.load(**load_options)
            generator_options.update(draft_model=draft, draft_cache=draft_cache, num_draft_tokens=args.mtp_tokens)
        model.load(**load_options)
        tokenizer = Tokenizer.from_config(config)
        vision = None
        if args.vision:
            vision = Model.from_config(config, component="vision")
            models.append(vision)
            vision.load(**load_options)
        engine = AsyncGenerator(model=model, cache=cache, tokenizer=tokenizer, **generator_options)
        context = min(context, config.max_position_embeddings,
                      engine.generator.max_total_tokens - 1 - engine.generator.num_draft_tokens)
        options = dict(model_name=args.model_name, context_tokens=context,
                       prefix_capacity=args.prefix_capacity, chat_template_kwargs=args.chat_template_kwargs,
                       image_embeddings=vision.get_image_embeddings if vision is not None else None)
        service = (build_generator_service(engine, tokenizer, head_directory=args.head,
                   calibration=args.calibration,
                   head_device=args.head_device, **options) if args.head else
                   GeneratorService(engine, tokenizer, **options))
        app = create_app(service, api_key=args.api_key, max_request_bytes=args.max_request_bytes)
        await uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port)).serve()
    finally:
        if app is not None:
            await app.state.fuzev_routes.close()
        if service is not None:
            await service.close()
        if engine is not None:
            await engine.close()
        for model in reversed(models):
            model.unload()


def main():
    asyncio.run(serve(parser().parse_args()))


if __name__ == "__main__":
    main()
