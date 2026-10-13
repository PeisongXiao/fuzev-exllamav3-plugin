# FuZev ExLlamaV3 Integration

`fuzev-exllamav3-plugin` adds FuZev decisions to an ExLlamaV3 Generator built
from the [FuZev fork](https://github.com/PeisongXiao/exllamav3-fuzev). It serves
`POST /v1/systemone`, the SystemOne API of hosted Jev and
[Kev](https://github.com/jaredpalmer/kev), and provides the `fuzev-native`
launcher. Generation and decisions share the model, tokenizer, scheduler and
cache.

[FuZev index](https://github.com/PeisongXiao/fuzev) |
[Decision head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev) |
[Runtime](https://github.com/PeisongXiao/fuzev-runtime) |
[Models and head bundle](https://github.com/PeisongXiao/fuzev/blob/main/docs/models.md) |
[Engine build guide](https://github.com/PeisongXiao/exllamav3-fuzev/blob/main/README.md#install-this-fork)

**Contents**

- [Install](#install)
- [Start The Server](#start-the-server)
- [Configure The Engine](#configure-the-engine)
- [Attach To An Application](#attach-to-an-application)
  - [Vision](#vision)
- [Tests](#tests)
- [Limitations](#limitations)

## Install

Install runtime and plugin release **1.0.0** with the FuZev ExLlamaV3 fork. Follow
the canonical guide:

1. [Environment setup](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#create-an-environment)
2. [ExLlamaV3 engine build](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#exllamav3)
3. [Decision integration](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#exllamav3-with-exl3)

The guide also covers [release wheels](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#local-release-wheels).
Keep that environment active for startup.

## Start The Server

You need:

- An EXL3 checkpoint of the head's backbone: Qwen3.8-Flash-Next for the
  published head (see [Models and head bundle](https://github.com/PeisongXiao/fuzev/blob/main/docs/models.md)).
- A head bundle containing `config.json`, `head.safetensors` and
  `calibration.json`.

```bash
export FUZEV_MODEL=/absolute/path/to/exl3-model
export FUZEV_HEAD=/absolute/path/to/fuzev-head
fuzev-native --model "$FUZEV_MODEL" --head "$FUZEV_HEAD" --model-name fuzev
```

The launcher creates one `AsyncGenerator`, attaches the decision service and
listens on **127.0.0.1:8000**. Change the address with `--host` and `--port`.
Before exposing the server beyond localhost, set `--api-key` and use a TLS proxy.

From another terminal, send a decision request:

```bash
curl --fail-with-body --silent --show-error \
  http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "fuzev",
    "state": {"message": "I was charged twice."},
    "questions": {
      "team": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {"billing": "Charges", "shipping": "Delivery"}
      }
    }
  }'
```

The answer appears under `answers.team`. The
[API guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md) covers
SDK examples, response fields, generation, images, retained prefixes,
cancellation and errors.

Optional FuZev [extensions](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#extensions)
beyond the Jev API: images, retained prefixes, cancellation, `/v1/decisions` and
Score levels 11-16.

## Configure The Engine

| Option | Purpose |
| --- | --- |
| `--model`, `--model-name` | Required checkpoint directory and advertised model ID |
| `--head` | Decision head bundle; omit for generation only |
| `--vision` | Load the model's vision encoder for image input |
| `--max-request-bytes` | Request body limit including base64 images; default 8 MiB |
| `--context-length` | Context window; defaults to the model configuration, capped by `--cache-tokens` |
| `--cache-tokens`, `--cache-bits K V` | Cache capacity and precision; capacity defaults to one page-aligned context |
| `--max-batch-size`, `--max-chunk-size` | Engine batch size; prefill chunk, default 8,192 tokens (lower it if prefill runs out of memory) |
| `--use-vram` | VRAM allocation in GiB per device |
| `--mtp-tokens N` | Load the model's MTP component with draft length N |
| `--head-device` | Decision head device; defaults to CUDA when available, otherwise CPU. Pass `cpu` to keep the head off the GPU |
| `--calibration raw\|calibrated` | Calibration mode, selected at startup; default `raw` |
| `--prefix-capacity` | Retained-prefix count; default 128, LRU eviction, no time-based expiry |
| `--chat-template-kwargs` | Default chat-template arguments for messages, as JSON; default `{}`. Example: `'{"enable_thinking":false}'`. Request `chat_template_kwargs` take precedence |

- Scheduling and memory allocation follow ExLlamaV3's defaults unless
  configured.
- Sampling starts from the model's `generation_config.json` values for
  `temperature`, `top_p`, `top_k` and `min_p`; request fields override them.
  When neither sets any of them, ExLlamaV3's default sampler applies.
- Retained prefixes store accepted tokens and image embeddings. The Generator
  reuses or recomputes model state through its existing cache.

## Attach To An Application

Use the application's existing `AsyncGenerator` on its event loop. To add
SystemOne to an app that already serves generation:

```python
from fuzev_exllamav3.generator_service import build_generator_service
from fuzev_exllamav3.systemone import add_systemone_routes

def attach(app, generator, tokenizer, *, head_directory, model_name, api_key=None):
    service = build_generator_service(
        generator, tokenizer, head_directory=head_directory, model_name=model_name,
    )
    routes = add_systemone_routes(
        app, lambda: service, include_models=True, api_key=api_key,
    )
    return service, routes

# During application shutdown:
# await routes.close()
# await service.close()
# The application closes its Generator separately.
```

- Use `include_models=True` when the application has no `/v1/models` route.
  Otherwise omit it and call
  `fuzev.api.models_response(service.model_name, service.head)` in the existing
  controller.
- Existing routes stay in place.
- Closing the service drains jobs, releases prefixes and closes the head. The
  application keeps responsibility for the Generator.

For a complete API on a new FastAPI app, use `create_app(service)` from
`fuzev_exllamav3.server`. Close its `app.state.fuzev_routes` before closing the
service.

### Vision

Add `--vision` to the launcher command to load the model's vision component.
An application can instead supply its already-loaded component:

```python
service = build_generator_service(
    generator, tokenizer, head_directory=head_directory, model_name=model_name,
    image_embeddings=vision.get_image_embeddings,
)
```

- The callback receives `(tokenizer, rgb)` and returns an ExLlamaV3
  `MMEmbedding`. Asynchronous callbacks are supported.
- Consume the PIL RGB image before returning.
- The application manages vision residency and scheduling.

Supported workflows include:

- Standalone image decisions.
- A retained text prefix followed by a new image.
- Decisions on retained prefixes from generation requests that included an
  image.

## Tests

From the plugin source directory in the installed environment:

```bash
python -m pip install ".[test]"
CUDA_VISIBLE_DEVICES='' python -m pytest -q
```

The tests use a small CPU head and simulated engine jobs. They cover SDK
requests, image-prefix retention, streaming and cancellation.

One opt-in check sends a SystemOne request to a running server. It skips unless
`FUZEV_NATIVE_SMOKE_URL` is set. Also set `FUZEV_API_KEY` when the server uses
`--api-key`:

```bash
FUZEV_NATIVE_SMOKE_URL=http://127.0.0.1:8000 python -m pytest -m gpu
```

## Limitations

- FuZev v1.0 is an experimental research release.
- v1.0 accepts 2 to 16 Choice options or Score levels per question. The head
  bundle declares the maximum (`max_options` in its `config.json`); the
  published head declares 16. Larger counts are planned for v1.1. Hosted Jev
  and Kev allow different counts; see the
  [capacity comparison](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#capacity-comparison).
- Noul has exactly two outcomes.
- ExLlamaV3 readout currently supports Generators without tensor parallelism.
  Multi-GPU layer splits are untested.
- The engine requires CUDA; the head runs on CPU or CUDA.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
- The first request after startup takes a few seconds longer while the prefill
  kernels compile.
- Decision probabilities can differ slightly between server starts, because
  kernel configurations are chosen by timing when they compile. They can also
  differ depending on which prompt prefixes are already cached.
- Images are experimental. Image decisions use the head's Qwen-format encoding.
  Generation requests with an image use the served model's own image
  placeholder.
- Only CPU tests cover image requests; v1.0 reports no GPU image run.

[Benchmarks](https://github.com/PeisongXiao/fuzev/blob/main/docs/benchmarks.md)
report decision quality;
[integration results](https://github.com/PeisongXiao/fuzev/blob/main/docs/integration.md)
cover recorded engine runs. Methods are in the
[research archive](https://github.com/PeisongXiao/fuzev-research-archive).
