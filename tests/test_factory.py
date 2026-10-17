"""Decision outputs across the compact/native projection boundary, without CUDA."""

from types import SimpleNamespace

import pytest
import torch

from fuzev_exllamav3.factory import LMConditionedHead


LABEL_IDS = [2, 0, 1] + list(range(3, 20))


class Projection:
    caps = {"logits_output": True}

    def __init__(self, values):
        self.values = torch.tensor([[values]], dtype=torch.float16)

    def forward(self, query, params):
        if self.values is None:
            raise RuntimeError("Projection unloaded")
        return self.values + query[..., :1]

    def unload(self):
        self.values = None


class NativeProjection(Projection):
    def __init__(self):
        super().__init__(list(range(20)))
        self.lora_a_tensors, self.lora_b_tensors = {}, {}

    def select_outputs(self, token_ids):
        # Compact columns are deliberately not in label order.
        self.compact = Projection([1, 2, 0] + list(range(3, 20)))
        return self.compact, torch.tensor([1, 2, 0] + list(range(3, 20)))

    def forward(self, query, params):
        result = super().forward(query, params)
        if self.lora_a_tensors:
            result[..., 2] -= 4
        if self.lora_b_tensors:
            result[..., 0] += 3
        return result


def wrap(native, lm):
    return LMConditionedHead(SimpleNamespace(fwd_modules=[(lm, None, None)]),
                             lambda text: [LABEL_IDS[ord(text[-1]) - ord("A")]],
                             native[0].head.head)


def score(head, count=3, *, route="labelled"):
    return head.score("choice", torch.tensor([0.5] + [0.0] * 7),
                      torch.zeros(count, 8), route=route)


def assert_scores(result, expected):
    logits, raw, calibrated = result
    assert logits == expected
    expected = torch.tensor(expected)
    assert raw == pytest.approx(expected.softmax(0).tolist())
    assert calibrated == pytest.approx((expected / 2).softmax(0).tolist())


def unavailable(*args, **kwargs):
    raise RuntimeError("Projection unavailable")


@pytest.mark.parametrize("path", ["selected", "unsupported"])
@pytest.mark.parametrize("count", [2, 3, 20])
def test_label_logits_and_whole_slate_probabilities(native, path, count):
    lm = NativeProjection()
    if path == "unsupported":
        lm.select_outputs = lambda ids: None
    head = wrap(native, lm)
    expected = ([2.5, 0.5, 1.5] + [float(i) + 0.5 for i in range(3, 20)])[:count]
    assert_scores(score(head, count), expected)
    assert_scores(score(head, count), expected)


@pytest.mark.parametrize("adapter,expected", [
    ("lora_a_tensors", [-1.5, 0.5, 1.5]),
    ("lora_b_tensors", [2.5, 3.5, 1.5]),
])
@pytest.mark.parametrize("initially_active", [False, True])
def test_adapters_preserve_native_logits_before_and_after_selection(native, adapter, expected, initially_active):
    lm = NativeProjection()
    tensors = getattr(lm, adapter)
    if initially_active:
        tensors["active"] = object()
    head = wrap(native, lm)
    assert_scores(score(head), expected if initially_active else [2.5, 0.5, 1.5])
    tensors["active"] = object()
    assert_scores(score(head), expected)
    tensors.clear()
    assert_scores(score(head), [2.5, 0.5, 1.5])


@pytest.mark.parametrize("warm", [False, True])
def test_label_free_zero_baseline_without_available_lm(native, warm):
    lm = NativeProjection()
    if not warm:
        lm.select_outputs = unavailable
        lm.forward = unavailable
    head = wrap(native, lm)
    if warm:
        assert_scores(score(head), [2.5, 0.5, 1.5])
        lm.select_outputs = lm.forward = lm.compact.forward = unavailable
    assert_scores(score(head, route="label_free"), [0.0, 0.0, 0.0])


@pytest.mark.parametrize("failure", ["builder", "selected_forward", "full_forward"])
def test_projection_errors_are_not_hidden_by_fallback(native, failure):
    lm = NativeProjection()
    if failure == "builder":
        def invalid_ids(ids):
            raise ValueError("Invalid output IDs")
        lm.select_outputs = invalid_ids
    elif failure == "selected_forward":
        compact = Projection([0.0])
        compact.forward = unavailable
        lm.select_outputs = lambda ids: (compact, torch.tensor([0]))
    else:
        lm.select_outputs = lambda ids: None
        lm.forward = unavailable
    head = wrap(native, lm)
    with pytest.raises(ValueError if failure == "builder" else RuntimeError,
                       match="Invalid output IDs" if failure == "builder" else "Projection unavailable"):
        score(head)


def test_close_releases_only_owned_compact_projection(native):
    lm = NativeProjection()
    head = wrap(native, lm)
    assert_scores(score(head), [2.5, 0.5, 1.5])
    compact = lm.compact
    head.close()
    with pytest.raises(RuntimeError, match="Projection unloaded"):
        compact.forward(torch.zeros(1, 1, 8), {})
    assert lm.forward(torch.zeros(1, 1, 8), {}).tolist() == [[list(range(20))]]
