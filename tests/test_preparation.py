"""Literal decision rendering with the shared encoding defaults."""

import pytest

from fuzev_exllamav3.preparation import ReadoutPreparer
from conftest import Tokenizer


@pytest.mark.parametrize("image", [False, True])
def test_decision_rendering_preserves_reasoning_and_image_content(image):
    messages = [{"role": "user", "content": "Question"},
                {"role": "assistant", "content": "Answer.", "reasoning_content": "Reason."},
                {"role": "user", "content": "Next."}]
    image_text = "<|vision_start|><|image_pad|><|vision_end|>" if image else ""
    expected = ("<|im_start|>user\n" + image_text + "Question<|im_end|>\n"
                "<|im_start|>assistant\n<think>\nReason.\n</think>\n\nAnswer.<|im_end|>\n"
                "<|im_start|>user\nNext.<|im_end|>\n")
    preparer = ReadoutPreparer(Tokenizer())
    assert preparer.render(messages, image=image) == expected
    assert messages[0]["content"] == "Question"
