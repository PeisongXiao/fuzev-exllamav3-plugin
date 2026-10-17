"""Condition the decision head on the target model's label logits."""

import logging

import torch

from fuzev.encoding import LABEL_CAPACITY, labels


logger = logging.getLogger(__name__)


class LMConditionedHead:
    def __init__(self, model, encode, head):
        self.head = head
        self.lm = next(module for module, _, _ in model.fwd_modules
                       if module.caps.get("logits_output"))
        self.labels = []
        for label in labels(min(head.max_options, LABEL_CAPACITY) if head.is_v2 else head.max_options):
            ids = encode(" " + label)
            if len(ids) != 1:
                raise ValueError("The model tokenizer must encode each decision label as one token")
            self.labels.append(ids[0])
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("The model tokenizer must encode distinct decision labels")
        self._selection = None
        self._selection_ready = False
        self._fallback_logged = False

    def __getattr__(self, name):
        return getattr(self.head, name)

    @torch.inference_mode()
    def score(self, primitive, query, candidates, *, route="labelled", modality="text"):
        # Project on the model's device even when the decision head runs on CPU.
        if route == "label_free":
            baseline = query.new_zeros(len(candidates), dtype=torch.float32)
        else:
            adapters = bool(self.lm.lora_a_tensors or self.lm.lora_b_tensors)
            if not adapters and not self._selection_ready:
                self._selection = self.lm.select_outputs(self.labels)
                self._selection_ready = True
                if self._selection is not None:
                    logger.info("Decision label projection: native selected outputs")
            projection, indices = self.lm, self.labels[:len(candidates)]
            if not adapters and self._selection is not None:
                projection, gather = self._selection
                indices = gather[:len(candidates)]
            elif not self._fallback_logged:
                reason = "active LM adapters" if adapters else "native output selection unavailable or unsupported"
                logger.info("Decision label projection: full native fallback (%s)", reason)
                self._fallback_logged = True
            vocabulary = projection.forward(query.reshape(1, 1, self.metadata["hidden_size"]).half(), {})[0, -1]
            baseline = vocabulary[indices].float()
        return self.head.score(primitive, query, candidates, baseline=baseline, modality=modality)

    def close(self):
        selection, self._selection = self._selection, None
        try:
            if selection is not None:
                selection[0].unload()
        finally:
            self.head.close()
