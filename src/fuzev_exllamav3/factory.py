"""Condition the decision head on the target model's label logits."""

import torch

from fuzev.encoding import labels


class LMConditionedHead:
    def __init__(self, model, encode, head):
        self.head = head
        self.lm = next(module for module, _, _ in model.fwd_modules
                       if module.caps.get("logits_output"))
        self.labels = []
        for label in labels(head.max_options):
            ids = encode(" " + label)
            if len(ids) != 1:
                raise ValueError("The model tokenizer must encode each decision label as one token")
            self.labels.append(ids[0])
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("The model tokenizer must encode distinct decision labels")

    def __getattr__(self, name):
        return getattr(self.head, name)

    @torch.inference_mode()
    def score(self, primitive, query, candidates):
        # Project on the model's device even when the decision head runs on CPU.
        vocabulary = self.lm.forward(query.reshape(1, 1, self.metadata["hidden_size"]).half(), {})[0, -1]
        baseline = vocabulary[self.labels[:len(candidates)]].float()
        return self.head.score(primitive, query, candidates, baseline=baseline)

    def close(self):
        self.head.close()
