"""Render the shared decision template with the model's tokenizer."""

from copy import deepcopy

from fuzev.encoding import TEMPLATE


class ReadoutPreparer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.template = TEMPLATE.read_text()

    def encode(self, text, embeddings=None):
        return tuple(self.tokenizer.encode(
            text, encode_special_tokens=True, embeddings=embeddings).reshape(-1).tolist())

    def render(self, messages, image=False):
        context = deepcopy(messages)
        if image:
            user = next((row for row in context if row["role"] == "user"), None)
            if user is None:
                user = {"role": "user", "content": "Consider the supplied image."}
                context.insert(0, user)
            user["content"] = [{"type": "image"}, {"type": "text", "text": user["content"]}]
        return self.tokenizer.hf_render_chat_template(
            context, add_generation_prompt=False, chat_template=self.template,
            enable_thinking=False, reasoning_effort="none", preserve_thinking=True)

    def base(self, messages):
        return self.encode(self.render(messages))
