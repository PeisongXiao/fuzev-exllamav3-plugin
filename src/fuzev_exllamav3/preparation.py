"""Render the shared decision template with the model's tokenizer."""

from copy import deepcopy

from fuzev.encoding import LIST_AWARE_VERSION, TEMPLATE, TEMPLATE_KWARGS, VERSION, list_aware_query_suffix, query_suffix


class ReadoutPreparer:
    def __init__(self, tokenizer, encoding_version=VERSION):
        if encoding_version not in (VERSION, LIST_AWARE_VERSION):
            raise ValueError(f"Unsupported decision encoding: {encoding_version}; supported: {VERSION}, {LIST_AWARE_VERSION}")
        self.tokenizer = tokenizer
        self.template = TEMPLATE.read_text()

    def query_suffix(self, question, candidates, additional_context="", *, route="labelled"):
        suffix = list_aware_query_suffix if route == "label_free" else query_suffix
        return suffix(question, candidates, additional_context)

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
            **TEMPLATE_KWARGS)

    def base(self, messages):
        return self.encode(self.render(messages))
