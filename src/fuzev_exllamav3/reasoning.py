"""Incremental Qwen-style reasoning and answer text separation."""


class ReasoningSplitter:
    def __init__(self, *, thinking=False):
        self.field = "reasoning" if thinking else "content"
        self.pending = ""

    def feed(self, text, *, final=False):
        self.pending += text
        parts = []
        tags = ("<think>", "</think>")
        while self.pending:
            matches = [(self.pending.find(tag), tag) for tag in tags if tag in self.pending]
            if matches:
                position, tag = min(matches)
                if position:
                    parts.append({self.field: self.pending[:position]})
                self.pending = self.pending[position + len(tag):]
                self.field = "reasoning" if tag == "<think>" else "content"
                continue
            # A delimiter can span any number of engine events.
            held = max((length for tag in tags for length in range(1, len(tag))
                        if self.pending.endswith(tag[:length])), default=0) if not final else 0
            ready = self.pending[:-held] if held else self.pending
            if ready:
                parts.append({self.field: ready})
            self.pending = self.pending[-held:] if held else ""
            break
        return parts


def chat_message(text, *, thinking=False):
    splitter = ReasoningSplitter(thinking=thinking)
    parts = splitter.feed(text, final=True)
    content = "".join(part.get("content", "") for part in parts)
    reasoning = "".join(part.get("reasoning", "") for part in parts)
    return {"role": "assistant", "content": content or None, "reasoning": reasoning or None}
