"""Chat prompt formatting with a fallback when chat_template is missing."""

from __future__ import annotations


def format_chat(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Prefer ``tokenizer.apply_chat_template``; else a simple Instruct layout."""
    tmpl = getattr(tokenizer, "chat_template", None)
    if tmpl:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    # Mistral-Instruct-style fallback used by older tokenizer builds.
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg["content"]
        if role == "system":
            parts.append(f"{content}\n")
        elif role == "user":
            parts.append(f"[INST] {content} [/INST]")
        else:
            parts.append(f" {content}</s>")
    text = "".join(parts)
    if add_generation_prompt and messages and messages[-1].get("role") == "user":
        if not text.endswith("[/INST]"):
            text += " "
    return text
