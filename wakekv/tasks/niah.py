"""Needle-in-a-haystack task construction (synthetic filler, UUID needle).

Follows 2602.11162: needle = dynamically generated UUID so retrieval cannot
come from memorization. Filler is synthetic template text so the repo ships
no external corpora.
"""

from __future__ import annotations

import random
import uuid

_TOPICS = [
    "the harbor at dawn", "an old observatory", "the municipal archive",
    "a mountain rail line", "the botanical garden", "a river ferry",
    "the clocktower square", "an abandoned foundry", "the coastal lighthouse",
    "a terraced vineyard",
]

_TEMPLATES = [
    "On day {i}, the caretaker walked past {topic} and noted that the weather "
    "had shifted slightly compared with the previous week.",
    "Visitors to {topic} on day {i} remarked that the light in the late "
    "afternoon made everything look unusually calm.",
    "The maintenance log for day {i} mentions {topic} twice, both times in "
    "connection with routine inspections that found nothing unusual.",
    "A short entry from day {i} describes {topic} as quiet, with only a few "
    "people passing through before noon.",
]


def make_needle(rng: random.Random) -> tuple[str, str, str]:
    key = f"entry-{rng.randint(100, 999)}"
    value = str(uuid.UUID(int=rng.getrandbits(128)))
    needle = f"The special magic identifier for {key} is {value}."
    question = (
        f"\n\nQuestion: What is the special magic identifier for {key}? "
        "Answer with the identifier only.\nAnswer:"
    )
    return needle, question, value


def build_niah_prompt(
    tokenizer,
    context_tokens: int = 5000,
    depth_fraction: float = 0.5,
    seed: int = 0,
) -> dict:
    """Returns dict(prompt, needle, question, answer, depth_fraction, seed)."""
    rng = random.Random(seed)
    needle, question, answer = make_needle(rng)

    sentences: list[str] = []
    i = 0
    # Build filler until the assembled prompt exceeds the token target.
    while True:
        i += 1
        sentences.append(
            rng.choice(_TEMPLATES).format(i=i, topic=rng.choice(_TOPICS))
        )
        if i % 50 == 0:
            body = " ".join(sentences)
            if len(tokenizer(body, add_special_tokens=False)["input_ids"]) > context_tokens:
                break

    insert_at = int(len(sentences) * depth_fraction)
    sentences.insert(insert_at, needle)
    prompt = (
        "Read the following notes carefully; a question follows.\n\n"
        + " ".join(sentences)
        + question
    )
    return {
        "prompt": prompt,
        "needle": needle,
        "question": question,
        "answer": answer,
        "depth_fraction": depth_fraction,
        "seed": seed,
    }
