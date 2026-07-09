"""Long chain-of-thought prompts for churn measurement on reasoning models.

Primary source: MATH-500 / AIME 2024 via HuggingFace ``datasets`` (network).
Fallback: a small bundled set of original competition-style problems so the
harness runs offline.

For CoT runs the "needle" span is the problem statement itself — the
measured signal is question-lookback (retrieval heads re-reading the
problem during reasoning, cf. Wu et al. 2404.15574).
"""

from __future__ import annotations

_BUNDLED_PROBLEMS = [
    "Let S be the sum of all positive integers n <= 200 such that n^2 + 1 is "
    "divisible by 5. Find the remainder when S is divided by 1000.",
    "A bag contains 6 red, 5 blue, and 4 green marbles. Three marbles are "
    "drawn without replacement. Compute the probability that all three "
    "colors appear, expressed as a fraction in lowest terms.",
    "Find the number of ordered pairs (a, b) of positive integers with "
    "a <= b and lcm(a, b) = 720.",
    "A circle of radius 5 is inscribed in a right triangle with hypotenuse "
    "26. Find the perimeter of the triangle.",
    "Define f(x) = x^3 - 3x + 1. How many real solutions does "
    "f(f(x)) = 0 have?",
]


def load_problems(source: str = "auto", limit: int = 5) -> list[dict]:
    """Returns [{id, problem}]. source: auto|math500|aime2024|bundled."""
    if source in ("auto", "math500"):
        try:
            from datasets import load_dataset

            ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
            return [
                {"id": f"math500-{i}", "problem": row["problem"]}
                for i, row in enumerate(ds.select(range(limit)))
            ]
        except Exception:
            if source == "math500":
                raise
    if source == "aime2024":
        from datasets import load_dataset

        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [
            {"id": f"aime24-{i}", "problem": row["Problem"]}
            for i, row in enumerate(ds.select(range(limit)))
        ]
    return [
        {"id": f"bundled-{i}", "problem": p}
        for i, p in enumerate(_BUNDLED_PROBLEMS[:limit])
    ]


def build_cot_prompt(tokenizer, problem: str) -> dict:
    """Chat-formatted prompt; needle = the problem statement."""
    messages = [
        {
            "role": "user",
            "content": "Solve the following problem. Think step by step, then "
            "state the final answer.\n\n" + problem,
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "needle": problem}
