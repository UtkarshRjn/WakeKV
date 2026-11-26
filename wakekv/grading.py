"""Task grading for masked-accuracy and system evals.

NIAH uses exact needle match. MATH extracts a boxed / final answer and
compares with normalized string or numeric equivalence. Multiturn checks
that all planted facts appear in the generation.
"""

from __future__ import annotations

import re
from fractions import Fraction


_BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LAST_NUMBER = re.compile(r"(-?\d+(?:/\d+)?(?:\.\d+)?)")


def extract_math_answer(text: str) -> str | None:
    """Prefer the last \\boxed{...}; else the last number-like token."""
    boxes = _BOXED.findall(text)
    if boxes:
        return boxes[-1].strip()
    matches = _LAST_NUMBER.findall(text.replace(",", ""))
    if matches:
        return matches[-1].strip()
    return None


def _normalize_math(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "")
    s = re.sub(r"\\left|\\right", "", s)
    s = s.replace("\\%", "%").replace("\\dfrac", "\\frac")
    return s.strip().lower()


def _as_number(s: str) -> float | None:
    s = _normalize_math(s)
    s = s.replace("%", "")
    try:
        if "/" in s and s.count("/") == 1 and "." not in s:
            return float(Fraction(s))
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_answers_equal(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    a, b = _normalize_math(pred), _normalize_math(gold)
    if a == b:
        return True
    na, nb = _as_number(a), _as_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) <= 1e-6 * max(1.0, abs(nb))
    return a.replace(" ", "") == b.replace(" ", "")


def grade_niah(generated: str, answer: str) -> bool:
    return answer.strip() in generated


def grade_math(generated: str, answer: str) -> bool:
    return math_answers_equal(extract_math_answer(generated), answer)


def grade_multiturn(generated: str) -> bool:
    """Require locker code, wifi password, and cat name."""
    text = generated.lower()
    return (
        "4831" in text
        and "maple-syrup-42" in text
        and "biscuit" in text
    )


def grade(task_kind: str, generated: str, answer: str) -> bool:
    if task_kind.startswith("niah"):
        return grade_niah(generated, answer)
    if task_kind.startswith("math") or task_kind.startswith("cot"):
        return grade_math(generated, answer)
    if task_kind.startswith("multiturn"):
        return grade_multiturn(generated)
    return answer.strip() in generated
