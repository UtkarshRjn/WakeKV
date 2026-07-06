"""Scripted multi-turn conversation with a topic pivot and late recall.

Turns 1..N-1 are fixed text (both roles) so runs are reproducible; only the
final assistant turn is generated and instrumented. The needle is the fact
sentence from turn 1, recalled after several unrelated turns — the setting
where frozen head roles are most likely to mis-serve (SnapKV multi-turn
failure; FlexiCache/HeteroCache untested here).
"""

from __future__ import annotations

FACTS = (
    "my locker code is 4831, my wifi password is 'maple-syrup-42', and my "
    "cat's name is Biscuit"
)

_FILLER_TURNS = [
    (
        "Can you suggest a simple weeknight pasta recipe?",
        "A reliable one: cook spaghetti, and while it boils, saute garlic in "
        "olive oil, add a tin of crushed tomatoes, simmer ten minutes, season, "
        "and toss with the pasta and a handful of basil. Grated parmesan on "
        "top. The whole thing takes about twenty minutes.",
    ),
    (
        "Nice. Unrelated - I'm planning a weekend trip to the mountains. "
        "What should I pack for early autumn?",
        "Layers are the key: a base layer, a warm mid layer, and a light "
        "waterproof shell. Add sturdy shoes, a hat, sunscreen, a headlamp, "
        "and more water than you think you need. Evenings get cold fast at "
        "altitude in autumn, so pack one warmer item than the forecast "
        "suggests.",
    ),
    (
        "Good tips. Also, how do I keep sourdough starter alive if I bake "
        "only once a week?",
        "Keep it in the fridge and feed it once a week: discard most of it, "
        "then refresh with equal weights of flour and water. Take it out the "
        "night before baking, give it a feed at room temperature, and it "
        "should be active by morning.",
    ),
]


def build_multiturn_prompt(tokenizer, n_filler_turns: int = 3) -> dict:
    messages = [
        {
            "role": "user",
            "content": f"Here are my account details: {FACTS}. "
            "Please remember these for later.",
        },
        {
            "role": "assistant",
            "content": "Got it - I'll remember your locker code, your wifi "
            "password, and your cat Biscuit's name for later.",
        },
    ]
    for user_msg, asst_msg in _FILLER_TURNS[:n_filler_turns]:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": asst_msg})
    messages.append(
        {"role": "user", "content": "By the way, what was my wifi password again?"}
    )
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt, "needle": FACTS, "answer": "maple-syrup-42"}
