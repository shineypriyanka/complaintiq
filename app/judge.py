import json
from app.llm import chat, LLMResult

JUDGE_SYSTEM = (
    "You are a strict compliance auditor. You check whether a bank's reply to a "
    "customer is faithful to the regulatory excerpts it was given. Judge ONLY "
    "against the excerpts, never against your own knowledge of the rules."
)

def judge_reply(reply: str, sources: list[dict]) -> tuple[dict, LLMResult]:
    excerpts = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(sources, start=1))
    prompt = (
        f"Regulatory excerpts:\n{excerpts}\n\n"
        f"Reply to audit:\n<reply>\n{reply}\n</reply>\n\n"
        "Tasks:\n"
        "1. List every claim in the reply about regulations, customer rights, deadlines, "
        "eligibility or complaint procedure. Ignore greetings, apologies, restatements of "
        "the customer's issue, and the bank's promise to investigate.\n"
        "2. For each claim give:\n"
        '   - "cited": the [n] numbers attached to it in the reply (empty list if none)\n'
        '   - "supported": "yes" if the excerpts fully support it; "partial" if they support '
        "only part of it, or the wording changes its meaning; \"no\" otherwise\n"
        '   - "supporting": the excerpt numbers that actually support it\n'
        '3. "promises_compensation": true if the reply promises a refund, compensation or '
        'a specific outcome. "admits_liability": true if it admits the bank was at fault.\n\n'
        "Respond only with JSON:\n"
        '{"claims": [{"claim": "...", "cited": [1], "supported": "yes", "supporting": [1]}], '
        '"promises_compensation": false, "admits_liability": false}'
    )
    res = chat(prompt, system=JUDGE_SYSTEM, json_mode=True, reasoning_effort="medium")
    return json.loads(res.text), res