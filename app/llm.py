import time
from dataclasses import dataclass
from groq import Groq
from app.config import GROQ_API_KEY, LLM_MODEL, REASONING_EFFORT

_client = None

def get_client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=GROQ_API_KEY, max_retries=5)
    return _client

@dataclass
class LLMResult:
    text: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int

def chat(prompt: str, system: str = "You are a helpful assistant.",
         temperature: float = 0.0, json_mode: bool = False,
         model: str | None = None, reasoning_effort: str | None = None) -> LLMResult:
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    effort = reasoning_effort or REASONING_EFFORT
    if effort:
        kwargs["reasoning_effort"] = effort
    t0 = time.perf_counter()
    resp = get_client().chat.completions.create(
        model=model or LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        **kwargs,
    )
    u = resp.usage
    return LLMResult(resp.choices[0].message.content,
                     time.perf_counter() - t0,
                     u.prompt_tokens, u.completion_tokens)