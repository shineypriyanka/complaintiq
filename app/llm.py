from groq import Groq
from app.config import GROQ_API_KEY, LLM_MODEL

client = Groq(api_key=GROQ_API_KEY)

def chat(prompt: str, system: str = "You are a helpful assistant.",
         temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content