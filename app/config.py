import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if not GROQ_API_KEY or not LLM_MODEL:
    raise RuntimeError("Set GROQ_API_KEY and LLM_MODEL in .env")