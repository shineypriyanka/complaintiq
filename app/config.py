import os
from dotenv import load_dotenv

load_dotenv()   

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")