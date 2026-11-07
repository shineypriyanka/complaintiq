import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
SPACE_ID = os.environ["HF_SPACE_ID"]
api = HfApi()   # reads HF_TOKEN from the environment

api.create_repo(SPACE_ID, repo_type="space", space_sdk="docker",
                private=True, exist_ok=True)

# Secrets and settings: stored by Hugging Face, never inside the code
api.add_space_secret(SPACE_ID, "GROQ_API_KEY", os.environ["GROQ_API_KEY"])
api.add_space_variable(SPACE_ID, "LLM_MODEL", os.getenv("LLM_MODEL", "openai/gpt-oss-120b"))
api.add_space_variable(SPACE_ID, "REASONING_EFFORT", os.getenv("REASONING_EFFORT", "low"))

api.upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    repo_type="space",
    ignore_patterns=[".env", ".venv/*", ".git/*", "data/raw/*", "data/chroma/*",
                     "results/*", "*__pycache__*", "*.pyc"],
    commit_message="Deploy ComplaintIQ",
)
print(f"Uploaded. Watch the build: https://huggingface.co/spaces/{SPACE_ID}")