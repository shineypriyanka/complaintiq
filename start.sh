#!/bin/sh
set -e

# Start the API internally (not exposed publicly)
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 &

# Wait until the API is healthy, so the UI never shows "offline" at startup
python - <<'EOF'
import time, httpx
for _ in range(60):
    try:
        if httpx.get("http://127.0.0.1:8000/health", timeout=2).status_code == 200:
            print("API is ready")
            break
    except Exception:
        pass
    time.sleep(2)
EOF

# Start the public UI on the port Hugging Face expects
exec python -m streamlit run ui/streamlit_app.py \
  --server.port 7860 --server.address 0.0.0.0 --server.headless true