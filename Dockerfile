FROM python:3.14-slim

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# 1) Dependencies first (cached unless requirements.txt changes)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) App code, then build both vector indexes inside the image
COPY --chown=user . .
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh \
 && python -m scripts.build_index \
 && python -m scripts.build_policy_index

EXPOSE 7860
CMD ["./start.sh"]