# Container for free hosting (Hugging Face Spaces / Render). PORT is set by the host.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Free tiers have little CPU; keep ONNX Runtime from oversubscribing
ENV ORT_THREADS=2

ENV PORT=7860
EXPOSE 7860
# --proxy-headers so sample image URLs use https behind the host's proxy
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips=*"]
