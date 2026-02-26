FROM python:3.11-slim

WORKDIR /app

# Force rebuild - v4
# Встановлення системних залежностей (якщо потрібно)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway сам встановлює PORT, але додамо fallback
# ENV PORT=8001

# Простий формат без змінних
# At the very end of your Dockerfile
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
