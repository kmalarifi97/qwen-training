FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ ./server/

RUN mkdir -p /app/data/uploads /app/data/datasets /app/data/adapters /app/data/db

EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
