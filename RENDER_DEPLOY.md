# Render deployment

Service type: Web Service

- Build: `pip install --upgrade pip && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.10.0 && pip install --no-cache-dir -r requirements-render.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health check: `/api/health`
- Python: `3.13.5`

The app uses one in-memory simulation engine and WebSocket connections, so keep one process/worker.
