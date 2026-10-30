#!/bin/bash
set -e

PORT=${PORT:-8501}

if [ "$SERVICE_TYPE" = "backend" ]; then
    echo "Starting FastAPI Backend on port $PORT..."
    exec uvicorn backend:app --host 0.0.0.0 --port "$PORT"
elif [ "$SERVICE_TYPE" = "frontend" ]; then
    echo "Starting Streamlit Frontend on port $PORT..."
    exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
else
    # Default: Run both in unified container
    echo "Starting FastAPI Backend on internal port 8000..."
    uvicorn backend:app --host 127.0.0.1 --port 8000 &
    BACKEND_PID=$!

    echo "Starting Streamlit Frontend on public port $PORT..."
    streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0 --server.headless true
fi
