#!/usr/bin/env bash
# Starts the FastAPI server for BRIO
python -m uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
