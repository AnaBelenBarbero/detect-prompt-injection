#!/bin/bash
poetry run streamlit run app/main.py --server.port=8501 --server.address=0.0.0.0 &
poetry run uvicorn src.predict:app --host 0.0.0.0 --port 80 