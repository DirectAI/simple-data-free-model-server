timestamp=$(date +%s)
log_dir="logs/fastapi_$timestamp"
mkdir -p "$log_dir"
python -m unittest unit_tests/test.py
uvicorn server:app --host 0.0.0.0 --port 8000 --log-level warning
