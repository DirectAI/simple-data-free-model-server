timestamp=$(date +%s)
log_dir="logs/fastapi_$timestamp"
mkdir -p "$log_dir"
echo "Running unit tests..."
python -m unittest unit_tests/test.py
echo "Done running unit tests, launching server..."
uvicorn server:app --host 0.0.0.0 --port 8000 --log-level warning