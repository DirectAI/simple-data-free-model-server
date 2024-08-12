# ray start --head --num-gpus=$(nvidia-smi -L | wc -l) --ray-client-server-port=10001
python -m unittest unit_tests/test.py
uvicorn server:app --host 0.0.0.0 --port 8000 --log-level warning