FROM tiangolo/uvicorn-gunicorn:python3.10-slim-2023-04-03
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY server.py .
COPY run.sh .
RUN chmod +x run.sh
CMD ./run.sh