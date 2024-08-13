FROM tiangolo/uvicorn-gunicorn:python3.10-slim-2023-04-03
WORKDIR /directai_fastapi

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pydantic_models.py .
COPY server.py .
COPY run.sh .

RUN mkdir unit_tests
COPY unit_tests unit_tests

RUN chmod +x run.sh
CMD ./run.sh