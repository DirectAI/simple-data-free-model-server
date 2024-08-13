FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /directai_fastapi

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY *.py ./
COPY run.sh .

RUN mkdir unit_tests
COPY unit_tests unit_tests

ENV PYTHONPATH "${PYTHONPATH}:/directai_fastapi"

RUN chmod +x run.sh
CMD ./run.sh