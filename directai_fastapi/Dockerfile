FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
WORKDIR /directai_fastapi

RUN apt-get update
RUN apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y

RUN apt-get install cmake build-essential -y
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pydantic_models.py .
COPY server.py .
COPY run.sh .
COPY utils.py .

RUN mkdir unit_tests
COPY unit_tests unit_tests

RUN chmod +x run.sh
CMD ./run.sh