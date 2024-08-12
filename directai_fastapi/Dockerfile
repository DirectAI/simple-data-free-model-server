FROM tiangolo/uvicorn-gunicorn:python3.10-slim-2023-04-03 as install

WORKDIR /directai_fastapi

# opencv dep
RUN apt-get update && apt-get install -y git wget

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

RUN apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y

RUN apt-get install cmake build-essential zip -y