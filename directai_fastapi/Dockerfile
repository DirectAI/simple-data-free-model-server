FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
WORKDIR /directai_fastapi

RUN apt-get update
RUN apt-get install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 -y

RUN apt-get install git -y

RUN apt-get install cmake build-essential -y
COPY requirements.txt .
RUN pip install -r requirements.txt

# replace pillow with pillow-simd for faster image processing using CPU SIMD and AVX2
RUN apt-get install libjpeg-dev zlib1g-dev -y
RUN pip uninstall -y pillow
RUN CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

COPY logging_config.py .
COPY pydantic_models.py .
COPY server.py .
COPY run.sh .
COPY utils.py .

COPY classify_directory.py .

RUN mkdir unit_tests
COPY unit_tests unit_tests

RUN mkdir modeling
COPY modeling modeling

RUN mkdir reproduction_scripts
COPY reproduction_scripts reproduction_scripts

ENV PYTHONPATH "${PYTHONPATH}:/directai_fastapi"

RUN chmod +x run.sh
CMD ./run.sh