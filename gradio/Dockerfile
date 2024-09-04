FROM python:3.10-slim 

WORKDIR /gradio

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY interface.py .
COPY modeling.py .
COPY utils.py .
COPY illustrating.py .

CMD ["python", "interface.py"]