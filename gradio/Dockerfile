FROM python:3.10-slim 

WORKDIR /gradio

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY interface.py .
COPY modeling.py .

EXPOSE 7860

CMD ["python", "interface.py"]