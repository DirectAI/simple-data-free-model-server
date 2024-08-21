FROM python:3.9-slim

WORKDIR /gradio

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY interface.py .

EXPOSE 7860

CMD ["python", "interface.py"]