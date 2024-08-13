FROM python:3.9-slim

WORKDIR /
RUN mkdir test_modules
COPY test_modules test_modules

WORKDIR /integration_tests

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY test.py .

WORKDIR /

CMD ["python", "-m", "integration_tests.test"]