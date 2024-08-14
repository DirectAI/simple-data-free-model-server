FROM python:3.9-slim

WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir test_modules
COPY test_modules test_modules
RUN mkdir sample_data
COPY sample_data sample_data

WORKDIR /integration_tests

COPY test.py .

WORKDIR /

CMD ["python", "-m", "integration_tests.test"]