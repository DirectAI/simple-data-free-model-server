FROM python:3.9-slim

WORKDIR /
COPY .mypy.ini
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /integration_tests

RUN mkdir test_modules
COPY test_modules test_modules
RUN mkdir sample_data
COPY sample_data sample_data

WORKDIR /integration_tests

COPY test.py .
COPY __init__.py .


WORKDIR /

CMD ["python", "-m", "integration_tests.test"]