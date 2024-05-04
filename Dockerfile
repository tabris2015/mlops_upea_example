FROM python:3.11-slim

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY ml_package ml_package

WORKDIR /ml_package

COPY dataset.csv dataset.csv
COPY new_data.csv new_data.csv

CMD [ "/bin/bash" ]