FROM ubuntu:latest
FROM python:3.8
RUN apt update 


COPY ./requirements.txt .
RUN apt-get update
RUN pip3 install -r requirements.txt
RUN apt-get install zip unzip --yes


WORKDIR /app

COPY ./deepl.py .
COPY ./MLProject .

# CMD python3 deepl.py 10
