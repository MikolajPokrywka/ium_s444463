FROM ubuntu:latest
FROM python:3.8
RUN apt update 


# Stwórzmy w kontenerze (jeśli nie istnieje) katalog /app i przejdźmy do niego (wszystkie kolejne polecenia RUN, CMD, ENTRYPOINT, COPY i ADD będą w nim wykonywane)
COPY ./requirements.txt .
RUN apt-get update
RUN pip3 install -r requirements.txt
RUN apt-get install zip unzip --yes


WORKDIR /app

COPY ./process_data.sh .
COPY ./download_data_and_process.py .
COPY ./stats.py .
COPY ./real-or-fake-fake-jobposting-prediction.zip .
RUN chmod +x process_data.sh
CMD python3 download_data_and_process.py
