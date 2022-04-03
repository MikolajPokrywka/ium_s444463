FROM ubuntu:latest
FROM python:3.8
RUN apt update 


# Stwórzmy w kontenerze (jeśli nie istnieje) katalog /app i przejdźmy do niego (wszystkie kolejne polecenia RUN, CMD, ENTRYPOINT, COPY i ADD będą w nim wykonywane)
WORKDIR /app
COPY ./requirements.txt .
RUN pip3 install -r ./requirements.txt
RUN pip3 install kaggle
# Skopiujmy nasz skrypt do katalogu /app w kontenerze
COPY ./process_data.sh ./
COPY ./download_data_and_process.py ./
COPY ./stats.py ./

ARG CUTOFF
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV CUTOFF=${CUTOFF}
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}
# Domyślne polecenie, które zostanie uruchomione w kontenerze po jego starcie
CMD python3 -u ./download_data_and_process.py