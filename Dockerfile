FROM ubuntu:latest
RUN apt update 


# Stwórzmy w kontenerze (jeśli nie istnieje) katalog /app i przejdźmy do niego (wszystkie kolejne polecenia RUN, CMD, ENTRYPOINT, COPY i ADD będą w nim wykonywane)
COPY ./requirements.txt .
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip  
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip install  kaggle
RUN apt-get install zip unzip

ARG CUTOFF
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV CUTOFF=${CUTOFF}
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

# Skopiujmy nasz skrypt do katalogu /app w kontenerze
RUN mkdir /data

WORKDIR /app

COPY ./process_data.sh .
COPY ./download_data_and_process.py .
COPY ./stats.py .
RUN chmod+x process_data.sh
RUN ./process_data.sh
