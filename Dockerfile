FROM python:3.7

CMD mkdir pulearn

WORKDIR /pulearn
COPY requirements.txt /pulearn/
COPY data_models/corpus_embeddings.pt /pulearn/data_models/
COPY app.py /pulearn/
COPY config.py /pulearn/
COPY /data_creation.py /pulearn/

RUN apt-get update && apt-get install -y unixodbc-dev gcc g++
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN chmod -R 777 app.py

