FROM python:3.6

EXPOSE 5001
WORKDIR /Sentiment
COPY requirements.txt /Sentiment
RUN apt-get update
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]
COPY . /Sentiment
CMD ["python", "app.py"]
