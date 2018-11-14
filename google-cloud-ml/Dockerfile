from debian:stretch
MAINTAINER luiza.orosanu@gmail.com

ENV \
  LANG=C.UTF-8 \
  TZ=Europe\Paris

RUN apt-get update \
  && apt-get install -y curl gnupg \
  && apt-get install -y python-pip python-dev build-essential

RUN \
  echo "deb http://packages.cloud.google.com/apt cloud-sdk-stretch main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

RUN apt-get update \
  && apt-get install -y google-cloud-sdk \
  && apt-get install -y google-cloud-sdk-app-engine-java \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /src
WORKDIR /src

COPY requirements.txt setup.py /src/
RUN pip install -r requirements.txt

CMD gcloud
