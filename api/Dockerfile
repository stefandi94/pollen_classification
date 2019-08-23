FROM ubuntu:16.04
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install software-properties-common python-software-properties -y
RUN apt-get install build-essential checkinstall -y
RUN apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y
RUN apt-get install python3 python3-dev -y
RUN apt-get install python3-pip -y
RUN pip3 install --upgrade pip

RUN mkdir /code
WORKDIR /code
COPY . /code/

RUN pip3 install -r requirements.txt

EXPOSE 8000
CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]
