FROM ubuntu:16.04
#RUN mkdir -p /home/stefan/docker_project/
#COPY . /home/stefan/docker_project/
WORKDIR /build
COPY . .
RUN apt-get update && \
    apt-get install sudo -y && \
    apt-get install python python-dev -y && \
    apt-get install python-pip -y && \
    python -m pip install --upgrade pip setuptools wheel && \
    apt-get install python3-pip -y && \
    pip3 install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    mkdir /install
#RUN cd /home/stefan/PycharmProjects/aaaa/
#COPY requirements.txt /tmp
#COPY main.py /tmp
#COPY settings.py /tmp
#COPY source /tmp
#COPY utils /tmp
WORKDIR /install
COPY requirements.txt .
RUN pip3 install -r requirements.txt
CMD python3 play.py
