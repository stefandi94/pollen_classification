FROM python:3.6-slim

RUN mkdir -p /home/stefan/docker_project/

WORKDIR /home/stefan/docker_project/

COPY source /home/stefan/docker_project/
COPY requirements.txt /home/stefan/docker_project/
COPY utils /home/stefan/docker_project/
COPY settings.py /home/stefan/docker_project/
COPY main.py /home/stefan/docker_project/

RUN pip install -r requirements.txt

CMD ["python", "app.py"]
