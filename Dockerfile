FROM python:3.12-slim-buster
WORKDIR /app
COPY . /app

# RUN pip install --no-cache-dir -r requirements.txt
RUN apt update -y && apt install awscli -y 

RUN apt-get update && pip install -r requirements.txt
CMD ["python", "app.py"]