#FROM python:3.8-slim-buster
FROM python:3.9-slim-buster
# Install python packages
RUN apt-get update && apt-get install -y gcc g++
RUN pip install mlflow boto3 pymysql