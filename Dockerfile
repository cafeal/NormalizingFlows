# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM tensorflow/tensorflow:nightly-gpu-py3

WORKDIR /work
ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

# RUN apt-get update \
#     && apt-get install -y \
#         curl git python3-dev python3-distutils

# RUN curl -kL https://bootstrap.pypa.io/get-pip.py | python3 \
#     && pip install git+https://github.com/pypa/pipenv

RUN apt-get update \
    && apt-get install -y git 
RUN pip install git+https://github.com/pypa/pipenv

COPY Pipfile /work/Pipfile
RUN pipenv install --system --dev --pre --skip-lock

EXPOSE 8888 6006
