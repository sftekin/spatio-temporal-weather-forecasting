FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ENV DEBIAN_FRONTEND=noninteractive

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN pip install matplotlib==3.2.1
RUN pip install pandas==1.0.3
RUN pip install netCDF4==1.5.3

WORKDIR /workspace
