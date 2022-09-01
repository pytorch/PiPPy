# Using cuda 11.3
FROM nvidia/cuda:11.3.1-devel-ubuntu18.04

# nvidia cuda 11.3 paths
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
ENV LIBRARY_PATH=${LIBRARY_PATH}:/usr/local/cuda-11.3/lib64

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

ENV LANG C.UTF-8

# Ignore `tzdata` asking questions
ENV DEBIAN_FRONTEND=noninteractive

RUN echo "US/Pacific" > /etc/timezone \
    && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
    && apt update && apt upgrade -y \
    && apt-get -y install build-essential checkinstall wget git \
	libreadline-gplv2-dev libncursesw5-dev libssl-dev \
	libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

# Set Python Version
ENV PYTHON_VERSION 3.9.12
ENV PYTHON_COMMAND 3.9

# Install Python from source.
RUN cd /opt \
	&& wget https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tgz \
	&& tar xzf Python-$PYTHON_VERSION.tgz \
	&& cd Python-$PYTHON_VERSION \
	&& ./configure --enable-optimizations \
	&& make altinstall \
    && ln -fs /usr/local/bin/python$PYTHON_COMMAND /usr/bin/python \
	&& ln -fs /usr/local/bin/python$PYTHON_COMMAND /usr/bin/python3 \
	&& ln -fs /usr/local/bin/pip$PYTHON_COMMAND /usr/bin/pip \
	&& ln -fs /usr/local/bin/pip$PYTHON_COMMAND /usr/bin/pip3 \
	&& cd /

# Install python libraries needed for CI test.
RUN pip3 install --upgrade pip \
	&& pip3 config set global.progress_bar off \
	&& pip3 install flake8 pytest pytest-cov pytest-shard numpy expecttest hypothesis pyyaml

LABEL versin="1.0.2"
LABEL description="Build docker image for ubuntu Linux OS with cuda 11.3 and Python."