# Dockerfile may have two Arguments: tag, branch
# tag - tag for the Base image, (e.g. 1.10.0-py3 for tensorflow)
# branch - user repository branch to clone (default: main, other option: test)

ARG tag=1.14.0-py3

# Base image, e.g. tensorflow/tensorflow:1.12.0-py3
FROM tensorflow/tensorflow:${tag}

# Add container's metadata to appear along the models metadata
ENV CONTAINER_MAINTAINER "Lara Lloret Iglesias <lloret@ifca.unican.es>"
ENV CONTAINER_VERSION "0.1"
ENV CONTAINER_DESCRIPTION "DEEP as a Service Container: Image Classification"

# What user branch to clone (!)
ARG branch=main

# Install ubuntu updates and python related stuff
# link python3 to python, pip3 to pip, if needed
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
         git \
         curl \
         wget \
         psmisc \
         python3-setuptools \
         python3-pip \
         python3-wheel && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/* && \
    python --version && \
    pip --version

RUN pip install --upgrade pip setuptools wheel

# Set LANG environment
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# Install rclone
RUN wget https://downloads.rclone.org/rclone-current-linux-amd64.deb && \
    dpkg -i rclone-current-linux-amd64.deb && \
    apt install -f && \
    mkdir /srv/.rclone/ && touch /srv/.rclone/rclone.conf && \
    rm rclone-current-linux-amd64.deb && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

# Disable FLAAT authentication by default
ENV DISABLE_AUTHENTICATION_AND_ASSUME_AUTHENTICATED_USER yes

# Useful tool to debug extensions loading
RUN pip install --no-cache-dir entry_point_inspector && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/*

ENV SHELL /bin/bash

# Initialization scripts
# deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/ai4os/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# Needed for open-cv
RUN apt-get update && apt-get install -y libgl1

# Install user app:
RUN git clone -b $branch https://github.com/ai4os-hub/ai4os-image-classification-tf && \
    cd ai4os-image-classification-tf && \
    pip install --no-cache-dir -e . && \
    rm -rf /root/.cache/pip/* && \
    rm -rf /tmp/* && \
    cd ..

# Download network weights
ENV SWIFT_CONTAINER https://api.cloud.ifca.es:8080/swift/v1/imagenet-tf/
ENV MODEL_TAR default_imagenet.tar.xz

RUN curl --insecure -o ./ai4os-image-classification-tf/models/${MODEL_TAR} \
    ${SWIFT_CONTAINER}${MODEL_TAR}

RUN cd ai4os-image-classification-tf/models && \
        tar -xf ${MODEL_TAR}

# Open ports (deepaas, monitoring, ide)
EXPOSE 5000 6006 8888

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000"]
