FROM tensorflow/tensorflow:2.6.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install protoc and pip
RUN apt-get update && apt-get -y install \
    git \
    vim \
    wget \
    python-opencv \
    protobuf-compiler


# Add new user to avoid running as root
RUN useradd -ms /bin/bash tensorflow
USER tensorflow
WORKDIR /home/tensorflow

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Pillow, matplotlib, cython
RUN pip install pillow matplotlib cython

# Clone tf models to get Orbit source code
RUN git clone https://github.com/tensorflow/models.git

# Clone and compile pycocotools
RUN git clone https://github.com/cocodataset/cocoapi.git
RUN cd cocoapi/PythonAPI && make && cd -

# Copy deeplab2 to the home dir
COPY --chown=tensorflow . /home/tensorflow/deeplab2

# Config env vars 
ENV PYTHONPATH=/home/tensorflow:/home/tensorflow/models:/home/tensorflow/cocoapi/PythonAPI

# Compile deeplab2
RUN bash deeplab2/compile.sh


