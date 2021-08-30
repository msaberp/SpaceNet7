# Build: docker build -t <project_name> .
# Run: docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

ENV CONDA_ENV_NAME=myenv
ENV PYTHON_VERSION=3.8
ENV HOME=/home

# Basic setup
RUN apt update
RUN apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   wget \
                   && rm -rf /var/lib/apt/lists
RUN apt update &&  apt install ffmpeg libsm6 libxext6  -y

# Set working directory
WORKDIR /workspace/project


# Install Miniconda and create main env
ADD https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh miniconda3.sh 
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"


# Switch to bash shell
SHELL ["/bin/bash", "-c"]


# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt 


# Uncomment this to install Apex for mixed-precision support
# RUN source activate ${CONDA_ENV_NAME} \
#     && git clone https://github.com/NVIDIA/apex \
#     && cd apex  \
#     && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
#     && cd .. \
#     && rm -r apex

RUN chmod 777 $HOME/
# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc

