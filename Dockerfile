FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
LABEL authors="Lukas Heumos (lukas.heumos@posteo.net)" \
      description="Docker image containing all requirements for running machine learning on CUDA enabled GPUs"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    build-essential \
    lshw \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory and set it as default
RUN mkdir /app
RUN chmod 777 /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user 
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

 # Install Miniconda
RUN wget -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Update Conda
RUN conda update conda

# To get real time output we need to disable the stdout buffer
ENV PYTHONUNBUFFERED 1

# Enable colors
ENV TERM xterm-256color

######################################

# Install the conda environment

RUN sudo apt-key del 7fa2af80
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN sudo apt-get update
RUN sudo apt-get install -y libgl1
RUN sudo DEBIAN_FRONTEND="noninteractive"  apt-get install -y libglib2.0-0

RUN sudo DEBIAN_FRONTEND="noninteractive"  apt-get -y install tzdata
RUN sudo apt-get install -y --reinstall openmpi-bin libopenmpi-dev
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
RUN echo "source activate root_tissue_segmentation" >> ~/.bashrc
ENV PATH /home/user/miniconda/envs/root_tissue_segmentation/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name root-tissue-segmentation > root-tissue-segmentation_environment.yml

# Currently required, since mlflow writes every file as root!
USER root

