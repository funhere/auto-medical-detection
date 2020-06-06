"""
Created on Fri Jun  5 12:53:10 2020

@author: simon.feng
"""

ARG ENV="cpu"
ARG HOME="/workspace"

FROM ubuntu:18.04 AS cpu

ARG HOME
ENV HOME="${HOME}"
WORKDIR ${HOME}

# Install base dependencies
RUN apt-get update && \
    apt-get install -y curl git build-essential
    

# Install the required packages
RUN pip3 install -r requirements.txt


# Install Anaconda
ARG ANACONDA="https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh"
RUN curl ${ANACONDA} -o anaconda.sh && \
    /bin/bash anaconda.sh -b -p conda && \
    rm anaconda.sh 
ENV PATH="${HOME}/conda/envs/cv/bin:${PATH}"


# Setup Jupyter notebook configuration
ENV NOTEBOOK_CONFIG="${HOME}/.jupyter/jupyter_notebook_config.py"
RUN mkdir ${HOME}/.jupyter && \
    echo "c.NotebookApp.token = ''" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.allow_root = True" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >> ${NOTEBOOK_CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >> ${NOTEBOOK_CONFIG}


# GPU Stage
FROM nvidia/cuda:9.0-base AS gpu

ARG HOME
WORKDIR ${HOME}

COPY --from=cpu ${HOME} .

ENV PATH="${HOME}/conda/envs/cv/bin:${PATH}"


# Final Stage
FROM $ENV AS final

# Install Conda dependencies
RUN conda env create -f video_resnet/environment.yml && \
    conda clean -fay && \
    python -m ipykernel install --user --name 'cv' --display-name 'python3'


ARG HOME
WORKDIR ${HOME}/video_resnet

EXPOSE 8888
CMD ["jupyter", "notebook"]
