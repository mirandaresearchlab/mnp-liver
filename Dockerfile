FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG UID
ARG GID
ARG USER
ARG GROUP

SHELL [ "/bin/bash", "--login", "-c" ]

# install utilities
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    apt-utils \
    cmake \
    git \
    curl \
    ca-certificates \
    sudo \
    bzip2 \
    libx11-6 \
    wget \
    ssh-client \
    libjpeg-dev \
    bash-completion \
    libgl1-mesa-dev \
    ffmpeg \
    tmux \
    screen \
    htop \
    # nfs-common \
    cifs-utils \
    zip \
    unzip \
    # pydf \
    # aria2 \
    # mdadm \
    # net-tools \
    uidmap \
    # vim \
    # nano \
    # graphviz \
    # openslide-tools \
    libjemalloc-dev \
    openssh-client \
    libpng-dev \
    # python3-mpi4py \
    libopenmpi-dev \
    mpich \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ENV HOME /home/$USER

# Create a non-root user
RUN addgroup --gid $GID $GROUP \
    && adduser --disabled-password \
    --gecos "root" \
    --uid $UID \
    --gid $GID \
    --shell /bin/bash \
    --home $HOME \
    $USER 
WORKDIR $HOME
# switch to that user
# USER $USER

ENV MINICONDA_VERSION py311_24.9.2-0
 # latest
# if you want a specific version (you shouldn't) replace "latest" with that, e.g. ENV MINICONDA_VERSION py38_4.8.3

ENV CONDA_DIR=$HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p $CONDA_DIR \
    && rm ~/miniconda.sh

# add conda to path (so that we can just use conda install <package> in the rest of the dockerfile)
ENV PATH $CONDA_DIR/bin:$PATH

# make conda activate command available from /bin/bash --login shells
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile

# make conda activate command available from /bin/bash --interative shells
RUN conda init bash

# build the conda environment
# ENV ENV_PREFIX $HOME/env

# Create and configure Conda environment
RUN conda update -n base -c defaults conda \
    && conda create -n dev python=3.10.16 \
    && conda clean --all --yes

# Activate the created dev env
SHELL ["conda", "run", "--no-capture-output", "-n", "dev", "/bin/bash", "-c"]
RUN echo "source activate dev" > ~/.bashrc

RUN conda install -c anaconda pip \ 
    # && conda install seaborn colorama \
    # && conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia \
    && conda install -c conda-forge matplotlib \ 
        pandas numpy pillow \
        scikit-learn \
        scikit-image \
        easydict \
        pyyaml \
    && conda clean --all --yes

# Install pip packages as non-root user
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    "ipykernel>=6.29.5" \
    "matplotlib==3.10.3" \
    "numpy>=1.23.2,<2.3.0" \
    "pandas>=1.5.3,<2.3.0" \
    "scikit-learn>=1.6.1" \
    "seaborn==0.13.2" \
    "umap-learn==0.5.7" \
    "plotly==6.1.1" \
    "nbformat==5.10.4" \
    "kaleido==0.2.1" \
    "ipywidgets>=8.1.7"

# RUN pip install opencv-python==4.1.2.30

# RUN apt-get update -y \
#     # && apt-get upgrade -y \
#     && apt-get install -y python3-mpi4py \
#     && apt-get install -y libopenmpi-dev \
#     && apt-get install -y mpich \
#     && rm -rf /var/lib/apt/lists/* \
#     && pip3 install --upgrade pip \
#     && pip install --no-cache-dir mpi4py 
# RUN pip3 install --upgrade pip \
#     && pip install --no-cache-dir mpi4py 

ENV SHELL=/bin/bash

# Use C.UTF-8 locale to avoid issues with ASCII encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN env > /root/env.txt #&& cron -f

RUN /bin/bash -c "source activate dev"

CMD ["/bin/bash"]
# CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]