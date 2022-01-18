ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    

# Install MMCV
RUN pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone -b v2.11.0 https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
WORKDIR /
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
WORKDIR /
RUN git clone https://github.com/IuliuNovac/Swin-Transformer-Object-Detection.git /localmodel
WORKDIR /localmodel
RUN pip install -r requirements/build.txt
RUN pip install wandb
RUN python setup.py develop

# install the SageMaker Training Toolkit 
RUN pip install sagemaker-training
WORKDIR /
# copy the training script inside the container
COPY  /localmodel /opt/ml/code/

# define train.py as the script entry point
ENV SAGEMAKER_PROGRAM train.py
