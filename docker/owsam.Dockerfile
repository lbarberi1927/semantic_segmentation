FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
# Arguments to build Docker Image using CUDA
ARG USE_CUDA=1
ARG TORCH_ARCH="7.0;7.5;8.0;8.6+PTX"

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV CUDA_HOME=/usr/local/cuda-12.1/
# Ensure CUDA is correctly set up
ENV PATH=/usr/local/cuda-12.1/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH}

COPY ./OpenWorldSAM/requirements.txt /app/OpenWorldSAM/requirements.txt
WORKDIR /app/OpenWorldSAM
RUN pip install -r requirements.txt
RUN pip install opencv-python==4.5.5.62
RUN apt-get update && apt-get install libgl1 -y && apt-get install libglib2.0-0 -y

COPY ./OpenWorldSAM/model/segment_anything_2 /app/OpenWorldSAM/model/segment_anything_2
WORKDIR /app/OpenWorldSAM/model/segment_anything_2
RUN python setup.py build_ext --inplace

RUN pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+2a420edpt2.5.0cu121
RUN pip install flask einops "transformers==4.50.3"

COPY ./OpenWorldSAM /app/OpenWorldSAM/
COPY ./app /app/app/
