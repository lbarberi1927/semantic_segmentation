FROM nvcr.io/nvidia/pytorch:22.08-py3
LABEL image=san

RUN pip install cython scipy shapely timm h5py submitit scikit-image wandb setuptools numpy Pillow pycocotools~=2.0.4 fvcore tabulate tqdm ftfy regex opencv-python open_clip_torch cityscapesscripts tensorboard flask
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN pip install opencv-python==4.5.5.62
RUN apt-get update && apt-get install libgl1 -y

WORKDIR /app
COPY . /app/SAN
RUN pip install --no-cache-dir /app/SAN
# RUN pip install .
RUN ln -s /app/SAN/resources/clip /root/.cache/clip
