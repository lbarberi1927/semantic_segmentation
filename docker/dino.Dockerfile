FROM nvcr.io/nvidia/pytorch:25.02-py3
LABEL image=dino

RUN pip install cython scipy shapely timm h5py submitit scikit-image wandb setuptools numpy Pillow pycocotools~=2.0.4 fvcore tabulate tqdm ftfy regex opencv-python open_clip_torch cityscapesscripts tensorboard flask
RUN pip install omegaconf scikit-learn termcolor torch torchmetrics torchvision

COPY ./dinov3/ /app/SAN/dinov3/
# RUN pip install --no-cache-dir /app/SAN
# RUN pip install .
# RUN pip install --no-cache-dir /app/SAN/dinov3
# RUN ln -s /app/SAN/resources/clip /root/.cache/clip
