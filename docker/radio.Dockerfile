FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

RUN pip install einops transformers flask timm open_clip_torch
WORKDIR /app
COPY ./app /app/app
#RUN ln -s /app/SAN/resources/clip /root/.cache/clip
