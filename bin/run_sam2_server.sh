#!/usr/bin/bash
docker run --privileged --gpus all --shm-size 4G -p 7880:7860 lbarberi/sam2:dev /app/app/server.py
