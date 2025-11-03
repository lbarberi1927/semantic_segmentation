#!/usr/bin/bash
docker run --privileged --gpus all --shm-size 4G -p 7890:7860 lbarberi/radio /app/app/server.py
