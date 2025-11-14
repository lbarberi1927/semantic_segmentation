#!/usr/bin/bash
docker run --privileged --gpus all --shm-size 4G -p 7900:7860 lbarberi/owsam /app/app/server.py
