#!/usr/bin/bash
docker run --gpus all --shm-size 4G -p 7860:7860 lbarberi/san /app/app/server.py
