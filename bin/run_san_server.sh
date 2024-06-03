#!/usr/bin/bash
docker run --gpus all --shm-size 4G -p 7860:7860 mktk1117/san /app/SAN/app/server.py
