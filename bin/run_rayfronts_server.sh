docker run --privileged --gpus all --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all --shm-size 4G -p 7890:7860 lbarberi/rayfronts  /app/app/server.py
