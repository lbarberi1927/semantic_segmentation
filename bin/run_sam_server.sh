#!/usr/bin/bash
docker run --privileged --gpus all --shm-size 4G -p 7870:7860 lbarberi/sam:dev bash /app/app/SAM_entrypoint.sh
