# Open Vocabulary Semantic Segmentation
This repository contains code to run open-vocabulary semantic segmentation using different models. At the moment, the options implemented are:
- Side Adapter Network (SAN) [[Paper]](https://arxiv.org/abs/2302.12242) [[Project Page]](https://mendelxu.github.io/SAN/)
- Grounded Segment Anything Model (Grounded-SAM)[[Project Page]](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#grounded-segment-anything)
- Grounded Segment Anything Model 2 (Grounded-SAM 2)[[Project Page]](https://github.com/IDEA-Research/Grounded-SAM-2)

### Installation and Setup

To run the code, please use docker for an easy setup. Depending on which model you want to use,
please run the corresponding model server file to start the docker container.

Example:
```bash
# For SAN
bash bin/run_san_server.sh
# For Grounded-SAM
bash bin/run_sam_server.sh
# For Grounded-SAM 2
bash bin/run_sam2_server.sh
```
This will start a docker container with all the dependencies installed and the model loaded.
The model servers will be running on port the following ports:
- SAN: 7860
- Grounded-SAM: 7870
- Grounded-SAM 2: 7880
