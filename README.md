# Open Vocabulary Semantic Segmentation
This repository contains code to run open-vocabulary semantic segmentation using different models. At the moment, the options implemented are:
- Side Adapter Network (SAN) [[Paper]](https://arxiv.org/abs/2302.12242) [[Project Page]](https://mendelxu.github.io/SAN/)
- Grounded Segment Anything Model (Grounded-SAM)[[Project Page]](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#grounded-segment-anything)
- Grounded Segment Anything Model 2 (Grounded-SAM 2)[[Project Page]](https://github.com/IDEA-Research/Grounded-SAM-2)

### Installation and Setup

To clone this repo including its submodules, please run:
```bash
git clone --recurse-submodules <repo_url>
```

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

### Possible Issues
There are a few possible issues you might encounter when running the code:
1. Grounded-SAM-2 `RuntimeError: No available kernel. Aborting execution.`
   This is a common issue addressed in the [Grounded-SAM-2 repo](https://github.com/facebookresearch/sam2/issues/48)
2. Grounded-SAM-2 `value.type() is deprecated` warning:
   I could not ignore this warning, and fixed it by changing lines 30 and 52 of semantic_segmentation/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h from `value.type().is_cuda()` to `value.options().is_cuda()`.

### Usage
You will need to download the model weights for the models you want to use. Please follow the instructions in the respective project pages linked above.
