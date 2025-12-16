# Open Vocabulary Semantic Segmentation
This repository contains code to run open-vocabulary semantic segmentation using different models. At the moment, the options implemented are:
- Side Adapter Network (SAN) [[Paper]](https://arxiv.org/abs/2302.12242) [[Project Page]](https://mendelxu.github.io/SAN/)
- Grounded Segment Anything Model (Grounded-SAM)[[Project Page]](https://github.com/IDEA-Research/Grounded-Segment-Anything?tab=readme-ov-file#grounded-segment-anything)
- Grounded Segment Anything Model 2 (Grounded-SAM 2)[[Project Page]](https://github.com/IDEA-Research/Grounded-SAM-2)
- RayFronts 2D Open Vocabulary Segmentation [[Paper]](https://arxiv.org/abs/2504.06994) [[Project Page]](https://github.com/RayFronts/RayFronts)
- OpenWorldSAM [[Paper]](https://arxiv.org/abs/2507.05427)[[Project Page]](https://github.com/lbarberi1927/semantic_segmentation)

### Installation and Setup

To clone this repo including its submodules, please run:
```bash
git clone --recurse-submodules <repo_url>
```

To run the code, please use docker for an easy setup. Depending on which model you want to use,
please run the corresponding model server file to start the docker container.

### Running the Model Servers
To start the model servers, please run one of the following commands in the terminal:
```bash
# For SAN
bash bin/run_san_server.sh
# For Grounded-SAM
bash bin/run_sam_server.sh
# For Grounded-SAM 2
bash bin/run_sam2_server.sh
# For RayFronts
bash bin/run_rayfronts_server.sh
# For OpenWorldSAM
bash bin/run_owsam_server.sh
```
This will start a docker container with all the dependencies installed and the model loaded.
The model servers will be running on port the following ports:
- SAN: 7860
- Grounded-SAM: 7870
- Grounded-SAM 2: 7880
- RayFronts: 7890
- OpenWorldSAM: 7900

### Common Issues
There are a few possible issues you might encounter when running the code:
1. Grounded-SAM-2 `RuntimeError: No available kernel. Aborting execution.`
   This is a common issue addressed in the [Grounded-SAM-2 repo](https://github.com/facebookresearch/sam2/issues/48)
2. Grounded-SAM-2 `value.type() is deprecated` warning:
   I could not ignore this warning, and fixed it by changing lines 30 and 52 of `semantic_segmentation/Grounded-SAM-2/grounding_dino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h` from `value.type().is_cuda()` to `value.options().is_cuda()`.
3. OpenWorldSAM `NotImplementedError: Cannot copy out of meta tensor; no data!`: This seems to be a [common issue](https://github.com/huggingface/transformers/issues/31104) that I fixed by specifying the
OpenWorldSAM requirements to install `transformers==4.50.3`. This and other problems were solved by also installing other package versions, specifically:
`ray==2.51.1` and `bitsandbytes==0.42.0`.


### Usage
You will need to download the model weights for the models you want to use. Please follow the instructions in the respective project pages linked above.

### Sample Output
Below are some sample outputs from the different models using the same input image and text prompts. The code used to produce
these outputs cannot be shared publicly at the moment.
![example_output.png](docs/images/example_output.png)
