import time
from typing import List, Union

import numpy as np
import torchvision

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv
import cv2



import torch
import torch.nn.functional as F
from PIL import Image

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

class Predictor(object):
    def __init__(
            self,
    ):
        """
        Args:
            config_file (str): the config file path
            model_path (str): the model path
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device, flush=True)
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH,
                                     model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

        # Building SAM Model and SAM Predictor
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)


    def predict(
        self,
        image_data_or_path: Union[Image.Image, str],
        vocabulary: List[str] = [],
    ) -> Union[dict, None]:
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
        Returns:
            Union[dict, None]: the segmentation result
        """
        image_data = cv2.imread(image_data_or_path)

        print("Original image:", image_data.shape, flush=True)
        #image_tensor: torch.Tensor = self._preprocess(image_data).unsqueeze(0)
        #print("image tensor shape:", image_tensor.shape, flush=True)
        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]
        ori_vocabulary = vocabulary

        with torch.no_grad():
            # Move image to device and extract features
            #image_tensor = image_tensor.to(self.device)

            # Step 1: Object Detection using GroundingDINO
            detections = self.object_detection(image_data, vocabulary)
            print("Detections:", detections, flush=True)

            # NMS post process
            print(f"Before NMS: {len(detections.xyxy)} boxes")
            nms_time = time.time()
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            print(f"After NMS: {len(detections.xyxy)} boxes, done in {time.time() - nms_time:.3f}s", flush=True)

            # Step 2: Generate masks using SAM
            # convert detections to masks
            detections.mask = self.segment(
                image=cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

            print("Generated masks shape:", detections.mask.shape, flush=True)
            print("Generated masks", detections.mask, flush=True)

        return {
            "result": result,
            "image": image_data,
            #"sem_seg": seg_map,
            "vocabulary": vocabulary,
        }


    def object_detection(self, image: torch.Tensor, vocabulary: List[str]):
        """
        Perform object detection on the input image using GroundingDINO.
        Args:
            image (torch.Tensor): the input image
            vocabulary (List[str]): the vocabulary used for detection
        Returns:
            List[dict]: list of detected objects with bounding boxes and labels
        """
        #caption_vocab = ". ".join(vocabulary)
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=vocabulary,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        return detections

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)


    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.
        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the preprocessed image
        """
        image = image.convert("RGB")
        # resize short side to 640
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image, None)  # 3, h, w
        return image

    def _postprocess(
            self, result: torch.Tensor, ori_vocabulary: List[str]
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = result.argmax(dim=0).cpu().numpy()  # (H, W)
        print("segmap result", result, flush=True)
        print("segmap result shape", result.shape, flush=True)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)
        print("segmap after adjust", result, flush=True)
        return result


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True, help="path to config file"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="path to model file"
    )
    parser.add_argument(
        "--img-path", type=str, required=True, help="path to image file."
    )
    parser.add_argument("--aug-vocab", action="store_true", help="augment vocabulary.")
    parser.add_argument(
        "--vocab",
        type=str,
        default="",
        help="list of category name. seperated with ,.",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    predictor = Predictor(config_file=args.config_file, model_path=args.model_path)
    result = predictor.predict(
        args.img_path,
        args.vocab.split(","),
        args.aug_vocab,
        output_file=args.output_file,
    )
    print("result", result)

    result_tensor = result["result"]
    vacabulary = result["vocabulary"]
    print("result_tensor", result_tensor.shape)
    # print("seg_map", seg_map)
    # print(result[0])
    import matplotlib.pyplot as plt
    import seaborn as sns

    for i, vocab in enumerate(vacabulary):
        plt.figure(figsize=(10, 10))
        sns.heatmap(result_tensor[i].cpu().numpy())
        plt.savefig(f"{vocab}_heatmap.png")
        plt.close()
