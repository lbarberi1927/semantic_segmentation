import gc
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
SAM_ENCODER_VERSION = "vit_h"  # choose from "vit_h", "vit_l", "vit_b"
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
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=self.device,
        )

        # Building SAM Model and SAM Predictor
        self.sam_model = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)


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

        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]

        #image_data = self.resize_image(image_data, max_size=1024)

        with torch.no_grad():

            # Step 1: Object Detection using GroundingDINO
            detections = self.object_detection(image_data, vocabulary)

            # NMS post process
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # Force python GC and clear the CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 2: Generate masks using SAM
            # convert detections to masks
            detections.mask = self.segment(
                image=cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy
            )

            print("Generated masks shape:", detections.mask.shape, flush=True)

        seg_map = self._postprocess(detections.mask, detections.class_id, len(vocabulary))
        print("seg_map shape:", seg_map.shape, flush=True)

        return {
            "result": seg_map,
            "image": image_data,
            #"sem_seg": seg_map,
            "vocabulary": vocabulary,
        }

    def resize_image(self, image: np.ndarray, max_size: int = 512) -> np.ndarray:
        """
        resize the input image to have the maximum side length of max_size.
        Args:
            image (np.ndarray): the input image
            max_size (int): the maximum side length
        Returns:
            np.ndarray: the resized image
        """
        h, w, _ = image.shape
        scale = max_size / max(h, w)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"resized image from ({w}, {h}) to ({new_w}, {new_h})", flush=True)
        return image


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
        # --- Free GroundingDINO GPU activations and cache before running SAM ---
        try:
            # make sure any pending CUDA work finishes
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        # If GroundingDINO is a torch module, try moving it to CPU to free GPU params
        try:
            if hasattr(self.grounding_dino_model, "to"):
                self.grounding_dino_model.to("cpu")
        except Exception as e:
            print("Warning: could not move grounding dino to cpu:", e, flush=True)
        return detections

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        # Ensure SAM is on the GPU only now (lazy move / initialize)
        self.sam_predictor = SamPredictor(self.sam_model.to(self.device))
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

    def _postprocess(
            self, mask, classes, len_vocab
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        #result = result.argmax(dim=0).cpu().numpy()  # (H, W)
        n_boxes, h, w = mask.shape
        out = torch.zeros((len_vocab, h, w), dtype=torch.uint8, device=self.device)

        # Assign by iterating masks; later masks override earlier ones
        for i in range(n_boxes):
            cls_id = classes[i]
            m = torch.from_numpy(mask[i]).to(device=self.device)  # (h, w)
            out[cls_id, m] = 1

        out = out.cpu().numpy()
        return out


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
