import gc
from typing import List, Union, Tuple

import numpy as np
import torchvision

import torch
from PIL import Image
import cv2
import sys
from pathlib import Path

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root
from app.base import Predictor


# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/app/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = (
    "/app/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"
)

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/app/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
RETURN_LOGITS = False
MANUAL_MEMORY_PURGE = True


class SAM_Predictor(Predictor):
    def __init__(
        self,
    ):
        super().__init__(MANUAL_MEMORY_PURGE)
        self._load_model()

    def _load_model(self):
        """Load the GroundingDINO and SAM models."""
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device=self.device,
        )

        # Building SAM Model
        self.model = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH
        )
        self.predictor = SamPredictor(self.model)

    def _run_model(
        self,
        image_tensor: torch.Tensor,
        vocabulary: List[str],
    ) -> torch.Tensor:
        """Run the GroundingDINO + SAM model on the input image and vocabulary.

        Args:
            image_tensor (torch.Tensor): the input image tensor
            vocabulary (List[str]): the vocabulary used for segmentation
        Returns:
            torch.Tensor: the segmentation result
        """
        with torch.no_grad():

            # Step 1: Object Detection using GroundingDINO
            detections = self.object_detection(image_tensor, vocabulary)

            # NMS post process
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    NMS_THRESHOLD,
                )
                .numpy()
                .tolist()
            )

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

            # Force python GC and clear the CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 2: Generate masks using SAM
            # convert detections to masks
            detections.mask, logits = self.segment(
                image=cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
            )

        if RETURN_LOGITS:
            result = logits
        else:
            result = detections.mask, detections.class_id, len(vocabulary)

        return result

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array.

        Args:
            image (Image.Image): the input image
        Returns:
            np.ndarray: the converted image
        """
        image = np.array(image)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def object_detection(self, image: torch.Tensor, vocabulary: List[str]):
        """Perform object detection on the input image using GroundingDINO.

        Args:
            image (torch.Tensor): the input image
            vocabulary (List[str]): the vocabulary used for detection
        Returns:
            List[dict]: list of detected objects with bounding boxes and labels
        """
        # caption_vocab = ". ".join(vocabulary)
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=vocabulary,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
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

    def segment(
        self, image: np.ndarray, xyxy: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform segmentation on the input image using SAM.

        Args:
            image (np.ndarray): the input image
            xyxy (np.ndarray): bounding boxes for segmentation
        Returns:
            Tuple[np.ndarray, np.ndarray]: tuple of masks and logits
        """
        # Ensure SAM is on the GPU only now (lazy move / initialize)
        self.predictor = SamPredictor(self.model.to(self.device))
        self.predictor.set_image(image)
        result_masks = []
        result_logits = []
        for box in xyxy:
            masks, scores, logits = self.predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
            result_logits.append(logits[index])

        return np.array(result_masks), np.array(result_logits)

    def _postprocess(self, model_output: Union[torch.Tensor, tuple]) -> dict:
        """Postprocess the segmentation result.

        Args:
            model_output (Union[torch.Tensor, tuple]): the raw model output
        Returns:
            dict: dict with the final segmentation map
        """
        if type(model_output) == tuple:
            mask, classes, len_vocab = model_output
            n_boxes, h, w = mask.shape
            out = torch.zeros((len_vocab, h, w), dtype=torch.uint8, device=self.device)

            # Assign by iterating masks; later masks override earlier ones
            for i in range(n_boxes):
                cls_id = classes[i]
                m = torch.from_numpy(mask[i]).to(device=self.device)  # (h, w)
                out[cls_id, m] = 1

            out = out.cpu().numpy()
        else:
            out = model_output.cpu().numpy()

        return {"result": out}


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
