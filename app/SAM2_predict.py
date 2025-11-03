import gc
from typing import List, Union, Tuple

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from app.base import Predictor

import torch
from PIL import Image

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "/app/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
RETURN_LOGITS = True
MANUAL_MEMORY_PURGE = True


class SAM2_Predictor(Predictor):
    def __init__(
        self,
    ):
        super().__init__(MANUAL_MEMORY_PURGE)
        self._load_model()

    def _load_model(self):
        """Load the Florence2 and SAM2 models."""
        # Building florence2 inference model
        self.florence2_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype="auto"
        ).eval()
        self.florence2_processor = AutoProcessor.from_pretrained(
            FLORENCE2_MODEL_ID, trust_remote_code=True
        )

        # build sam 2
        self.model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT)

    def _run_model(
        self,
        image_tensor: torch.Tensor,
        vocabulary: List[str],
    ) -> Union[torch.Tensor, Tuple]:
        """Predict the segmentation result.

        Args:
            image_tensor (torch.Tensor): the input image tensor
            vocabulary (List[str]): the vocabulary used for segmentation
        Returns:
            Union[torch.Tensor, Tuple]: the segmentation result
        """
        with torch.no_grad():

            # Step 1: Object Detection using Florence2
            detections = self.object_detection(image_tensor, vocabulary)[
                "<OPEN_VOCABULARY_DETECTION>"
            ]
            classes = detections["bboxes_labels"]

            # Force python GC and clear the CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 2: Generate masks using SAM2
            # convert detections to masks
            masks, logits = self.segment(
                image=np.array(image_tensor), boxes=detections["bboxes"]
            )

        if RETURN_LOGITS:
            result = logits
        else:
            result = masks, classes, vocabulary

        return result

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess the input image.

        Args:
            image (Image.Image): the input image
        Returns:
            the same image
        """
        return image

    def object_detection(self, image: torch.Tensor, vocabulary: List[str]):
        """Perform object detection on the input image using GroundingDINO.

        Args:
            image (torch.Tensor): the input image
            vocabulary (List[str]): the vocabulary used for detection
        Returns:
            List[dict]: list of detected objects with bounding boxes and labels
        """
        florence_2_vocab = " <and> ".join(vocabulary)
        prompt = "<OPEN_VOCABULARY_DETECTION>" + florence_2_vocab
        self.florence2_model.to(self.device)
        inputs = self.florence2_processor(
            text=prompt, images=image, return_tensors="pt"
        )

        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        with torch.autocast(self.device.type):
            generated_ids = self.florence2_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

        generated_text = self.florence2_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]
        parsed_answer = self.florence2_processor.post_process_generation(
            generated_text,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(image.width, image.height),
        )
        # --- Free GPU activations and cache before running SAM ---
        try:
            # make sure any pending CUDA work finishes
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass

        self.florence2_model.to("cpu")

        return parsed_answer

    def segment(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform segmentation on the input image using SAM2.

        Args:
            image (np.ndarray): the input image
            boxes (np.ndarray): the bounding boxes for segmentation
        Returns:
            Tuple[np.ndarray, np.ndarray]: the segmentation masks and logits
        """
        # Ensure SAM is on the GPU only now (lazy move / initialize)
        self.predictor = SAM2ImagePredictor(self.model.to(self.device))
        self.predictor.set_image(image)
        masks, scores, logits = self.predictor.predict(
            box=boxes, multimask_output=False
        )

        if logits.ndim == 4:
            logits = logits.squeeze(1)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks, logits

    def _postprocess(
        self,
        result: Union[torch.Tensor, Tuple],
    ) -> dict:
        """Postprocess the segmentation result.

        Args:
            result (torch.Tensor): the segmentation result
        Returns:
            dict: the postprocessed segmentation result
        """
        if RETURN_LOGITS:
            if type(result) == torch.Tensor:
                out = result.cpu().numpy()
            else:
                out = result
        else:
            mask, classes, vocabulary = result
            n_boxes, h, w = mask.shape

            out = torch.zeros(
                (len(vocabulary), h, w), dtype=torch.uint8, device=self.device
            )

            for i in range(n_boxes):
                cls_id = vocabulary.index(classes[i])
                m = torch.from_numpy(mask[i]).to(device=self.device).bool()  # (h, w)
                out[cls_id, m] = 1

            out = out.cpu().numpy()

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
    predictor = SAM2_Predictor(config_file=args.config_file, model_path=args.model_path)
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
