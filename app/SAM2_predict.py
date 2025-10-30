import gc
import os
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
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForCausalLM
import cv2

import torch
from PIL import Image

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
FLORENCE2_MODEL_ID = "microsoft/Florence-2-large"
SAM2_CHECKPOINT = "/app/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
RETURN_LOGITS = True
MANUAL_MEMORY_PURGE = True

# Force python GC and clear the CUDA cache
gc.collect()
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

class SAM2_Predictor(object):
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

        # Building florence2 inference model
        self.florence2_model = AutoModelForCausalLM.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True, torch_dtype="auto").eval()
        self.florence2_processor = AutoProcessor.from_pretrained(FLORENCE2_MODEL_ID, trust_remote_code=True)

        # build sam 2
        self.sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT)
        self.sam2_predictor = None


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
        image_data = Image.open(image_data_or_path).convert("RGB")

        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]

        #image_data = self.resize_image(image_data, max_size=1024)

        with torch.no_grad():

            # Step 1: Object Detection using Florence2
            detections = self.object_detection(image_data, vocabulary)["<OPEN_VOCABULARY_DETECTION>"]
            classes = detections["bboxes_labels"]

            # Force python GC and clear the CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Step 2: Generate masks using SAM2
            # convert detections to masks
            masks, logits = self.segment(
                image=np.array(image_data),
                boxes=detections["bboxes"]
            )

        if RETURN_LOGITS:
            result = logits
        else:
            result = self._postprocess(masks, classes, vocabulary)

        if MANUAL_MEMORY_PURGE:
            # Clean up temporaries and free CUDA memory (keep models loaded by default)
            try:
                del detections
            except Exception:
                pass

            self.purge_memory(unload_model=True)

        return {
            "result": result,
            "image": image_data,
            "vocabulary": vocabulary,
        }

    def purge_memory(self, unload_model: bool = False):
        """
        Free Python and CUDA memory. If unload_model is True, attempt to move model weights to CPU and remove references.
        """
        try:
            # delete transient predictor (holds activations)
            if hasattr(self, "sam2_predictor") and self.sam2_predictor is not None:
                try:
                    del self.sam2_predictor
                except Exception:
                    pass
                self.sam2_predictor = None

            # optionally move models to CPU and drop references
            if unload_model:
                try:
                    if hasattr(self, "sam2_model"):
                        self.sam2_model.to("cpu")
                except Exception:
                    pass
                try:
                    if hasattr(self, "florence2_model"):
                        self.florence2_model.to("cpu")
                except Exception:
                    pass

                try:
                    del self.sam2_model
                except Exception:
                    pass
                try:
                    del self.florence2_model
                except Exception:
                    pass
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()

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
        florence_2_vocab = " <and> ".join(vocabulary)
        prompt = "<OPEN_VOCABULARY_DETECTION>" + florence_2_vocab
        self.florence2_model = self.florence2_model.to(self.device)
        inputs = self.florence2_processor(text=prompt, images=image, return_tensors="pt")

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

        generated_text = self.florence2_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.florence2_processor.post_process_generation(
            generated_text,
            task="<OPEN_VOCABULARY_DETECTION>",
            image_size=(image.width, image.height)
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

    def segment(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        # Ensure SAM is on the GPU only now (lazy move / initialize)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model.to(self.device))
        self.sam2_predictor.set_image(image)
        masks, scores, logits = self.sam2_predictor.predict(
            box=boxes,
            multimask_output=False
        )

        if logits.ndim == 4:
            logits = logits.squeeze(1)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks, logits

    def _postprocess(
            self, mask, classes, vocabulary: List[str]
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        n_boxes, h, w = mask.shape

        out = torch.zeros((len(vocabulary), h, w), dtype=torch.uint8, device=self.device)

        for i in range(n_boxes):
            cls_id = vocabulary.index(classes[i])
            m = torch.from_numpy(mask[i]).to(device=self.device).bool()  # (h, w)
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
