import gc
import os
from typing import List, Union, Tuple

import cv2
import numpy as np
from torchvision.transforms.functional import pil_to_tensor

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[1]  # project root
_openworld_dir = _project_root / "OpenWorldSAM"
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_openworld_dir))
from app.base import Predictor
from model import add_open_world_sam2_config
from train_net import Trainer
from datasets.dataset_mappers.open_world_sam_semantic_dataset_mapper import (
    beit3_preprocess,
    sam_preprocess,
)

import torch
from PIL import Image
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup, DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, random_color


config_file = "/home/zivi/Repositories/semantic_segmentation/OpenWorldSAM/configs/ade20k/semantic-segmentation/Open-World-SAM2-CrossAttention.yaml"
model_file = "/app/OpenWorldSAM/checkpoints/model_final.pth"
RETURN_LOGITS = True
MANUAL_MEMORY_PURGE = True


def setup(config_file):
    """Create configs and perform basic setups."""

    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Add this line before merging the file
    add_open_world_sam2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="open-world-sam2"
    )
    return cfg


class OWSAM_Predictor(Predictor):
    def __init__(
        self,
        config_file: str = config_file,
    ):
        super().__init__(MANUAL_MEMORY_PURGE)
        self.config_file = config_file
        self._load_model()

    def _load_model(self):
        cfg = setup(self.config_file)
        self.model = DefaultTrainer.build_model(cfg)
        self.model.training = False
        self.model.two_stage_inference = True
        # self.model.metadata = MetadataCatalog.get(cfg['DATASETS']['TEST'][0])
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS
        )

    def insert_labels_into_templates(self, labels: List[str]) -> List[List[str]]:
        """Inserts each labels into a set of stored templates.

        Args:
          labels: A list of length T of class names / labels. Ex. ['cat', 'dog']
        Returns:
          labeled_templates as a list of length T of a list of length P of strings.
          where T is the number of labels and P is the number of templates.
        """
        return [[pt(x) for pt in openai_imagenet_template] for x in labels]

    def _run_model(
        self,
        image_tensor: Tuple[torch.Tensor, torch.Tensor],
        vocabulary: List[str],
    ) -> Union[torch.Tensor, Tuple]:
        """Predict the segmentation result.

        Args:
            image_tensor (torch.Tensor): the input image tensor
            vocabulary (List[str]): the vocabulary used for segmentation
        Returns:
            Union[torch.Tensor, Tuple]: the segmentation result
        """
        self.model.to(device=self.device).eval()
        prompts = vocabulary  # self.insert_labels_into_templates(vocabulary)
        colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
        MetadataCatalog.get("_temp").set(stuff_classes=vocabulary, stuff_colors=colors)
        metadata = MetadataCatalog.get("_temp")
        self.model.metadata = metadata
        sam_image, evf_image = image_tensor
        height = sam_image.size(1)
        width = sam_image.size(2)
        with torch.no_grad():
            output = self.model(
                [
                    {
                        "image": sam_image,
                        "evf_image": evf_image,
                        "height": height,
                        "width": width,
                        "prompt": prompts,
                        "unique_categories": range(len(vocabulary)),
                    }
                ]
            )

        output = output[0]["sem_seg"]

        print("output", output.shape, flush=True)

        self.sem_seg = output.argmax(dim=0).cpu().numpy()
        self.vocabulary = vocabulary

        return output

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Preprocess the input image.

        Args:
            image (Image.Image): the input image
        Returns:
            the same image
        """
        self.image = image.resize((1024, 1024))
        image_tensor = pil_to_tensor(image)
        beit3_image = beit3_preprocess(image_tensor.double())
        sam_image = sam_preprocess(np.array(image))
        return sam_image, beit3_image

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

    def visualize(
        self,
        output_file: str = None,
        mode: str = "overlay",
    ) -> Union[Image.Image, None]:
        """Visualize the segmentation result.

        Args:
            image (Image.Image): the input image
            sem_seg (np.ndarray): the segmentation result
            vocabulary (List[str]): the vocabulary used for the segmentation
            output_file (str): the output file path
            mode (str): the visualization mode, can be "overlay" or "mask"
        Returns:
            Image.Image: the visualization result. If output_file is not None, return None.
        """
        # add temporary metadata
        # set numpy seed to make sure the colors are the same
        np.random.seed(0)
        metadata = MetadataCatalog.get("_temp")
        if mode == "overlay":
            v = Visualizer(self.image, metadata)
            v = v.draw_sem_seg(self.sem_seg, area_threshold=0).get_image()
            v = Image.fromarray(v)
        else:
            v = np.zeros((self.image.size[1], self.image.size[0], 3), dtype=np.uint8)
            labels, areas = np.unique(self.sem_seg, return_counts=True)
            sorted_idxs = np.argsort(-areas).tolist()
            labels = labels[sorted_idxs]
            for label in filter(lambda l: l < len(metadata.stuff_classes), labels):
                v[self.sem_seg == label] = metadata.stuff_colors[label]
            v = Image.fromarray(v)
        # remove temporary metadata
        MetadataCatalog.remove("_temp")
        if output_file is None:
            return v
        v.save(output_file)
        print(f"saved to {output_file}")


if __name__ == "__main__":
    vocab = {
        "runway": {"navigation_cost": 0.0},
        "plane": {"navigation_cost": 0.5},
        "car": {"navigation_cost": 1.0},
        "grass": {"navigation_cost": 0.3},
        "trees": {"navigation_cost": 1.0},
    }
    predictor = OWSAM_Predictor()
    result = predictor.predict(
        "/home/zivi/Downloads/release_test/testing/test/ADE_test_00000001.jpg",
        list(vocab.keys()),
    )
    predictor.visualize("segmentation_result.png", mode="overlay")

    result_tensor = result["result"]
    vacabulary = result["vocabulary"]
    print("result_tensor", result_tensor.shape)
    # print("seg_map", seg_map)
    # print(result[0])
    import matplotlib.pyplot as plt
    import seaborn as sns

    for i, vocab in enumerate(vacabulary):
        plt.figure(figsize=(10, 10))
        sns.heatmap(result_tensor[i])
        plt.savefig(f"{vocab}_heatmap.png")
        plt.close()
