from typing import List, Union, Tuple

import cv2
import numpy as np

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
from model import add_open_world_sam2_config, OpenWorldSAM2
from train_net import Trainer
from datasets import OpenWorldSAM2InstanceDatasetMapper

import torch
from PIL import Image
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, random_color


config_file = "/app/OpenWorldSAM/configs/ade20k/semantic-segmentation/Open-World-SAM2-CrossAttention.yaml"
checkpoints_path = "/app/OpenWorldSAM/checkpoints/"
RETURN_LOGITS = True
MANUAL_MEMORY_PURGE = True


def setup(config_file, checkpoints_path):
    """Create configs and perform basic setups."""

    cfg = get_cfg()
    cfg.set_new_allowed(True)  # Add this line before merging the file
    add_open_world_sam2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = checkpoints_path + "model_final.pth"
    cfg.MODEL.OpenWorldSAM2.ENCODER_PRETRAINED = (
        checkpoints_path + "beit3_large_patch16_224.pth"
    )
    cfg.MODEL.OpenWorldSAM2.VISION_PRETRAINED = checkpoints_path + "sam2_hiera_large.pt"
    cfg.MODEL.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="open-world-sam2"
    )
    return cfg


class OWSAM_Predictor(Predictor):
    def __init__(
        self,
        config_file: str = config_file,
        checkpoints_path: str = checkpoints_path,
    ):
        super().__init__(MANUAL_MEMORY_PURGE)
        self.config_file = config_file
        self.checkpoints_path = checkpoints_path
        self._load_model()

    def _load_model(self):
        cfg = setup(self.config_file, self.checkpoints_path)
        model_dict = OpenWorldSAM2.from_config(cfg)
        self.model = OpenWorldSAM2(**model_dict)
        self.model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)["model"], strict=False)
        self.model.to(device=self.device).eval()
        self.model.training = False
        self.mapper = OpenWorldSAM2InstanceDatasetMapper(cfg, is_train=False)

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
        prompts = vocabulary  # self.insert_labels_into_templates(vocabulary)
        colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
        MetadataCatalog.get("_temp").set(stuff_classes=vocabulary, stuff_colors=colors)
        metadata = MetadataCatalog.get("_temp")
        self.model.metadata = metadata

        inputs = self.mapper(image_tensor)
        inputs["prompt"] = prompts
        inputs["unique_categories"] = range(len(vocabulary))

        with torch.no_grad():
            output = self.model([inputs])

        output = output[0]["sem_seg"]
        MetadataCatalog.remove("_temp")

        return output

    def _preprocess(self, image_path: str) -> dict:
        """Preprocess the input image.

        Args:
            image_path (str): the image path
        Returns:
            dict: the preprocessed image and its metadata
        """
        cv2_image = cv2.imread(image_path)
        height, width = cv2_image.shape[:2]
        dataset_dict = {
            "file_name": image_path,
            "image_id": 0,
            "height": height,
            "width": width,
        }
        return dataset_dict

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
        # center values around 0
        result = result - torch.mean(result)
        if type(result) == torch.Tensor:
            out = result.cpu().numpy()
        else:
            out = result
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
        "person": {"navigation_cost": 0.0},
        "gravel_path": {"navigation_cost": 0.5},
        "dense_vegetation": {"navigation_cost": 1.0},
        "grass": {"navigation_cost": 0.3},
        "trees": {"navigation_cost": 1.0},
        "other": {"navigation_cost": 1.0},
    }
    predictor = OWSAM_Predictor(
        config_file="/home/zivi/Repositories/semantic_segmentation/OpenWorldSAM/configs/ade20k/semantic-segmentation/Open-World-SAM2-CrossAttention.yaml",
        checkpoints_path="/home/zivi/Repositories/semantic_segmentation/OpenWorldSAM/checkpoints/",
    )
    result = predictor.predict(
        "/home/zivi/hiking_images/test/000000.jpg",
        list(vocab.keys()),
    )
    predictor.visualize("results/segmentation_result.png", mode="overlay")

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
        plt.savefig(f"results/{vocab}_heatmap.png")
        plt.close()
