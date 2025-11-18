from typing import List

import numpy as np
from PIL import Image

from torchvision.transforms.functional import pil_to_tensor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from app.base import Predictor
from rayfronts.naradio import NARadioEncoder
from rayfronts.utils import compute_cos_sim

try:
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import torch
from torch.nn import functional as F


MANUAL_MEMORY_PURGE = True


class RayFronts_Predictor(Predictor):
    def __init__(
        self,
    ):
        super().__init__(manual_memory_purge=MANUAL_MEMORY_PURGE)
        self._load_model()

    def _load_model(self):
        """Load the RADIO model.

        Returns:
            torch.nn.Module: the loaded RADIO model
        """
        self.model = NARadioEncoder(
            device=self.device,
            input_resolution=(512, 512),
            model_version="radio_v2.5-h",
            compile=False,
        )

    def _run_model(self, image_tensor: torch.Tensor, vocabulary: List[str]):
        with torch.no_grad():
            text_features = self.model.encode_labels(vocabulary)
            image_features = self.model.encode_image_to_feat_map(image_tensor)

            aligned_image_features = self.model.align_spatial_features_with_language(
                image_features
            )

            resolution = self.model.input_resolution[0]
            chunk_size: int = 10000
            feat_map = torch.nn.functional.interpolate(
                aligned_image_features, resolution, mode="bilinear", antialias=True
            )
            feat_map = feat_map.squeeze(0).permute(1, 2, 0)
            H, W, C = feat_map.shape
            feat_map = feat_map.reshape(-1, C)
            num_chunks = int(np.ceil(feat_map.shape[0] / chunk_size))

            cos_sim = list()
            for c in range(num_chunks):
                cos_sim.append(
                    compute_cos_sim(
                        text_features,
                        feat_map[c * chunk_size : (c + 1) * chunk_size],
                        softmax=True,
                    )
                )
            cos_sim = torch.cat(cos_sim, dim=0)
            N = len(vocabulary)
            cos_sim = cos_sim.reshape(H, W, N)
            cos_sim = cos_sim.permute(2, 0, 1)  # N x H x W

        return cos_sim

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Resize the input image to have the maximum side length of max_size.

        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the resized image
        """
        image = self.load_image(image)
        x = pil_to_tensor(image).to(dtype=torch.float32, device="cuda") / 255.0
        x = x.unsqueeze(0)
        input_res = self.model.input_resolution
        x = F.interpolate(x, input_res)

        return x

    def _postprocess(self, model_output: torch.Tensor) -> dict:
        resize = (427, 640)  # original image size
        """model_output = F.interpolate( model_output.unsqueeze(0),
        size=resize, mode="bilinear",

        align_corners=False, ).squeeze(0)
        """
        result = model_output.detach().cpu().numpy()
        result = np.ascontiguousarray(result)
        return {"result": result}


if __name__ == "__main__":
    vocab = {
        "gravel_path": {"navigation_cost": 0.0},
        "dense_vegetation": {"navigation_cost": 0.5},
        "trees": {"navigation_cost": 1.0},
        "grass": {"navigation_cost": 0.3},
        "person": {"navigation_cost": 1.0},
    }
    predictor = RayFronts_Predictor()
    result = predictor.predict(
        "/home/zivi/hiking_images/test/000000.jpg",
        list(vocab.keys()),
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
