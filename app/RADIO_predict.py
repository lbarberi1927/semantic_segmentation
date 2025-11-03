from typing import List
from PIL import Image

from torchvision.transforms.functional import pil_to_tensor

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from app.base import Predictor

try:
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import torch
from torch.nn import functional as F


MANUAL_MEMORY_PURGE = False


class RADIO_Predictor(Predictor):
    def __init__(
        self,
        adaptor: str = "siglip",
    ):
        """
        args:
            adaptor (str): the adaptor to use for the RADIO model
        """
        super().__init__(manual_memory_purge=MANUAL_MEMORY_PURGE)
        self.adaptor = adaptor
        self._load_model()

    def _load_model(self):
        """Load the RADIO model.

        Returns:
            torch.nn.Module: the loaded RADIO model
        """
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="radio_v2.5-h",
            progress=True,
            adaptor_names=self.adaptor,
        )

    def _run_model(self, image_tensor: torch.Tensor, vocabulary: List[str]):
        self.model.to(device=self.device).eval()

        adaptor = self.model.adaptors[self.adaptor]
        tokenizer = adaptor.tokenizer

        # prepare tokens and run model
        tokens = tokenizer(vocabulary).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device))
            summary, features = output[self.adaptor]
            text_embeddings = adaptor.encode_text(tokens)

        # normalized similarity
        f_norm = F.normalize(features, p=2, dim=2)  # (1, N, D)
        t_norm = F.normalize(text_embeddings, p=2, dim=1)  # (M, D)
        sim = torch.einsum("bnd,md->bnm", f_norm, t_norm)  # (1, N, M)
        sim = sim.squeeze(0).T  # (M, N)

        # place-holder H/W; could be derived from model.patch_size or summary shape
        H, W = 48, 48
        result = sim.reshape(sim.shape[0], H, W)

        return result

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Resize the input image to have the maximum side length of max_size.

        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the resized image
        """
        x = pil_to_tensor(image).to(dtype=torch.float32, device="cuda")
        x = x.unsqueeze(0)
        nearest_res = (
            self.model.preferred_resolution
        )  # self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

        return x

    def _postprocess(self, model_output: torch.Tensor) -> dict:
        return {"result": model_output.cpu().numpy()}


if __name__ == "__main__":
    vocab = {
        "gravel_path": {"navigation_cost": 0.0},
        "dense_vegetation": {"navigation_cost": 0.5},
        "trees": {"navigation_cost": 1.0},
        "grass": {"navigation_cost": 0.3},
        "person": {"navigation_cost": 1.0},
    }
    predictor = RADIO_Predictor()
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
