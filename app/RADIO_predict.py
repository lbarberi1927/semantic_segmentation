import gc

from typing import List, Union

import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import torch
from torch.nn import functional as F


MANUAL_MEMORY_PURGE = False

# Force python GC and clear the CUDA cache
gc.collect()
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


class RADIO_Predictor(object):
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
        self.adaptor = "siglip"

        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version="radio_v2.5-h",
            progress=True,
            adaptor_names=self.adaptor,
        )

    def predict(
        self,
        image_data_or_path: Union[Image.Image, str],
        vocabulary: List[str] = [],
    ) -> Union[dict, None]:
        """Predict the segmentation result.

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

        image_tensor = self._preprocess(image_data)
        print("model patch size: ", self.model.patch_size)
        print("image size: ", image_tensor.shape, flush=True)
        print("preferred resolution: ", self.model.preferred_resolution, flush=True)

        with torch.no_grad():
            self.model.to(device=self.device.type).eval()

            adaptor = self.model.adaptors[self.adaptor]
            tokenizer = adaptor.tokenizer
            tokens = tokenizer(vocabulary).to(self.device.type)
            output = self.model(image_tensor)
            summary, features = output[self.adaptor]
            print("siglip summary shape:", summary.shape, flush=True)
            print("siglip features shape:", features.shape, flush=True)

            text_embeddings = adaptor.encode_text(tokens)
            print("text embeddings shape:", text_embeddings.shape, flush=True)

        f_norm = F.normalize(features, p=2, dim=2)  # normalize along feature dimension
        t_norm = F.normalize(text_embeddings, p=2, dim=1)

        # Compute similarity: (1, N, D) x (M, D) → (1, N, M)
        sim = torch.einsum("bnd,md->bnm", f_norm, t_norm)

        sim = sim.squeeze(0).T

        H, W = 48, 48
        result = sim.reshape(sim.shape[0], H, W)

        if MANUAL_MEMORY_PURGE:
            # Clean up temporaries and free CUDA memory (keep models loaded by default)
            try:
                del summary, features, text_embeddings
            except Exception:
                pass

            self.purge_memory(unload_model=True)

        return {
            "result": result,
            "image": image_data,
            "vocabulary": vocabulary,
        }

    def purge_memory(self, unload_model: bool = False):
        """Free Python and CUDA memory.

        If unload_model is True, attempt to move model weights to CPU and
        remove references.
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

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize the input image to have the maximum side length of max_size.

        Args:
            image (np.ndarray): the input image
        Returns:
            np.ndarray: the resized image
        """
        x = pil_to_tensor(image).to(dtype=torch.float32, device="cuda")
        x = x.unsqueeze(0)
        nearest_res = (
            self.model.preferred_resolution
        )  # self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

        return x

    def _postprocess(self, features, text_embeddings) -> np.ndarray:
        """Postprocess the segmentation result.

        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        pass


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
