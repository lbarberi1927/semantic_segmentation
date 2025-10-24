import math
from typing import List, Union

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os
import torchvision.transforms as TVT


import torch
import torch.nn.functional as F
from PIL import Image



class Predictor(object):
    def __init__(
            self,
            backbone_path: str = "SAN/dinov3/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            weight_path: str = "SAN/dinov3/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
    ):
        """
        Args:
            config_file (str): the config file path
            model_path (str): the model path
        """
        print("Loading model from: ", weight_path)
        # load dinov3 text model and load state dict
        self.model, self.tokenizer = torch.hub.load("SAN/dinov3", 'dinov3_vitl16_dinotxt_tet1280d20h24l', source='local', backbone_weights=backbone_path, weights=weight_path)
        print("Loaded model from: ", weight_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.cuda()

    def predict(
        self,
        image_data_or_path: Union[Image.Image, str],
        vocabulary: List[str] = [],
        augment_vocabulary: Union[str, bool] = True,
        output_file: str = None,
    ) -> Union[dict, None]:
        """
        Predict the segmentation result.
        Args:
            image_data_or_path (Union[Image.Image, str]): the input image or the image path
            vocabulary (List[str]): the vocabulary used for the segmentation
            augment_vocabulary (bool): whether to augment the vocabulary
            output_file (str): the output file path
        Returns:
            Union[dict, None]: the segmentation result
        """
        if isinstance(image_data_or_path, str):
            image_data = Image.open(image_data_or_path)
        else:
            image_data = image_data_or_path
        w, h = image_data.size
        image_tensor: torch.Tensor = self._preprocess(image_data)
        print("image tensor shape:", image_tensor.shape, flush=True)

        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]
        ori_vocabulary = vocabulary

        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)

            _, blocks_feats = self.encode_image(image_tensor.unsqueeze(0))  # [1, h, w, D]

            tokens = self.tokenizer.tokenize(vocabulary).to("cuda", non_blocking=True)
            text_feats = self.model.encode_text(tokens)  # [num_prompts, 2D]
            text_feats = text_feats[:, text_feats.shape[1] // 2:]  # The 1st half of the features corresponds to the CLS token, drop it
            print("text_features shape:", text_feats.shape, flush=True)
            text_feats = F.normalize(text_feats, p=2, dim=-1)  # Normalize each text embedding

            _, H, W = image_tensor.shape
            _, h, w, _ = blocks_feats.shape
            blocks_feats = blocks_feats.squeeze(0)  # [h, w, D]
            print("H, W, h, w:", H, W, h, w, flush=True)

            # Cosine similarity between patch features and text features (already normalized)
            blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)  # [h, w, D]
            cos = torch.einsum("cd,hwd->chw", text_feats, blocks_feats)  # [num_classes, h, w]
            print("cosine shape:", cos.shape, flush=True)
            print("cosine:", cos, flush=True)

        seg_map = self._postprocess(cos, ori_vocabulary, H, W)
        print("seg_map shape:", seg_map.shape)

        return {
            "result": cos,
            "image": image_data,
            "sem_seg": seg_map,
            "vocabulary": vocabulary,
        }

    def encode_image(self, image_tensor: torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        B, _, H, W = image_tensor.shape
        P = self.model.visual_model.backbone.patch_size
        new_H = math.ceil(H / P) * P
        new_W = math.ceil(W / P) * P

        # Stretch image to a multiple of patch size
        if (H, W) != (new_H, new_W):
            image_tensor = F.interpolate(image_tensor, size=(new_H, new_W), mode="bicubic",
                                         align_corners=False)  # [B, 3, H', W']

        print("Stretched image shape:", image_tensor.shape, flush=True)
        print("patch size:", P, flush=True)

        B, _, h_i, w_i = image_tensor.shape

        backbone_patches = None
        _, _, patch_tokens = self.model.visual_model.get_class_and_patch_tokens(image_tensor)
        print("patch tokens shape:", patch_tokens.shape, flush=True)
        blocks_patches = (
            patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
        )  # [1, h, w, D]
        print("blocks patches shape:", blocks_patches.shape, flush=True)

        return backbone_patches, blocks_patches

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.
        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the preprocessed image
        """
        image = image.convert("RGB")
        image = torch.from_numpy(np.asarray(image)).float()
        image = image.permute(2, 0, 1)
        return image

    def _postprocess(
            self, result: torch.Tensor, ori_vocabulary: List[str], H, W
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = F.interpolate(result.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)
        result = result.argmax(dim=0).cpu().numpy()  # (H, W)
        print("segmap result", result, flush=True)
        print("segmap result shape", result.shape, flush=True)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)
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
