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

import torch
import torch.nn.functional as F
from PIL import Image



class Predictor(object):
    def __init__(self, backbone_path: str = "SAN/dinov3/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", weight_path: str = "SAN/dinov3/dinov3/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"):
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
        print("Original image size:", w, h, flush=True)
        image_tensor: torch.Tensor = self._preprocess(image_data).unsqueeze(0)
        print("image tensor shape:", image_tensor.shape, flush=True)
        vocabulary = list(set([v.lower().strip() for v in vocabulary]))
        # remove invalid vocabulary
        vocabulary = [v for v in vocabulary if v != ""]
        ori_vocabulary = vocabulary

        with torch.no_grad():
            # Move image to device and extract features
            image_tensor = image_tensor.to(self.device)

            # Encode text tokens for each vocabulary item
            tokenized_vocab = self.tokenizer.tokenize(vocabulary).cuda()
            if self.device.type == "cuda":
                tokenized_vocab = tokenized_vocab.cuda()
            _, text_features, _, patch_tokens, _ = self.model(
                image_tensor, tokenized_vocab
            )
            text_features = text_features[:, 1024:]

            print("patch shape:", patch_tokens.shape, flush=True)
            print("text_features shape:", text_features.shape, flush=True)

            B, P, D = patch_tokens.shape  # number of patches
            H, W = 40, 59#int(P ** 0.5)
            x = patch_tokens.movedim(2, 1).unflatten(2, (H, W)).float() # [C, H, W]
            print("x shape after reshape:", x.shape, flush=True)
            x = F.interpolate(x, size=(427, 640), mode="bicubic", align_corners=False)
            print("x shape after interpolate:", x.shape, flush=True)
            x = F.normalize(x, p=2, dim=1)
            y = F.normalize(text_features.float(), p=2, dim=1)
            result = torch.einsum("bdhw,cd->bchw", x, y).squeeze(0)
            print("result shape:", result.shape, flush=True)
        seg_map = self._postprocess(result, ori_vocabulary)
        print("seg_map shape:", seg_map.shape)

        return {
            "result": result,
            "image": image_data,
            "sem_seg": seg_map,
            "vocabulary": vocabulary,
        }

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess the input image.
        Args:
            image (Image.Image): the input image
        Returns:
            torch.Tensor: the preprocessed image
        """
        image = image.convert("RGB")
        # resize short side to 640
        w, h = image.size
        if w < h:
            image = image.resize((640, int(h * 640 / w)))
        else:
            image = image.resize((int(w * 640 / h), 640))
        image = torch.from_numpy(np.asarray(image)).float()
        image = image.permute(2, 0, 1)
        return image

    def _postprocess(
            self, result: torch.Tensor, ori_vocabulary: List[str]
    ) -> np.ndarray:
        """
        Postprocess the segmentation result.
        Args:
            result (torch.Tensor): the segmentation result
            ori_vocabulary (List[str]): the original vocabulary used for the segmentation
        Returns:
            np.ndarray: the postprocessed segmentation result
        """
        result = result.argmax(dim=0).cpu().numpy()  # (H, W)
        print("segmap result", result, flush=True)
        print("segmap result shape", result.shape, flush=True)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)
        print("segmap after adjust", result, flush=True)
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
