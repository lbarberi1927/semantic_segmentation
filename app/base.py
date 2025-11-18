import gc
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from PIL import Image
import torch


class Predictor(ABC):
    def __init__(self, manual_memory_purge: bool = False):
        """
        Common predictor setup: device selection, optional manual memory purge.
        Subclasses should implement _load_model() and _run_model().
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device, flush=True)
        self.manual_memory_purge = manual_memory_purge
        print(
            "Manual memory purge is",
            "enabled" if self.manual_memory_purge else "disabled",
            flush=True,
        )
        self.model = None
        self.predictor = None
        self._setup_memory()

    def _setup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()

    @abstractmethod
    def _load_model(self):
        """Load and return the model.

        Set self.model and self.predictor as needed.
        """
        raise NotImplementedError

    @abstractmethod
    def _run_model(
        self, image_tensor: torch.Tensor, vocabulary: List[str]
    ) -> torch.Tensor:
        """Run model inference.

        Return a tensor with predictions.
        """
        raise NotImplementedError

    @abstractmethod
    def _postprocess(self, model_output: torch.Tensor) -> dict:
        """Optional postprocess over raw model output.

        Return final response dict with key "result" and predictions as value.
        """
        raise NotImplementedError

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Optional preprocessing: pil to tensor, cast and move to device. Subclasses may override."""
        raise NotImplementedError

    def purge_memory(self):
        """Free Python and CUDA memory and unload model to CPU and drop
        reference."""
        try:
            if self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
                try:
                    del self.model
                except Exception:
                    pass
                self.model = None
            if self.predictor is not None:
                try:
                    del self.predictor
                except Exception:
                    pass
                self.predictor = None
        finally:
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                torch.cuda.empty_cache()

    def load_image(self, image_data_or_path: Union[Image.Image, str]) -> Image.Image:
        """Load image from path or return the given PIL image."""
        if isinstance(image_data_or_path, Image.Image):
            return image_data_or_path
        else:
            return Image.open(image_data_or_path).convert("RGB")

    def predict(
        self, image_data_or_path: Union[Image.Image, str], vocabulary: List[str] = []
    ):
        """General predict flow: normalize vocab, ensure model loaded, preprocess, run, postprocess."""
        # normalize vocabulary
        vocabulary = list({v.lower().strip() for v in vocabulary if isinstance(v, str)})
        image_tensor = self._preprocess(image_data_or_path)

        # run model-specific logic
        raw_output = self._run_model(image_tensor, vocabulary)
        result = self._postprocess(raw_output)

        if self.manual_memory_purge:
            try:
                del raw_output
            except Exception:
                pass
            self.purge_memory()

        image = self.load_image(image_data_or_path)
        # include original image and vocabulary in response
        result.setdefault("image", image)
        result.setdefault("vocabulary", vocabulary)
        return result
