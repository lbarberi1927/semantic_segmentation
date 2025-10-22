#!/usr/bin/env python3
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import torch
import numpy as np

import io
import base64

# Assuming Predictor is a class from your script
from DINO_predict import Predictor

app = Flask(__name__)
"""
config_file = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "../configs/san_clip_vit_large_res4_coco.yaml",
)
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../resources/san_vit_large_14.pth")
predictor = Predictor(config_file=config_file, model_path=model_path)
"""
predictor = Predictor()

log_dir = "/san_logs"
os.makedirs(log_dir, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image part", 400
    image = request.files["image"]
    vocab = request.form.get("vocab", "")  # Default to empty string if not provided
    return_all_categories = request.form.get("return_all_categories", "False") == "True"
    print("return_all_categories", return_all_categories)

    if image.filename == "":
        return "No selected file", 400

    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(log_dir, filename)
        image.save(image_path)

        # Process with your existing script
        start_time = time.time()
        result = predictor.predict(image_path, vocab.split(","), False)
        print("prediction time: ", time.time() - start_time)
        if not return_all_categories:
            result["vocabulary"] = result["vocabulary"][: len(vocab.split(","))]
            result["result"] = result["result"][: len(result["vocabulary"])]
        result.pop("image")
        result["shape"] = result["result"].shape
        print("result", result)

        # convert tensors to base64
        for k, v in result.items():
            print("k", k, "v", type(v))
            if isinstance(v, torch.Tensor):
                arr = v.cpu().numpy()
                arr = np.ascontiguousarray(arr)
                result[k] = base64.b64encode(arr).decode("utf-8")
            if isinstance(v, np.ndarray):
                result[k] = base64.b64encode(v).decode("utf-8")

        # Format and return the result
        return jsonify(result)

    return "Error processing request", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=7860)
