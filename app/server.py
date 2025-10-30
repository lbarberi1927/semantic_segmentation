#!/usr/bin/env python3
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import torch
import numpy as np

import io
import base64


def parse_segmentation_model(model_str):
    if model_str=="SAM2":
        from SAM2_predict import SAM2_Predictor
        return SAM2_Predictor()
    elif model_str=="SAM":
        from SAM_predict import SAM_Predictor
        return SAM_Predictor()
    elif model_str=="SAN":
        from SAN_predict import SAN_Predictor
        return SAN_Predictor()
    else:
        raise ValueError(f"Unknown segmentation model: {model_str}")

app = Flask(__name__)

log_dir = "/san_logs"
os.makedirs(log_dir, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image part", 400
    image = request.files["image"]
    vocab = request.form.get("vocab", "")  # Default to empty string if not provided
    return_all_categories = request.form.get("return_all_categories", "False") == "True"
    segmentation_model = request.form.get("segmentation_model", "SAN")
    predictor = parse_segmentation_model(segmentation_model)
    print("Segmentation model: ", segmentation_model)

    if image.filename == "":
        return "No selected file", 400

    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join(log_dir, filename)
        image.save(image_path)

        # Process with your existing script
        start_time = time.time()
        result = predictor.predict(image_path, vocab.split(","))
        prediction_time = time.time() - start_time
        print("prediction time: ", time.time() - start_time)
        if not return_all_categories:
            result["vocabulary"] = result["vocabulary"][: len(vocab.split(","))]
            result["result"] = result["result"][: len(result["vocabulary"])]
        result.pop("image")
        result["shape"] = result["result"].shape
        #print("result", result)

        # convert tensors to base64
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                arr = v.cpu().numpy()
                arr = np.ascontiguousarray(arr)
                result[k] = base64.b64encode(arr).decode("utf-8")
            if isinstance(v, np.ndarray):
                result[k] = base64.b64encode(v.astype(np.float32)).decode("utf-8")

        result["prediction_time"] = prediction_time
        # Format and return the result
        return jsonify(result)

    return "Error processing request", 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=7860)
