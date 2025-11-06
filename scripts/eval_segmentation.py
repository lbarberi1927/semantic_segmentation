"""This script is used to run semantic segmentation to produce and save masks
of images."""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from utils import ade_palette

parser = argparse.ArgumentParser(description="Process some images.")
parser.add_argument(
    "directories",
    type=str,
    nargs="+",
    default=["/home/zivi/hiking_images/test"],
    help="One or more directories containing images to process.",
)
parser.add_argument(
    "-m", "--model", type=str, default="segformer", help="The model type to use."
)
parser.add_argument(
    "-s",
    "--seg_model",
    type=str,
    default="nvidia/segformer-b5-finetuned-ade-640-640",
    help="The segmentation model name to use.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducible results (default: 42)",
)
parser.add_argument(
    "--nb_img_per_analysis",
    type=int,
    default=5,
    help="Number of image analyses to perform",
)

parser.add_argument(
    "--evaluate_segmentation",
    action="store_true",
    help="call script with all segmentation models and combine output into one file.",
)


args = parser.parse_args()

# Set random seed for reproducible results
random.seed(args.seed)
np.random.seed(args.seed)

# Supported image extensions we will look for
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
color_palette = ade_palette()

# Find the next available version number for the main result directory
version = 1
while True:
    versioned_result_dir = Path(f"result_v{version:03d}")
    if not versioned_result_dir.exists():
        break
    version += 1

versioned_result_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving all results to: {versioned_result_dir}")

seg_models_to_run = [args.seg_model]
if args.evaluate_segmentation:
    seg_models_to_run = [
        "nvidia/segformer-b5-finetuned-ade-640-640",
        "twdent/segformer-b5-finetuned-Hiking",
    ]

for dir_path in args.directories:
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        print(f"Skipping {dir_path}: not a valid directory")
        continue

    image_files = [
        p for p in dir_path.iterdir() if p.suffix.lower() in image_extensions
    ]
    if len(image_files) == 0:
        print(f"No images found in {dir_path}")
        continue
    image_files.sort()

    # Perform segmentation analysis multiple times
    sample_images = np.arange(0, len(image_files), args.nb_img_per_analysis)

    for lmm_idx, img_idx in enumerate(sample_images):
        # Prepare the result directory for this input directory
        result_dir = versioned_result_dir / dir_path.parent.name
        result_dir.mkdir(parents=True, exist_ok=True)

        # Process only 5 images and create a combined visualization
        next_img_idx = np.arange(
            img_idx, min(img_idx + args.nb_img_per_analysis, len(image_files))
        )
        selected_images = [
            image_files[curr_next_img_idx] for curr_next_img_idx in next_img_idx
        ]

        per_model_combined_paths = []
        for i, seg_model in enumerate(seg_models_to_run):
            image_processor = AutoImageProcessor.from_pretrained(seg_model)
            model = SegformerForSemanticSegmentation.from_pretrained(seg_model)

            processed_results = []
            include_originals = i == 0

            print(f"Using segmentation model: {seg_model}")

            for img_path in selected_images:
                img = Image.open(img_path)
                start_time = time.time()
                inputs = image_processor(images=img, return_tensors="pt")
                seg_return = model(**inputs)
                prediction_time = time.time() - start_time
                if seg_return is None:
                    print(f"Segmentation failed for {img_path}")
                    continue

                seg_logits = seg_return.logits
                upsampled_logits = torch.nn.functional.interpolate(
                    seg_logits,
                    size=img.size[::-1],
                    mode="bilinear",
                    align_corners=False,
                )
                seg_labels = upsampled_logits.argmax(dim=1)[0]
                color_seg = np.zeros(
                    (seg_labels.shape[0], seg_labels.shape[1], 3), dtype=np.uint8
                )
                for label, color in enumerate(color_palette):
                    color_seg[seg_labels == label, :] = color
                # Convert to BGR
                color_seg = color_seg[..., ::-1]

                # Show image + mask
                seg_img = np.array(img) * 0.5 + color_seg * 0.5
                seg_img = seg_img.astype(np.uint8)

                processed_results.append(
                    {
                        "img_path": img_path,
                        "img": img,
                        "seg_logits": seg_logits,
                        "seg_labels": seg_labels,
                        "seg_image": seg_img,
                        "prediction_time": prediction_time,
                    }
                )

            # Create combined visualization plot
            if processed_results:
                num_images = len(processed_results)
                # Create a figure with subplots: original images + segmentation masks
                # Decide whether to include original-image row for this segmentation model
                rows = 1 + (1 if include_originals else 0)
                if rows <= 0:
                    rows = 1

                fig, axes = plt.subplots(
                    rows, num_images, figsize=(4 * num_images, 4 * rows)
                )

                # Normalize axes to 2D array indexed by [row, col]
                if rows == 1 and num_images == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = np.expand_dims(axes, 0)
                elif num_images == 1:
                    axes = np.expand_dims(axes, 1)

                for col_idx, result in enumerate(processed_results):
                    # If this model should include originals, put them in the top row
                    label_row_start = 0
                    if include_originals:
                        axes[0, col_idx].imshow(result["img"])
                        axes[0, col_idx].set_title(
                            f"Seg-Image: {result['img_path'].name}"
                        )
                        axes[0, col_idx].axis("off")
                        label_row_start = 1

                    ax_row = label_row_start
                    seg_image_np = result["seg_image"]
                    im = axes[ax_row, col_idx].imshow(
                        seg_image_np, cmap="jet", vmin=0, vmax=1
                    )
                    axes[ax_row, col_idx].set_title(
                        f"n_labels: {len(result['seg_labels'].unique())}"
                    )
                    axes[ax_row, col_idx].axis("off")

                plt.subplots_adjust(top=0.95)

                avg_prediction_time = (
                    float(np.mean([r["prediction_time"] for r in processed_results]))
                    if processed_results
                    else 0.0
                )
                if len(seg_models_to_run) > 1 and include_originals:
                    fig.suptitle(
                        f"{seg_model}: prediction time {avg_prediction_time:.2f}",
                        fontsize=14,
                        y=0.5,
                    )
                else:
                    fig.suptitle(
                        f"{seg_model}: prediction time {avg_prediction_time:.2f}",
                        fontsize=14,
                        y=0.95,
                    )

                # Save per-model combined plot
                per_model_dir = result_dir / seg_model
                per_model_dir.mkdir(parents=True, exist_ok=True)
                combined_plot_path = (
                    per_model_dir / f"combined_segmentation_results_{lmm_idx}.png"
                )
                plt.savefig(combined_plot_path, dpi=300, bbox_inches="tight")
                print(
                    f"Saved combined visualization for {seg_model} to: {combined_plot_path}"
                )
                per_model_combined_paths.append(combined_plot_path)

        # After processing all segmentation models, stitch the per-model combined images into one file
        if args.evaluate_segmentation and per_model_combined_paths:
            stitched_images = []
            for p in per_model_combined_paths:
                try:
                    stitched_images.append(Image.open(p).convert("RGBA"))
                except Exception as e:
                    print(f"Failed to open image {p}: {e}")

            if stitched_images:
                # Resize images to same width before vertical concatenation
                widths = [im.width for im in stitched_images]
                target_width = max(widths)
                resized = []
                for im in stitched_images:
                    if im.width != target_width:
                        new_h = int(im.height * (target_width / im.width))
                        resized.append(im.resize((target_width, new_h), Image.LANCZOS))
                    else:
                        resized.append(im)

                total_height = sum(im.height for im in resized)
                stitched = Image.new(
                    "RGBA", (target_width, total_height), (255, 255, 255, 255)
                )
                y_offset = 0
                for im in resized:
                    stitched.paste(im, (0, y_offset), im)
                    y_offset += im.height

                final_combined_path = (
                    result_dir / f"combined_all_seg_models_{lmm_idx}.png"
                )
                stitched.convert("RGB").save(final_combined_path, dpi=(300, 300))
                print(
                    f"Saved stitched combined visualization to: {final_combined_path}"
                )

    print(f"Finished processing directory {dir_path}")

print("All directories processed.")
