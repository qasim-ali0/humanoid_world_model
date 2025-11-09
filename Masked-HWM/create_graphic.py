import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def load_images_with_stride(folder, stride=3):
    image_paths = sorted(os.listdir(folder))
    selected = image_paths[::stride]
    return [Image.open(os.path.join(folder, fname)).convert('RGB') for fname in selected]

def create_subfigure(gt_images, pred_images):
    assert len(gt_images) == len(pred_images)
    combined = []
    for gt, pred in zip(gt_images, pred_images):
        hstack = np.vstack([np.array(gt), np.array(pred)])  # GT on top, pred on bottom
        combined.append(hstack)
    strip = np.hstack(combined)  # Side-by-side sequence
    return strip

def generate_comparison(pred_root, gt_root, output_path, subfolder_ids=[0,500,1000,1500], stride=1):
    all_rows = []

    for folder_id in tqdm(subfolder_ids, desc="Processing subfolders"):
        pred_folder = os.path.join(pred_root, str(folder_id))
        gt_folder = os.path.join(gt_root, str(folder_id))

        pred_images = load_images_with_stride(pred_folder, stride)
        gt_images = load_images_with_stride(gt_folder, stride)

        min_len = min(len(pred_images), len(gt_images))
        pred_images = pred_images[:min_len]
        gt_images = gt_images[:min_len]

        row = create_subfigure(gt_images, pred_images)
        all_rows.append(row)

    full_image = np.vstack(all_rows)
    final_img = Image.fromarray(full_image)
    final_img.save(output_path)
    print(f"Saved comparison image to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True, help="Path to prediction folder")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth folder")
    parser.add_argument("--out", type=str, default="comparison.png", help="Output image filename")
    args = parser.parse_args()

    generate_comparison(args.pred, args.gt, args.out)