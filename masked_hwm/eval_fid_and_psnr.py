import os
import glob
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
from cleanfid import fid
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import tempfile
import shutil

def collect_image_pairs(pred_dir, gt_dir, max_images=None):
    pred_paths, gt_paths = [], []

    pred_folders = sorted(os.listdir(pred_dir), key=lambda x: int(x))
    gt_folders = sorted(os.listdir(gt_dir), key=lambda x: int(x))

    # count = 0
    for pf, gf in zip(pred_folders, gt_folders):
        pred_images = sorted(glob.glob(os.path.join(pred_dir, pf, '*')))
        gt_images = sorted(glob.glob(os.path.join(gt_dir, gf, '*')))

        for pi, gi in zip(pred_images, gt_images):
            # if count >= max_images:
            #     break
            pred_paths.append(pi)
            gt_paths.append(gi)
            # count += 1
        # if count >= max_images:
        #     break

    return pred_paths, gt_paths

def save_images_to_temp_folder(image_paths, temp_dir):
    for i, path in enumerate(tqdm(image_paths, desc=f"Saving images to {temp_dir}")):
        img = Image.open(path).convert('RGB')
        img.save(os.path.join(temp_dir, f"{i:05d}.png"))

def calculate_fid(pred_folder, gt_folder):
    return fid.compute_fid(pred_folder, gt_folder, mode="clean")

def calculate_average_psnr(pred_paths, gt_paths):
    psnrs = []
    for pred_path, gt_path in tqdm(zip(pred_paths, gt_paths), total=len(pred_paths), desc="Computing PSNR"):
        pred = np.array(Image.open(pred_path).convert('RGB'))
        gt = np.array(Image.open(gt_path).convert('RGB'))

        if pred.shape != gt.shape:
            continue  # skip mismatched shapes

        psnr = compute_psnr(gt, pred, data_range=255)
        psnrs.append(psnr)

    return np.mean(psnrs) if psnrs else 0.0

def main(results):
    pred_root = f"{results}/pred_future"
    gt_root = f"{results}/gt_future"

    print("Collecting matching images...")
    pred_paths, gt_paths = collect_image_pairs(pred_root, gt_root)

    # Define the paths
    pred_temp = f"pred_temp_{results.split('/')[-1]}"
    gt_temp = f"gt_temp_{results.split('/')[-1]}"

    # Make sure the directories exist (create or clear them)
    os.makedirs(pred_temp, exist_ok=True)
    os.makedirs(gt_temp, exist_ok=True)

    for folder in [pred_temp, gt_temp]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    
    print("Saving temp images for FID...")
    save_images_to_temp_folder(pred_paths, pred_temp)
    save_images_to_temp_folder(gt_paths, gt_temp)

    print("Computing FID...")
    fid_score = calculate_fid(pred_temp, gt_temp)

    # Clean up afterwards
    shutil.rmtree(pred_temp)
    shutil.rmtree(gt_temp)

    print("Computing PSNR...")
    psnr_score = calculate_average_psnr(pred_paths, gt_paths)

    print(f"\nEvaluation Results for {args.results.split('/')[-1]}:")
    print(f"FID  = {fid_score:.2f}")
    print(f"PSNR = {psnr_score:.2f} dB")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True, help="Path to prediction root folder")
    args = parser.parse_args()

    main(args.results)