import datetime
import math
import os
import shutil
import tempfile
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import extract_model_from_parallel, set_seed

# Import clean-fid
from cleanfid import fid
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image

# Import safetensors loader
from safetensors.torch import load_file

# Import necessary components from your training script's modules
import data  # Assuming data.py contains get_dataloaders and encode_batch
import samplers  # Assuming samplers.py contains Sampler class
from conditioning import ConditioningManager
from models import get_model
from schedulers import get_scheduler

# No longer need EMAModel


# --- PSNR Helper Function ---
def calculate_psnr_torch(img1, img2, data_range=1.0):
    """Calculates PSNR between two torch tensors (expects values in [0, data_range])."""
    if not isinstance(img1, torch.Tensor):
        img1 = torch.tensor(img1)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.tensor(img2)

    img1 = img1.float()
    img2 = img2.float()
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        # PSNR is infinite if images are identical
        return float("inf")
        # return torch.tensor(float('inf'), device=mse.device) # Or return tensor infinity
    psnr = 20 * torch.log10(data_range / torch.sqrt(mse))
    return psnr.item()  # Return as float


# --- Configuration ---
@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    gen_batch_size = cfg.val.batch_size  # Batch size for generating videos
    fid_batch_size = (
        32  # Batch size for FID feature extraction (adjust based on GPU memory)
    )

    # --- Setup Accelerator ---
    accelerator = Accelerator(device_placement=True)
    device = accelerator.device
    logger.info(f"Using device: {device}")

    # --- Load Config and Components (similar to training script) ---
    logger.info("Loading components...")
    logger.info(f"model.type: {cfg.model.type}")
    logger.info(f"resume_model: {cfg.train.resume_model}")

    # VAE (Assuming VAE loading remains the same)
    tokenizer_config = TokenizerConfigs[cfg.image_tokenizer.tokenizer_type].value
    tokenizer_config.update(
        dict(spatial_compression=cfg.image_tokenizer.spatial_compression)
    )
    logger.disable("cosmos_tokenizer")
    if "I" in cfg.image_tokenizer.tokenizer_type:
        model_name = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = ImageTokenizer(  # Use appropriate class
            checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,
            dtype=cfg.image_tokenizer.dtype,
        )
    else:
        model_name = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.temporal_compression}x{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = CausalVideoTokenizer(  # Use appropriate class
            checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,
            dtype="bfloat16",
        )  # Or your preferred eval dtype
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Conditioning Manager
    conditioning_manager = ConditioningManager(cfg.conditioning)

    # Noise Scheduler
    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    # Main Model (UNet)
    latent_channels = tokenizer_config["latent_channels"]
    model = get_model(cfg, latent_channels)

    # --- Load Trained Model Checkpoint ---
    # Determine the checkpoint path: Use eval.checkpoint_dir if provided, otherwise default to train.resume_model
    checkpoint_folder = cfg.train.resume_model
    model_weights_path = os.path.join(checkpoint_folder, "model.safetensors")
    logger.info(f"Loading model weights from: {model_weights_path}")
    state_dict = load_file(model_weights_path, device="cpu")  # Load to CPU first

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.error(
            f"Error loading state dict, possibly due to missing/unexpected keys or prefixes: {e}"
        )
        logger.info("Attempting to load with strict=False (may ignore some keys)")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    logger.info("Model weights loaded successfully.")

    # --- Prepare Components with Accelerator ---
    model, vae, conditioning_manager = accelerator.prepare(
        model, vae, conditioning_manager
    )

    model_unwrapped = extract_model_from_parallel(model)
    vae_unwrapped = extract_model_from_parallel(vae)

    # Sampler
    sampler = samplers.Sampler(
        vae_unwrapped,
        noise_scheduler,
        cfg.seed,
        cfg.image_tokenizer.spatial_compression,
        latent_channels,
        cfg.model.noise_steps,
    )

    # --- Prepare Data ---
    logger.info("Setting up data loader for evaluation...")
    _, val_dataloader = data.get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vae_unwrapped,
        val_stride=1,
    )

    val_dataloader = accelerator.prepare(val_dataloader)

    # --- Create Temporary Directories for Real and Generated Frames (for FID) ---
    real_dir = tempfile.mkdtemp()
    fake_dir = tempfile.mkdtemp()
    logger.info(f"Temporary directory for real frames (FID): {real_dir}")
    logger.info(f"Temporary directory for fake frames (FID): {fake_dir}")

    # --- Process Data and Generate Videos ---
    saved_real_count = 0
    saved_fake_count = 0
    all_psnr_values = []  # <<< Initialize list to store PSNR values for averaging

    num_cond_frames = cfg.conditioning.get("num_past_frames", 1)
    num_gen_frames = cfg.conditioning.get("num_future_frames", 8)
    logger.info(
        f"Expecting {num_cond_frames} conditioning frames and {num_gen_frames} generated/ground truth frames."
    )

    # Determine number of samples (use length of dataloader)
    num_samples = int(5000 // 3)
    _ = (
        len(val_dataloader.dataset)
        if hasattr(val_dataloader, "dataset")
        else len(val_dataloader) * gen_batch_size
    )
    logger.info(
        f"Starting generation and evaluation for approximately {num_samples} samples..."
    )
    pbar = tqdm.tqdm(total=num_samples, disable=not accelerator.is_main_process)

    try:
        val_iter = iter(val_dataloader)
        processed_samples = 0
        while processed_samples < num_samples:
            try:
                batch = next(val_iter)
            except StopIteration:
                logger.warning("Validation dataloader exhausted.")
                break

            current_batch_size = batch["future_frames"].shape[
                0
            ]  # Use a key guaranteed to be present
            # Handle last potentially smaller batch
            # samples_to_process_in_batch = min(
            #     current_batch_size, num_samples - processed_samples
            # )
            # if samples_to_process_in_batch < current_batch_size:
            #     for key in batch:
            #         if (
            #             isinstance(batch[key], torch.Tensor)
            #             and batch[key].shape[0] == current_batch_size
            #         ):
            #             batch[key] = batch[key][:samples_to_process_in_batch]
            #         elif (
            #             isinstance(batch[key], list)
            #             and len(batch[key]) == current_batch_size
            #         ):  # Handle lists if present in batch
            #             batch[key] = batch[key][:samples_to_process_in_batch]
            #     current_batch_size = samples_to_process_in_batch

            # if current_batch_size == 0:
            #     continue  # Skip if batch becomes empty

            # --- Prepare Real Frames (Ground Truth Continuation) ---
            # B, T, C, H, W, values expected in [-1, 1] from dataloader
            real_videos_batch = batch["future_frames"]
            # Select the first future frame for FID and PSNR (B, C, H, W)
            gt_first_frame_batch = real_videos_batch[:, :, 0, ...]
            # Normalize to [0, 1] for PSNR calculation and saving
            gt_first_frame_batch_01 = (gt_first_frame_batch * 0.5 + 0.5).clamp(0, 1)

            # --- Save Real Frames for FID ---
            # Convert to uint8 PIL for saving
            gt_first_frame_batch_uint8 = (
                (gt_first_frame_batch_01 * 255).byte().cpu().numpy()
            )
            for i in range(current_batch_size):
                # Limit saving if needed (though FID usually needs all)
                # if saved_real_count < num_fid_samples: # Check if needed based on FID requirements
                video_idx = (
                    processed_samples + i + 1
                )  # Use overall processed count for unique naming
                frame_img = Image.fromarray(
                    gt_first_frame_batch_uint8[i].transpose(1, 2, 0)
                )
                frame_img.save(
                    os.path.join(real_dir, f"vid{video_idx:05d}_frame{0:03d}.png")
                )
                saved_real_count += 1

            # --- Generate Fake Videos ---
            with torch.no_grad():
                model_dtype = next(model_unwrapped.parameters()).dtype
                sample_videos_pil, _ = sampler.sample_video(
                    cfg=cfg,
                    batch=batch,
                    vae=vae_unwrapped,
                    accelerator=accelerator,
                    model=model_unwrapped,
                    guidance_scale=cfg.model.cfg_scale,
                    dtype=model_dtype,
                    n_samples=current_batch_size,
                    device=device,
                    use_progress_bar=False,
                )
                # sample_videos_pil is list[list[PIL.Image]] [batch_size, num_gen_frames]

            # --- Process Fake Frames for FID and PSNR ---
            batch_psnr_values = []  # Store PSNR for the current batch (on this process)
            for i in range(current_batch_size):
                # Get the first generated frame
                fake_first_frame_pil = sample_videos_pil[i][0]

                # --- Save Fake Frame for FID ---
                # Limit saving if needed
                # if saved_fake_count < num_fid_samples:
                video_idx = processed_samples + i + 1
                fake_first_frame_pil.save(
                    os.path.join(fake_dir, f"vid{video_idx:05d}_frame{0:03d}.png")
                )
                saved_fake_count += 1

                # --- Calculate PSNR ---
                # Convert fake PIL [0, 255] to tensor [0, 1]
                fake_first_frame_np = np.array(fake_first_frame_pil) / 255.0  # H, W, C
                # Ensure channel first (C, H, W) and correct device/dtype
                fake_first_frame_tensor = torch.tensor(
                    fake_first_frame_np,
                    dtype=gt_first_frame_batch_01.dtype,
                    device=device,
                ).permute(2, 0, 1)

                # Get corresponding ground truth frame tensor [0, 1]
                real_first_frame_tensor = gt_first_frame_batch_01[i]

                # Calculate PSNR for this pair
                psnr_val = calculate_psnr_torch(
                    real_first_frame_tensor, fake_first_frame_tensor, data_range=1.0
                )
                batch_psnr_values.append(psnr_val)

            # Extend the list holding all PSNR values for this process
            all_psnr_values.extend(batch_psnr_values)

            processed_samples += current_batch_size
            if accelerator.is_main_process:
                pbar.update(current_batch_size)

            if cfg.debug:
                break

    except Exception as e:
        logger.error(f"Error during generation/evaluation: {e}")
        # Optionally re-raise e
    finally:
        if accelerator.is_main_process:
            pbar.close()

    # Ensure all processes finish generation and local PSNR calculation
    accelerator.wait_for_everyone()

    # --- Aggregate PSNR Values Across Processes ---
    if accelerator.num_processes > 1:
        # Convert local PSNR list to tensor for gathering
        local_psnr_tensor = torch.tensor(all_psnr_values, device=device)
        # Gather tensors from all processes into a list of tensors on the main process
        gathered_psnr_tensors = accelerator.gather(
            local_psnr_tensor
        )  # Shape will be roughly (num_processes * samples_per_process)
        # Concatenate gathered tensors and convert to list on main process
        if accelerator.is_main_process:
            # gathered_psnr_tensors now contains all PSNR values from all processes
            # Make sure to only take the number of values corresponding to processed_samples
            # Note: processed_samples might slightly differ across processes if batch sizes vary.
            # Using the length of the gathered tensor is usually correct if dataloader handled distribution evenly.
            final_psnr_values = gathered_psnr_tensors.cpu().numpy().tolist()
            # Trim potentially padded values if accelerator added any (less common with gather)
            # Or rely on saved_fake_count as the total number of samples evaluated across all processes
            # We need a reliable global count. Let's use saved_fake_count assuming it's synchronized implicitly
            # (though safer would be to gather counts too).
            # Use len(final_psnr_values) as the most direct measure from gathered data.
            actual_evaluated_count = len(final_psnr_values)

    else:  # Single process
        final_psnr_values = all_psnr_values
        actual_evaluated_count = len(final_psnr_values)

    # --- Calculate Average PSNR (on main process) ---
    average_psnr = float("nan")
    num_psnr_samples = 0
    if accelerator.is_main_process:
        if final_psnr_values:
            # Handle potential infinities if images were identical
            finite_psnr_values = [p for p in final_psnr_values if math.isfinite(p)]
            if finite_psnr_values:
                average_psnr = np.mean(finite_psnr_values)
                num_psnr_samples = len(
                    final_psnr_values
                )  # Report count including infinities
                logger.info(
                    f"Average PSNR ({num_psnr_samples} samples): {average_psnr:.4f}"
                )
                if len(finite_psnr_values) < num_psnr_samples:
                    logger.warning(
                        f"{num_psnr_samples - len(finite_psnr_values)} samples had infinite PSNR (identical images)."
                    )
            else:
                # All values were infinite
                average_psnr = float("inf")
                num_psnr_samples = len(final_psnr_values)
                logger.info(
                    f"Average PSNR ({num_psnr_samples} samples): {average_psnr}"
                )

        else:
            logger.warning("No PSNR values were calculated.")
            average_psnr = float("nan")
            num_psnr_samples = 0

    # --- Calculate FID (on main process) ---
    fid_score = float("nan")
    final_num_fid_samples = 0
    results_log_path = None  # Define scope outside if block

    if accelerator.is_main_process:
        logger.info("Calculating FID score...")
        # Use actual saved counts for robustness
        real_files = [f for f in os.listdir(real_dir) if f.endswith(".png")]
        fake_files = [f for f in os.listdir(fake_dir) if f.endswith(".png")]
        num_real_videos_saved = len(real_files)  # Saved one frame per video
        num_fake_videos_saved = len(fake_files)

        if num_real_videos_saved != num_fake_videos_saved:
            logger.warning(
                f"Number of actual saved real ({num_real_videos_saved}) and fake ({num_fake_videos_saved}) frames differ. FID might be inaccurate."
            )
            final_num_fid_samples = min(num_real_videos_saved, num_fake_videos_saved)
            logger.warning(f"Calculating FID based on {final_num_fid_samples} pairs.")
        else:
            # Use the actual saved count, which should match num_eval_samples ideally
            final_num_fid_samples = num_fake_videos_saved
            if final_num_fid_samples != actual_evaluated_count:
                logger.warning(
                    f"FID sample count ({final_num_fid_samples}) differs from PSNR sample count ({actual_evaluated_count})."
                )

        if final_num_fid_samples == 0:
            logger.error(
                "No samples were saved correctly for FID. Cannot calculate FID."
            )
            fid_score = float("nan")
        else:
            try:
                # Limit FID calculation samples if needed (e.g., FID5k instead of full set)
                # num_fid_to_calc = min(final_num_fid_samples, cfg.val.get("num_fid_samples", final_num_fid_samples))
                num_fid_to_calc = (
                    final_num_fid_samples  # Use all generated samples by default
                )

                fid_score = fid.compute_fid(
                    real_dir,
                    fake_dir,
                    mode="legacy_pytorch",  # Or "clean"
                    num_workers=4,  # Adjust as needed
                    batch_size=fid_batch_size,
                    device=device,
                    num_gen=num_fid_to_calc,  # Use the number of samples we want to compute FID over
                    verbose=False,  # Reduce verbosity
                )
                logger.info(f"FID score ({num_fid_to_calc} samples): {fid_score:.4f}")
            except Exception as e:
                logger.error(f"FID calculation failed: {e}")
                fid_score = float("nan")
            finally:
                # --- Clean Up Temporary Directories ---
                logger.info("Cleaning up temporary FID directories...")
                try:
                    shutil.rmtree(real_dir)
                    shutil.rmtree(fake_dir)
                    logger.info("FID cleanup complete.")
                except Exception as e:
                    logger.error(f"Error during FID cleanup: {e}")

        # Log or save the results
        output_dir = Path(
            cfg.val.get("output_dir", Path(checkpoint_folder) / "evaluation_results")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        results_log_path = (
            output_dir
            / f"eval_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

        # <<< Log both FID and PSNR to file >>>
        with open(results_log_path, "w") as f:
            f.write(f"Evaluation Results:\n")
            f.write(f"Checkpoint Folder: {checkpoint_folder}\n")
            f.write(f"Model Weights: {model_weights_path}\n")
            f.write("-" * 20 + "\n")
            f.write(
                f"FID Score ({final_num_fid_samples} samples evaluated): {fid_score:.4f}\n"
            )
            f.write(
                f"Average PSNR ({num_psnr_samples} samples evaluated): {average_psnr:.4f}\n"
            )
            f.write("-" * 20 + "\n")
            # Optional: Add config details
            # f.write(f"Config used:\n{OmegaConf.to_yaml(cfg)}\n")
        logger.info(f"Evaluation results saved to: {results_log_path}")

    else:
        # Ensure non-main processes wait if needed (already done with wait_for_everyone)
        pass

    # --- Final Output ---
    logger.info("Evaluation script finished.")
    if accelerator.is_main_process:
        logger.info(f"Final Results:")
        logger.info(f"  FID Score ({final_num_fid_samples} samples): {fid_score:.4f}")
        logger.info(f"  Average PSNR ({num_psnr_samples} samples): {average_psnr:.4f}")
        if results_log_path:
            logger.info(f"  Results saved to: {results_log_path}")


if __name__ == "__main__":
    main()
