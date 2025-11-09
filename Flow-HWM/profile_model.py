import datetime
import math
import os
import shutil  # Not strictly needed for this version, but kept for consistency
import tempfile  # Not strictly needed for this version
import time  # For timing
from pathlib import Path

import hydra
import numpy as np
import torch
import tqdm  # Keep for progress bar if any, though mostly removed
from accelerate import Accelerator
from accelerate.utils import extract_model_from_parallel, set_seed

# Import clean-fid (Not needed for this script)
# from cleanfid import fid
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image  # Not strictly needed for this version

# Import safetensors loader
from safetensors.torch import load_file

# Import necessary components from your training script's modules
import data  # Assuming data.py contains get_dataloaders and encode_batch
import samplers  # Assuming samplers.py contains Sampler class
from models import get_model
from schedulers import get_scheduler


# --- Helper Functions for Metrics ---
def count_parameters(model: torch.nn.Module) -> int:
    """Counts the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def get_gpu_memory_usage(device: torch.device = None):
    """
    Returns current and peak GPU memory usage in GB.
    Reports: current allocated, current reserved, peak allocated, peak reserved.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0, 0.0

    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)
    return allocated, reserved, max_allocated, max_reserved


# --- Configuration ---
@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    # Use a small batch size for profiling, can be overridden by cfg if needed
    # For speed, it's good to use a realistic batch size.
    # For memory, a single sample might be enough if OOM is a concern, but batch_size helps find peak for batch processing.
    eval_batch_size = cfg.val.batch_size

    # --- Setup Accelerator ---
    # We will use Accelerator for device management on a single GPU
    # It will correctly handle device placement without DDP.
    accelerator = Accelerator(device_placement=True)  # device_placement=True is default
    device = accelerator.device
    logger.info(f"Using device: {device}")

    if not torch.cuda.is_available() and device.type == "cuda":
        logger.error("CUDA device requested but not available. Exiting.")
        return
    if device.type == "cuda":
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(device)}")

    # --- Initial GPU Memory ---
    if device.type == "cuda":
        torch.cuda.empty_cache()  # Clear cache before measurements
        torch.cuda.reset_peak_memory_stats(device)
        initial_allocated, initial_reserved, _, _ = get_gpu_memory_usage(device)
        logger.info(f"Initial GPU Memory (before model loading):")
        logger.info(f"  Allocated: {initial_allocated:.3f} GB")
        logger.info(f"  Reserved:  {initial_reserved:.3f} GB")

    # --- Load Config and Components (similar to training script) ---
    logger.info("Loading components...")
    logger.info(f"Model type: {cfg.model.type}")
    logger.info(f"Resume model from: {cfg.train.resume_model}")

    # VAE
    tokenizer_config = TokenizerConfigs[cfg.image_tokenizer.tokenizer_type].value
    tokenizer_config.update(
        dict(spatial_compression=cfg.image_tokenizer.spatial_compression)
    )
    logger.disable("cosmos_tokenizer")  # To reduce verbose VAE loading messages
    if "I" in cfg.image_tokenizer.tokenizer_type:
        model_name_vae = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = ImageTokenizer(
            checkpoint=Path(cfg.image_tokenizer.path)
            / model_name_vae
            / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path)
            / model_name_vae
            / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path)
            / model_name_vae
            / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,  # Will be moved by accelerator
            dtype=cfg.image_tokenizer.dtype,
        )
    else:
        model_name_vae = f"Cosmos-Tokenizer-{cfg.image_tokenizer.tokenizer_type}{cfg.image_tokenizer.temporal_compression}x{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
        vae = CausalVideoTokenizer(
            checkpoint=Path(cfg.image_tokenizer.path)
            / model_name_vae
            / "autoencoder.jit",
            checkpoint_enc=Path(cfg.image_tokenizer.path)
            / model_name_vae
            / "encoder.jit",
            checkpoint_dec=Path(cfg.image_tokenizer.path)
            / model_name_vae
            / "decoder.jit",
            tokenizer_config=tokenizer_config,
            device=None,  # Will be moved by accelerator
            dtype=cfg.image_tokenizer.dtype,  # specified as "bfloat16" in original script
        )
    logger.enable("cosmos_tokenizer")
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # Noise Scheduler
    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    # Main Model (Diffusion Transformer)
    latent_channels = tokenizer_config["latent_channels"]
    model_input_height_width = cfg.image_size // cfg.image_tokenizer.spatial_compression
    model = get_model(
        cfg,
        latent_channels,
    )

    # --- Load Trained Model Checkpoint ---
    checkpoint_folder = cfg.train.resume_model
    model_weights_path = os.path.join(checkpoint_folder, "model.safetensors")
    logger.info(f"Loading model weights from: {model_weights_path}")
    state_dict = load_file(model_weights_path, device="cpu")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning(
            f"Error loading state dict with strict=True: {e}. Attempting strict=False."
        )
        model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model weights loaded successfully.")

    # --- Prepare Components with Accelerator (moves to device) ---
    # This will move models to `accelerator.device`
    model, vae = accelerator.prepare(model, vae)

    model_unwrapped = extract_model_from_parallel(model)
    vae_unwrapped = extract_model_from_parallel(vae)
    model_dtype = next(model_unwrapped.parameters()).dtype
    logger.info(
        f"Model and VAE prepared and moved to {device} with dtype {model_dtype}."
    )

    # --- 1. Number of Parameters ---
    total_params_model = count_parameters(model_unwrapped)
    total_params_vae = count_parameters(
        vae_unwrapped
    )  # VAE params might also be of interest
    logger.info("-" * 40)
    logger.info("Parameter Counts:")
    logger.info(
        f"  Diffusion Model ({cfg.model.type}): {total_params_model / 1e6:.2f} M parameters"
    )
    logger.info(
        f"  VAE ({cfg.image_tokenizer.tokenizer_type}): {total_params_vae / 1e6:.2f} M parameters"
    )
    logger.info(
        f"  Total (Model + VAE): {(total_params_model + total_params_vae) / 1e6:.2f} M parameters"
    )

    # --- 2. Memory Usage (After Model Load) ---
    if device.type == "cuda":
        torch.cuda.synchronize(device)  # Ensure all ops are done
        mem_after_load_allocated, mem_after_load_reserved, _, _ = get_gpu_memory_usage(
            device
        )
        logger.info("-" * 40)
        logger.info("GPU Memory Usage (After Model Load):")
        logger.info(f"  Current Allocated: {mem_after_load_allocated:.3f} GB")
        logger.info(f"  Current Reserved:  {mem_after_load_reserved:.3f} GB")
        model_footprint_allocated = mem_after_load_allocated - initial_allocated
        model_footprint_reserved = (
            mem_after_load_reserved - initial_reserved
        )  # Less reliable due to caching strategy
        logger.info(
            f"  Approx. Model Footprint (Allocated): {model_footprint_allocated:.3f} GB (Model + VAE + utils)"
        )

    # --- Prepare for Inference Speed and Peak Memory Test ---
    sampler = samplers.Sampler(
        vae_unwrapped,
        noise_scheduler,
        cfg.seed,
        cfg.image_tokenizer.spatial_compression,
        latent_channels,
        cfg.model.noise_steps,
    )

    logger.info("Setting up data loader for inference test...")
    # We only need one batch for profiling
    _, val_dataloader = data.get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vae_unwrapped,  # Pass unwrapped VAE
        hmwm_train_dir=cfg.data.hmwm_train_dir,
        hmwm_val_dir=cfg.data.hmwm_val_dir,
        coco_train_imgs=cfg.data.get("coco_train_imgs"),
        coco_val_imgs=cfg.data.get("coco_val_imgs"),
        coco_train_ann=cfg.data.get("coco_train_ann"),
        coco_val_ann=cfg.data.get("coco_val_ann"),
        image_size=cfg.image_size,
        train_batch_size=eval_batch_size,  # Use eval_batch_size
        val_batch_size=eval_batch_size,  # Use eval_batch_size
        conditioning_type=cfg.conditioning.type,
        num_past_frames=cfg.conditioning.get("num_past_frames"),
        num_future_frames=cfg.conditioning.get("num_future_frames"),
        val_stride=1,  # Ensure we get varied samples if dataloader is iterated multiple times
        # Ensure `get_dataloaders` doesn't apply its own `accelerator.prepare` to the dataloader
        # if we want to prepare it explicitly later. Original script did `val_dataloader = accelerator.prepare(val_dataloader)`.
    )
    val_dataloader = accelerator.prepare(
        val_dataloader
    )  # Prepare dataloader with accelerator

    try:
        val_iter = iter(val_dataloader)
        batch = next(val_iter)
        logger.info(
            f"Successfully loaded a batch of data for inference of size {eval_batch_size}."
        )
    except StopIteration:
        logger.error("Validation dataloader is empty. Cannot perform inference test.")
        return
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Batch is now on device due to accelerator.prepare(val_dataloader) or internal dataloader logic
    # Or, ensure batch elements are on the correct device if sampler expects it.
    # The provided `sampler.sample_video` seems to handle device for some inputs,
    # but it's safer if `batch` tensors are already on `device`.
    # `accelerator.prepare(val_dataloader)` should handle this.

    num_gen_frames = cfg.conditioning.get(
        "num_future_frames", 8
    )  # Frames generated per sample
    actual_batch_size = batch["future_frames"].shape[
        0
    ]  # This is the true batch size from dataloader
    if actual_batch_size != eval_batch_size:
        logger.warning(
            f"Actual batch size {actual_batch_size} differs from configured eval_batch_size {eval_batch_size}. Using actual."
        )
        eval_batch_size = actual_batch_size

    # --- 3. Inference Speed and Peak Memory during Inference ---
    logger.info("-" * 40)
    logger.info(f"Performing Inference Speed & Peak Memory Test:")
    logger.info(f"  Batch size: {eval_batch_size}")
    logger.info(f"  Frames generated per sample: {num_gen_frames}")
    logger.info(f"  Guidance scale (CFG): {cfg.model.cfg_scale}")
    logger.info(
        f"  Diffusion steps: {cfg.model.noise_steps}"
    )  # Assuming this is num_inference_steps for sampler

    num_warmup_runs = cfg.get("val", {}).get("num_warmup_runs", 3)
    num_timed_runs = cfg.get("val", {}).get("num_timed_runs", 10)

    logger.info(f"Starting {num_warmup_runs} warm-up runs...")
    for _ in range(num_warmup_runs):
        with torch.no_grad():
            _ = sampler.sample_video(
                cfg=cfg,
                batch=batch,  # Already on device
                vae=vae_unwrapped,
                accelerator=accelerator,  # Pass accelerator if sampler uses it
                model=model_unwrapped,
                guidance_scale=cfg.model.cfg_scale,
                dtype=model_dtype,
                n_samples=1,
                device=device,  # Sampler might use this to create new tensors
                use_progress_bar=False,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    logger.info("Warm-up complete.")

    # Reset peak memory stats before timed runs for accurate peak measurement during inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings_ms = []
    logger.info(f"Starting {num_timed_runs} timed inference runs...")
    for i in range(num_timed_runs):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:  # CPU timing
            start_time_cpu = time.perf_counter()

        with torch.no_grad():
            _, _ = sampler.sample_video(
                cfg=cfg,
                batch=batch,
                vae=vae_unwrapped,
                accelerator=accelerator,
                model=model_unwrapped,
                guidance_scale=cfg.model.cfg_scale,
                dtype=model_dtype,
                n_samples=eval_batch_size,
                device=device,
                use_progress_bar=False,
            )

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(device)  # Crucial for accurate GPU timing
            current_time_ms = start_event.elapsed_time(end_event)
        else:  # CPU timing
            current_time_ms = (time.perf_counter() - start_time_cpu) * 1000.0

        timings_ms.append(current_time_ms)
        logger.info(f"  Run {i+1}/{num_timed_runs}: {current_time_ms:.2f} ms")

    # --- Calculate and Log Speed Metrics ---
    if timings_ms:
        avg_time_ms = sum(timings_ms) / len(timings_ms)
        avg_time_s_per_batch = avg_time_ms / 1000.0

        # Samples are "videos" in this context
        samples_per_second = eval_batch_size / avg_time_s_per_batch
        # Frames are individual video frames
        frames_per_second = (eval_batch_size * num_gen_frames) / avg_time_s_per_batch

        logger.info("Inference Speed Results:")
        logger.info(
            f"  Average time per batch ({eval_batch_size} samples): {avg_time_s_per_batch:.4f} seconds"
        )
        logger.info(
            f"  Throughput (Samples/sec): {samples_per_second:.2f} samples/second"
        )
        logger.info(
            f"  Throughput (Frames/sec): {frames_per_second:.2f} frames/second (at {num_gen_frames} frames/sample)"
        )
    else:
        logger.warning("No timings recorded. Speed metrics cannot be calculated.")

    # --- Peak Memory during Inference (already captured by max_memory_allocated) ---
    if device.type == "cuda":
        # The peak was tracked since the last reset_peak_memory_stats
        # get_gpu_memory_usage() will report current and peak values.
        # The 'max_allocated' from this call will be the peak *during the timed runs*.
        _, _, peak_inf_allocated, peak_inf_reserved = get_gpu_memory_usage(device)
        logger.info("Peak GPU Memory Usage (During Timed Inference Runs):")
        logger.info(f"  Peak Allocated: {peak_inf_allocated:.3f} GB")
        logger.info(f"  Peak Reserved:  {peak_inf_reserved:.3f} GB (overall pool peak)")

        # Calculate activation memory (approximate)
        # Peak allocated during inference minus memory allocated after model load (but before inference variables)
        # This is a bit tricky because mem_after_load_allocated includes model weights.
        # A better estimate for activation memory is peak_inf_allocated - mem_after_load_allocated
        # if mem_after_load_allocated was taken right before the inference loop and after any prep.
        # For simplicity, peak_inf_allocated is the total peak for model + activations + intermediates.
        activation_approx_memory = (
            peak_inf_allocated - model_footprint_allocated
        )  # This is a rough estimate
        logger.info(
            f"  Approx. Peak Activation/Intermediate Tensors Memory (Peak Allocated - Model Footprint): {max(0, activation_approx_memory):.3f} GB"
        )

    # --- Final Summary ---
    logger.info("-" * 40)
    logger.info("Summary of Metrics:")
    logger.info(f"  Model: {cfg.model.type} from {checkpoint_folder}")
    logger.info(f"  Diffusion Model Params: {total_params_model / 1e6:.2f} M")
    if device.type == "cuda":
        logger.info(
            f"  GPU Memory - Model Footprint (Allocated): {model_footprint_allocated:.3f} GB"
        )
        logger.info(
            f"  GPU Memory - Peak During Inference (Allocated): {peak_inf_allocated:.3f} GB"
        )
    if timings_ms:
        logger.info(f"  Inference Speed - Samples/sec: {samples_per_second:.2f}")
        logger.info(f"  Inference Speed - Frames/sec: {frames_per_second:.2f}")
    logger.info("-" * 40)
    logger.info("Script finished.")

    # Clean up (optional, if temporary files were created, not in this script)
    # Example: shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
