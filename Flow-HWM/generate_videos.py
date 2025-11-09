import datetime
import os
from pathlib import Path

import hydra
import numpy as np
import safetensors
import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from loguru import logger as logging
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader

# Assuming these local modules are in your project structure
import samplers
from data import get_dataloaders
from models import get_model
from schedulers import get_scheduler


def setup_accelerator(config: DictConfig) -> Accelerator:
    """Initializes and returns the Accelerator."""
    accelerator = Accelerator(mixed_precision=config.get("mixed_precision", "no"))
    set_seed(config.get("seed", 42))

    logging.info(f"Accelerator initialized on device: {accelerator.device}")
    logging.info(
        f"Using {accelerator.num_processes} process(es) with {accelerator.mixed_precision} precision."
    )
    return accelerator


def load_vae(config: DictConfig) -> CausalVideoTokenizer:
    """
    Loads the CausalVideoTokenizer that is used for all VAE operations.
    This ensures consistency between encoding (in DataLoader) and decoding (in main loop).
    """
    logging.disable("cosmos_tokenizer")  # Suppress verbose VAE logging

    model_name_vid = (
        f"Cosmos-0.1-Tokenizer-CV{config.image_tokenizer.temporal_compression}x"
        f"{config.image_tokenizer.spatial_compression}x{config.image_tokenizer.spatial_compression}"
    )

    vae_path = Path(config.image_tokenizer.path) / model_name_vid

    vid_vae = CausalVideoTokenizer(
        checkpoint=vae_path / "autoencoder.jit",
        checkpoint_enc=vae_path / "encoder.jit",
        checkpoint_dec=vae_path / "decoder.jit",
        device=None,  # Accelerator will handle device placement
        dtype="bfloat16",
    )
    vid_vae.eval()
    for param in vid_vae.parameters():
        param.requires_grad = False

    logging.info(f"Video VAE loaded from: {vae_path}")
    return vid_vae


def load_checkpoint(
    accelerator: Accelerator, model: torch.nn.Module, config: DictConfig
):
    """Loads the model checkpoint based on the inference configuration."""
    checkpoint_path = Path(config.inference.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    if config.inference.use_ema:
        # Load EMA weights manually from a .safetensors file
        ema_path = checkpoint_path / "model.safetensors"
        logging.info(f"Loading EMA weights from: {ema_path}")
        if not ema_path.is_file():
            raise FileNotFoundError(f"EMA weights file not found: {ema_path}")

        state_dict = safetensors.torch.load_file(ema_path, device="cpu")

        # Clean state dict prefixes if necessary
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        unwrapped_model = accelerator.unwrap_model(model)
        missing, unexpected = unwrapped_model.load_state_dict(state_dict, strict=False)

        if missing:
            logging.warning(f"Missing keys when loading EMA weights: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading EMA weights: {unexpected}")
        logging.info("Successfully loaded EMA weights into the model.")

    else:
        # Load standard training state using Accelerator
        logging.info(
            f"Loading standard checkpoint state from directory: {checkpoint_path}"
        )
        if not checkpoint_path.is_dir():
            raise ValueError(
                f"Expected a directory for standard checkpoint, but got: {checkpoint_path}"
            )
        accelerator.load_state(checkpoint_path)
        logging.info("Successfully loaded checkpoint state via Accelerator.")


def save_frames(
    output_paths: dict,
    sample_videos: list,
    gt_frames: torch.Tensor,
    gt_autoencoded: torch.Tensor,
    batch_sample_ids: np.ndarray,
    num_past_frames: int,
    num_future_frames: int,
):
    """Saves predicted, ground truth, and autoencoded ground truth frames."""
    # Convert tensors to list of PIL Images
    gt_pil = [
        Image.fromarray(np.moveaxis(frame, 0, -1).astype(np.uint8))
        for frame in samplers.denormalize_video(gt_frames).reshape((-1, 3, 256, 256))
    ]
    gt_autoencoded_pil = [
        Image.fromarray(np.moveaxis(frame, 0, -1).astype(np.uint8))
        for frame in samplers.denormalize_video(gt_autoencoded).reshape(
            (-1, 3, 256, 256)
        )
    ]

    batch_size = len(sample_videos)
    for i in range(batch_size):
        video_id = batch_sample_ids[0][
            i
        ]  # Assuming video_id is consistent across frames

        # Create subdirectories for each video sequence
        pred_dir = output_paths["pred"] / str(video_id)
        gt_dir = output_paths["gt"] / str(video_id)
        auto_gt_dir = output_paths["autoencoded"] / str(video_id)
        pred_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        auto_gt_dir.mkdir(parents=True, exist_ok=True)

        for j in range(num_future_frames):
            frame_idx = batch_sample_ids[j][i].item()
            filename = f"{frame_idx:04d}.png"  # Use formatted frame index for sorting

            # Save predicted frame
            pred_img = sample_videos[i][j + num_past_frames]
            pred_img.save(pred_dir / filename)

            # Save original ground truth frame
            gt_pil[(i * num_future_frames) + j].save(gt_dir / filename)

            # Save autoencoded ground truth frame
            gt_autoencoded_pil[(i * num_future_frames) + j].save(auto_gt_dir / filename)


@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg: DictConfig):
    # 1. SETUP
    accelerator = setup_accelerator(cfg)
    logging.info("--- Starting Inference ---")
    logging.info(f"Configuration:\n{cfg.inference}")

    # 2. LOAD COMPONENTS
    # Load the VAE that will be used for ALL encoding and decoding
    vid_vae = load_vae(cfg)
    img_vae = vid_vae  # Ensure img_vae is an alias for vid_vae for consistency

    # Load other components
    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    # 3. LOAD DATA
    # The dataloader will use the provided VAEs (now identical) to pre-compute latents
    _, val_dataloader = get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vid_vae,
        val_stride=cfg.conditioning.get("num_future_frames") * 2,
    )
    logging.info(
        f"Validation dataloader loaded with {len(val_dataloader.dataset)} samples."
    )

    # 4. INSTANTIATE MODEL
    latent_h = cfg.image_size // cfg.image_tokenizer.spatial_compression
    latent_w = cfg.image_size // cfg.image_tokenizer.spatial_compression
    latent_channels = cfg.model.get("latent_channels", 16)

    model = get_model(cfg, latent_channels)
    logging.info(
        f"Model '{cfg.model.type}' instantiated with latent size: {latent_channels}x{latent_h}x{latent_w}"
    )

    # 5. PREPARE WITH ACCELERATOR
    # This moves all models to the correct device and handles precision
    model, vid_vae, img_vae, val_dataloader = accelerator.prepare(
        model, vid_vae, img_vae, val_dataloader
    )
    logging.info("All components prepared with Accelerator.")

    # 6. LOAD CHECKPOINT
    load_checkpoint(accelerator, model, cfg)
    model.eval()

    # 7. SETUP SAMPLER AND OUTPUT DIRECTORIES
    sampler = samplers.Sampler(
        vae=accelerator.unwrap_model(vid_vae),
        scheduler=noise_scheduler,
        seed=cfg.get("seed", 42),
        spatial_compression=cfg.image_tokenizer.spatial_compression,
        num_inference_steps=cfg.model.noise_steps,
        latent_channels=latent_channels,
    )

    if accelerator.is_main_process:
        output_dir = (
            Path(cfg.inference.output_dir) / Path(cfg.inference.checkpoint_path).name
        )
        output_paths = {
            "pred": output_dir / "predicted",
            "gt": output_dir / "ground_truth",
            "autoencoded": output_dir / "ground_truth_autoencoded",
        }
        for path in output_paths.values():
            path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output will be saved to: {output_dir}")

    # 8. INFERENCE LOOP
    progress_bar = tqdm.tqdm(
        total=len(val_dataloader),
        desc="Generating Samples",
        disable=not accelerator.is_main_process,
    )

    for step, batch in enumerate(val_dataloader):
        with torch.no_grad():
            # Generate predicted video frames from noise
            sample_videos, _ = sampler.sample_video(
                cfg,
                batch=batch,
                vae=accelerator.unwrap_model(vid_vae),  # Pass unwrapped VAE
                accelerator=accelerator,
                model=accelerator.unwrap_model(model),  # Pass unwrapped diffusion model
                guidance_scale=cfg.inference.guidance_scale,
                device=accelerator.device,
                n_samples=cfg.inference.batch_size,
                use_progress_bar=False,
            )

            # Generate autoencoded ground truth for debugging and comparison
            # This uses the same VAE that encoded the latents in the dataloader
            all_latents = torch.cat(
                (batch["past_latents"], batch["future_latents"]), dim=2
            )
            gt_autoencoded = accelerator.unwrap_model(vid_vae).decode(
                all_latents.to(torch.bfloat16)
            )

            # We only care about the future frames for saving
            gt_autoencoded_future = gt_autoencoded[
                :, :, cfg.conditioning.get("num_past_frames") :, ...
            ]

        # Gather results from all processes if doing multi-GPU inference
        # For single GPU, this is a no-op but good practice
        all_samples = accelerator.gather(sample_videos)
        all_gt_frames = accelerator.gather(batch["future_frames"])
        all_gt_autoencoded = accelerator.gather(gt_autoencoded_future)
        all_ids = accelerator.gather(batch["future_frames_idxs"])

        if accelerator.is_main_process:
            save_frames(
                output_paths=output_paths,
                sample_videos=all_samples,
                gt_frames=all_gt_frames,
                gt_autoencoded=all_gt_autoencoded,
                batch_sample_ids=[i.cpu().numpy() for i in all_ids],
                num_past_frames=cfg.conditioning.get("num_past_frames"),
                num_future_frames=cfg.conditioning.get("num_future_frames"),
            )
            progress_bar.update(1)

    if accelerator.is_main_process:
        progress_bar.close()
    logging.info("--- Inference Complete ---")


if __name__ == "__main__":
    main()
