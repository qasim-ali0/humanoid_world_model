import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from diffusers.utils import make_image_grid

import wandb
from data import encode_batch


def compute_val_loss(
    cfg,
    model,
    val_dataloader,
    noise_scheduler,
    accelerator,
    progress_bar_enabled=True,
    img_vae=None,
    vid_vae=None,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    n_timesteps = cfg.model.noise_steps

    # Create progress bar only on the main process and if enabled
    if accelerator.is_main_process and progress_bar_enabled:
        pbar = tqdm.tqdm(total=len(val_dataloader), desc="Validating")
    else:
        pbar = None

    for step, batch in enumerate(val_dataloader):
        latents, batch = encode_batch(
            cfg, batch, accelerator, vae=vid_vae, img_vae=img_vae, vid_vae=vid_vae
        )
        bs = latents.shape[0]
        noise = torch.randn(latents.shape, device=accelerator.device)
        if cfg.use_discrete_time:
            timesteps = torch.randint(
                0,
                cfg.model.noise_steps,
                (bs, 1),
                device=latents.device,
                dtype=torch.int64,
            )
        else:
            u = torch.randn((bs, 1), device=accelerator.device)
            timesteps = 1.0 / (1.0 + torch.exp(-u))
            timesteps = timesteps
        # Add noise to latents
        batch["noisy_latents"] = noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():
            # Predict the noise residual
            noise_pred = model(batch, timesteps, accelerator.device, use_cfg=True)
            noise_target = noise_scheduler.get_target(latents, noise, timesteps)
            # Compute mean squared error loss (averaged over the batch)
            loss = F.mse_loss(noise_pred, noise_target, reduction="mean")

        total_loss += loss.item() * bs
        total_samples += bs

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(loss=loss.item())
    # if pbar is not None:
    #     pbar.close()

    # Gather metrics from all processes
    total_loss_tensor, total_samples_tensor = accelerator.gather_for_metrics(
        (
            torch.tensor(total_loss, device=accelerator.device),
            torch.tensor(total_samples, device=accelerator.device),
        )
    )
    # Compute average loss across all processes
    avg_loss = total_loss_tensor.sum() / total_samples_tensor.sum()
    return avg_loss.item()


def get_wandb_config(cfg):
    return {
        "learning_rate": cfg.train.learning_rate,
        "num_epochs": cfg.train.num_epochs,
        "train_batch_size": cfg.train.batch_size,
        "eval_batch_size": cfg.val.batch_size,
        "noise_steps": cfg.model.noise_steps,
        "image_size": cfg.image_size,
        "mixed_precision": cfg.mixed_precision,
        "token_dim": cfg.model.token_dim,
        "num_heads": cfg.model.num_heads,
        "num_layers": cfg.model.num_layers,
        "gradient_accumulation_steps": cfg.train.gradient_accumulation_steps,
        "model_type": cfg.model.type,
        "resume_train": cfg.train.resume_train,
        "resume_model": cfg.train.resume_model,
    }


def override_for_one_sample(cfg):
    """Adjust config for single-sample debugging."""
    if not cfg.one_sample:
        return cfg

    cfg.train.learning_rate = 1e-4
    cfg.train.lr_warmup_steps = 0
    cfg.train.gradient_accumulation_steps = 1
    cfg.train.batch_size = 4
    cfg.val.batch_size = 1
    cfg.exp_prefix = "one-sample"
    cfg.train.save_model_iters = 50000
    cfg.train.val_iters = 50
    cfg.conditioning.prompt_file = "prompts_one_sample.txt"
    cfg.val.run = True
    cfg.val.skip_val_loss = False
    cfg.val.skip_img_sample = False
    return cfg


def load_video_tokenizer(cfg):
    model_name = (
        f"Cosmos-0.1-Tokenizer-CV{cfg.image_tokenizer.temporal_compression}"
        f"x{cfg.image_tokenizer.spatial_compression}"
        f"x{cfg.image_tokenizer.spatial_compression}"
    )
    path = Path(cfg.image_tokenizer.path) / model_name
    vid_vae = CausalVideoTokenizer(
        checkpoint=path / "autoencoder.jit",
        checkpoint_enc=path / "encoder.jit",
        checkpoint_dec=path / "decoder.jit",
        dtype="bfloat16",
    )
    for p in vid_vae.parameters():
        p.requires_grad = False
    return vid_vae


def log_img_and_vids_wandb(
    cfg,
    accelerator,
    eval_dir,
    path,
    epoch,
    step,
    sample_videos=None,
    sample_imgs=None,
):

    if cfg.gen_type.lower() in ["image", "frame"]:
        rows = max(len(sample_imgs) // 4, 1)
        cols = len(sample_imgs) // max(rows, 1)
        image_grid = make_image_grid(sample_imgs, rows=rows, cols=cols)
        os.makedirs(eval_dir, exist_ok=True)
        image_grid.save(path)
        if accelerator.is_main_process and not cfg.debug:
            wandb.log(
                {
                    "generated_images": wandb.Image(
                        str(path),
                        caption=f"Epoch {epoch}, iter {step}",
                    )
                }
            )
    else:
        os.makedirs(eval_dir, exist_ok=True)
        b = len(sample_videos)
        h, w = sample_videos[0][0].size
        for i in range(b):
            path_vid = os.path.join(eval_dir, f"output-{i}.mp4")
            out = cv2.VideoWriter(
                path_vid,
                cv2.VideoWriter_fourcc(*"mp4v"),
                0.2,
                (w, h),
            )
            for frame in sample_videos[i]:
                out.write(np.asarray(frame))
            out.release()
            path_img0 = os.path.join(eval_dir, f"tokenizedprompt+predictions.jpeg")
            sample_imgs[0][i].save(path_img0)

            path_img1 = os.path.join(eval_dir, f"gtprompt+predictions.jpeg")
            sample_imgs[1][i].save(path_img1)

            path_img2 = os.path.join(eval_dir, f"gt_vs_predictions.jpeg")
            sample_imgs[2][i].save(path_img2)

            if accelerator.is_main_process and not cfg.debug:
                wandb.log(
                    {
                        f"samples/gt_vs_predictions-{i}": wandb.Image(
                            str(path_img2),
                            caption=f"Epoch {epoch}, iter {step}",
                        )
                    }
                )
                wandb.log(
                    {
                        f"samples/tokenizedprompt+predictions-{i}": wandb.Image(
                            str(path_img0),
                            caption=f"Epoch {epoch}, iter {step}",
                        )
                    }
                )
                wandb.log(
                    {
                        f"samples/gtprompt+predictions-{i}": wandb.Image(
                            str(path_img1),
                            caption=f"Epoch {epoch}, iter {step}",
                        )
                    }
                )
                # wandb.log({"samples/video": wandb.Video(path_vid, fps=0.2, format="mp4")})
