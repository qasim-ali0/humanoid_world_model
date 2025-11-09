import datetime
import os
from collections import OrderedDict
from itertools import islice
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import RNGType, set_seed
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from loguru import logger as logging

import samplers
import wandb
from conditioning import ConditioningManager
from data import encode_batch, get_dataloaders
from models import get_model
from schedulers import get_scheduler
from train_utils import (
    compute_val_loss,
    get_wandb_config,
    load_video_tokenizer,
    log_img_and_vids_wandb,
    override_for_one_sample,
)


def setup_distributed_training(config):
    """Setup for distributed training"""
    accelerator = Accelerator(
        rng_types=[RNGType.TORCH],
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        project_dir=config.log_dir,
        # Add these parameters for multi-GPU
        split_batches=False,  # Split batches across devices
        device_placement=True,
        kwargs_handlers=[
            InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=18000))
        ],
    )
    # Set seed for reproducibility
    set_seed(config.seed)
    return accelerator


@hydra.main(config_path="configs", config_name="flow_video_mmdit", version_base=None)
def main(cfg):
    torch.manual_seed(0)
    np.random.seed(0)
    gen = torch.Generator()
    gen.manual_seed(cfg.seed)

    logging.info(f"debug: {cfg.debug}")
    logging.info(f"learning_rate: {cfg.train.learning_rate}")
    logging.info(f"lr_warmup_steps: {cfg.train.lr_warmup_steps}")
    logging.info(f"one_sample: {cfg.one_sample}")
    logging.info(f"image_size: {cfg.image_size}")
    logging.info(f"log_dir: {cfg.log_dir}")
    logging.info(f"train_batch_size: {cfg.train.batch_size}")
    logging.info(
        f"gradient_accumulation_steps: {cfg.train.gradient_accumulation_steps}"
    )
    logging.info(f"val_iters: {cfg.train.val_iters}")
    logging.info(f"exp_prefix: {cfg.exp_prefix}")
    logging.info(f"data: {cfg.data.type}")
    logging.info(f"conditioning: {cfg.conditioning.type}")
    logging.info(f"model: {cfg.model.type}")
    logging.info(f"resume_train: {cfg.train.resume_train}")
    logging.info(f"resume_model: {cfg.train.resume_model}")
    logging.info(f"save_model_iters: {cfg.train.save_model_iters}")
    logging.info(f"spatial_compression: {cfg.image_tokenizer.spatial_compression}")

    if cfg.one_sample:
        cfg = override_for_one_sample(cfg)

    logging.disable("cosmos_tokenizer")  # turnoff logging on tokenizer
    vid_vae = load_video_tokenizer(cfg)
    vid_vae = vid_vae.module if hasattr(vid_vae, "module") else vid_vae

    # Get dataloaders
    train_dataloader, val_dataloader = get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vid_vae,
        generator=gen,
    )

    # Create noise scheduler
    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    latent_channels = 16  # 3
    model = get_model(cfg, latent_channels)

    # Create sampler, used for generating vids and imgs from model
    sampler = samplers.Sampler(
        vid_vae,
        noise_scheduler,
        cfg.seed,
        cfg.image_tokenizer.spatial_compression,
        latent_channels,
        cfg.model.noise_steps,
    )

    # Load optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=1e-2,
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.train.num_epochs),
    )

    # Create accelerator
    accelerator = setup_distributed_training(cfg)
    if accelerator.is_main_process and not cfg.debug:
        wandb.init(
            project="diffusion-video-generator",
            name=f"{cfg.exp_prefix}: {datetime.datetime.now().strftime('%B-%d %H-%M-%S')}",
            config=get_wandb_config(cfg),
        )
        if cfg.log_dir is not None and not cfg.debug:
            os.makedirs(cfg.log_dir, exist_ok=True)
        accelerator.init_trackers("train_ddpm")

    # Wrap objects in accelerator
    (
        model,
        vid_vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        vid_vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )

    # Compile the model
    # if torch.cuda.is_available() and not cfg.debug:
    #     model = torch.compile(model)  # defaults to mode="reduce-overhead"

    # Create EMA model
    if not cfg.debug:
        ema_model = EMAModel(
            model,
            inv_gamma=cfg.model.ema_inv_gamma,
            power=cfg.model.ema_power,
            max_value=cfg.model.ema_max_decay,
        )

    # Create epoch and iteration counters
    start_epoch, start_iter = 0, 0
    if cfg.train.resume_train:
        accelerator.load_state(cfg.train.resume_model)
        path = os.path.basename(cfg.train.resume_model)
    global_step = 0

    # Calculate how often to validate and save
    val_iters = int(cfg.train.val_iters * cfg.train.gradient_accumulation_steps)
    save_model_iters = int(
        cfg.train.save_model_iters * cfg.train.gradient_accumulation_steps
    )

    # Initialize path to save directory
    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%B-%d-%H-%M")
    eval_dir = Path(cfg.log_dir) / (cfg.model.scheduler_type + " " + formatted_datetime)

    # If user requests, train model on just the first batch. Used in debugging.
    if cfg.one_sample:
        first_batch = next(iter(train_dataloader))
    grad_norm = 0.0

    for epoch in range(start_epoch, cfg.train.num_epochs):
        if accelerator.is_main_process:
            progress_bar = tqdm.tqdm(
                total=len(train_dataloader) // cfg.train.gradient_accumulation_steps,
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(
            islice(train_dataloader, start_iter, None), start=start_iter
        ):
            # Skip the first start_iter batches, used when resuming a training run
            # if step < start_iter:
            # continue

            if cfg.one_sample:
                batch = first_batch

            latents, batch = encode_batch(
                cfg, batch, accelerator, vid_vae=vid_vae, vae=vid_vae
            )

            # Sample timesteps and latent noise
            noise = torch.randn(latents.shape, device=latents.device)
            bs = latents.shape[0]
            u = torch.randn((bs, 1), device=accelerator.device)
            timesteps = 1.0 / (1.0 + torch.exp(-u))
            timesteps = timesteps
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            batch["noisy_latents"] = noisy_latents
            model.train()

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(batch, timesteps, accelerator.device, use_cfg=True)

                # Calculate loss and backprop
                noise_target = noise_scheduler.get_target(latents, noise, timesteps)
                loss = F.mse_loss(noise_pred, noise_target)
                accelerator.backward(loss)

                # Apply grad norms
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    grad_norm = grad_norm.detach().item()

                # Update optimizer, learning rate scheduler, and ema model
                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients and not cfg.debug:
                    ema_model.step(model)
                optimizer.zero_grad()

            # Log training loss
            if accelerator.sync_gradients and accelerator.is_main_process:
                logs = OrderedDict(
                    {
                        "train/loss": loss.detach().item(),
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }
                )
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(ordered_dict=logs)
                if not cfg.debug:
                    wandb.log(logs)

            # Save the checkpoint
            if (
                accelerator.is_main_process
                and (step + 1) % save_model_iters == 0
                and step > cfg.train.gradient_accumulation_steps
                and not cfg.debug
            ):
                os.makedirs(eval_dir, exist_ok=True)
                path = eval_dir / ("checkpoint-" + str(epoch) + "-" + str(step))
                accelerator.save_state(path, safe_serialization=True)
                ema_model.store(model.parameters())
                ema_model.copy_to(model.parameters())
                accelerator.save_model(
                    model, eval_dir / ("checkpoint-" + str(epoch)) / "ema"
                )
                ema_model.restore(model.parameters())

            # Sample some imgs and videos, log them to wandb
            if (
                ((step + 1) % val_iters == 0 or step == len(train_dataloader) - 1)
                and step > cfg.train.gradient_accumulation_steps
                and cfg.val.run
            ):
                path = eval_dir / ("samples-" + str(epoch) + "-" + str(step) + ".png")
                if cfg.debug:
                    path = "debug/debug.png"
                model_unwrapped = model  # model # accelerator.unwrap_model(model)
                model_unwrapped.eval()

                # Log validation loss
                if not cfg.val.skip_val_loss:
                    val_loss = compute_val_loss(
                        cfg,
                        model_unwrapped,
                        val_dataloader,
                        noise_scheduler,
                        accelerator,
                        vid_vae=vid_vae,
                    )
                    if accelerator.is_main_process:
                        if not cfg.debug:
                            wandb.log(
                                {
                                    "validation/loss": val_loss,
                                    "validation/epoch": epoch,
                                    "validation/step": step,
                                }
                            )
                        logging.info(f"Validation loss: {val_loss:.6f}")

                # Sample some videos, log them to wandb
                if accelerator.is_main_process and not cfg.val.skip_img_sample:
                    with torch.no_grad():
                        sample_video_seqs = None
                        sample_imgs = None
                        if "video" in cfg.gen_type.lower():
                            sample_video_seqs, sample_imgs = sampler.sample_video(
                                cfg,
                                dataloader=train_dataloader,
                                batch=batch,
                                batch_idx=2,
                                vae=vid_vae,
                                accelerator=accelerator,
                                model=model_unwrapped,
                                guidance_scale=cfg.model.cfg_scale,
                                dtype=noisy_latents.dtype,
                                device=accelerator.device,
                            )
                        elif "future_frame" in cfg.gen_type.lower():
                            sample_imgs = sampler.sample_future_frame(
                                cfg,
                                batch=batch,
                                dataloader=val_dataloader,
                                batch_idx=0,
                                vid_vae=vid_vae,
                                accelerator=accelerator,
                                model=model_unwrapped,
                                guidance_scale=cfg.model.cfg_scale,
                                dtype=noisy_latents.dtype,
                                device=accelerator.device,
                            )

                        log_img_and_vids_wandb(
                            cfg,
                            accelerator,
                            eval_dir,
                            path,
                            epoch,
                            step,
                            sample_video_seqs,
                            sample_imgs,
                        )
                        logging.info(f"Sampled images to : {path}")

        # Reset start_iter
        start_iter = 0

        # Save the model at the end of the epoch
        if accelerator.is_main_process and not cfg.debug:
            os.makedirs(eval_dir, exist_ok=True)
            path = eval_dir / ("checkpoint-" + str(epoch))
            accelerator.save_state(path, safe_serialization=True)
            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())
            accelerator.save_model(
                model, eval_dir / ("checkpoint-" + str(epoch) + "-final") / "ema"
            )
            ema_model.restore(model.parameters())

    if accelerator.is_main_process and not cfg.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
