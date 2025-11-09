import datetime
import os
import shutil
from collections import OrderedDict
from itertools import islice
from pathlib import Path

import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedDataParallelKwargs, RNGType, set_seed
from cosmos_tokenizer.image_lib import ImageTokenizer
from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from diffusers.utils import make_image_grid
from einops import rearrange
from loguru import logger
from loguru import logger as logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

import data
import samplers
import wandb
from conditioning import ConditioningManager
from data import encode_batch, get_dataloaders
from models import get_model
from models.unet import UNet
from schedulers import get_scheduler


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


def compute_val_loss(
    cfg,
    model,
    val_dataloader,
    noise_scheduler,
    accelerator,
    progress_bar_enabled=True,
    vae=None,
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
        cfg.train.learning_rate = 1e-4  # 8e-5
        cfg.train.lr_warmup_steps = 0
        cfg.train.gradient_accumulation_steps = 1
        cfg.train.batch_size = 4
        cfg.val.batch_size = 1
        cfg.exp_prefix = "one-sample"
        cfg.train.save_model_iters = 60000
        cfg.train.val_iters = 5
        cfg.conditioning.prompt_file = "prompts_one_sample.txt"
        cfg.val.run = True
        cfg.val.skip_val_loss = True
        cfg.val.skip_img_sample = False

    # tokenizer_config = TokenizerConfigs['CI'].value
    # tokenizer_config.update(dict(spatial_compression=cfg.image_tokenizer.spatial_compression))
    logger.disable("cosmos_tokenizer")  # turnoff logging
    model_name = f"Cosmos-Tokenizer-CI{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
    img_vae = ImageTokenizer(
        checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
        checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
        checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
        # tokenizer_config=tokenizer_config,
        device=None,
        dtype=cfg.image_tokenizer.dtype,
    )
    # tokenizer_config = TokenizerConfigs['CV'].value
    # tokenizer_config.update(dict(spatial_compression=cfg.image_tokenizer.spatial_compression))
    model_name = f"Cosmos-0.1-Tokenizer-CV{cfg.image_tokenizer.temporal_compression}x{cfg.image_tokenizer.spatial_compression}x{cfg.image_tokenizer.spatial_compression}"
    vid_vae = CausalVideoTokenizer(
        checkpoint=Path(cfg.image_tokenizer.path) / model_name / "autoencoder.jit",
        checkpoint_enc=Path(cfg.image_tokenizer.path) / model_name / "encoder.jit",
        checkpoint_dec=Path(cfg.image_tokenizer.path) / model_name / "decoder.jit",
        # tokenizer_config=tokenizer_config,
        device=None,
        dtype="bfloat16",
    )

    # img_vae = VAEWrapper() # img_vae.to(cfg.device, dtype=cfg.image_tokenizer.dtype)
    img_vae = vid_vae

    for param in img_vae.parameters():
        param.requires_grad = False
    for param in vid_vae.parameters():
        param.requires_grad = False

    conditioning_manager = ConditioningManager(cfg.conditioning)

    train_dataloader, val_dataloader = get_dataloaders(
        cfg.data.type,
        cfg,
        vae=vid_vae,
        img_vae=img_vae.module if hasattr(img_vae, "module") else img_vae,
        vid_vae=vid_vae.module if hasattr(vid_vae, "module") else vid_vae,
        hmwm_train_dir=cfg.data.hmwm_train_dir,
        hmwm_val_dir=cfg.data.hmwm_val_dir,
        coco_train_imgs=cfg.data.coco_train_imgs,
        coco_val_imgs=cfg.data.coco_val_imgs,
        coco_train_ann=cfg.data.coco_train_ann,
        coco_val_ann=cfg.data.coco_val_ann,
        image_size=cfg.image_size,
        train_batch_size=cfg.train.batch_size,
        val_batch_size=cfg.val.batch_size,
        conditioning_type=cfg.conditioning.type,
        conditioning_manager=conditioning_manager,
        num_past_frames=cfg.conditioning.get("num_past_frames"),
        num_future_frames=cfg.conditioning.get("num_future_frames"),
        generator=gen,
    )

    noise_scheduler = get_scheduler(cfg.model.scheduler_type, cfg.model.noise_steps)

    latent_channels = 16  # 3

    model = get_model(
        cfg,
        latent_channels,
        conditioning_manager,
        cfg.image_size // cfg.image_tokenizer.spatial_compression,
    )

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

    sampler = samplers.Sampler(
        vid_vae,
        noise_scheduler,
        cfg.seed,
        cfg.image_tokenizer.spatial_compression,
        latent_channels,
        cfg.model.noise_steps,
    )

    accelerator = setup_distributed_training(cfg)
    if accelerator.is_main_process and not cfg.debug:
        wandb.init(
            project="diffusion-video-generator",
            name=f"{cfg.exp_prefix}: {datetime.datetime.now().strftime('%B-%d %H-%M-%S')}",
            config={
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
            },
        )
        if cfg.log_dir is not None and not cfg.debug:
            os.makedirs(cfg.log_dir, exist_ok=True)
        accelerator.init_trackers("train_ddpm")

    (
        model,
        img_vae,
        vid_vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        model,
        img_vae,
        vid_vae,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
    )
    conditioning_manager.to(accelerator.device)

    # Compile the model
    # if torch.cuda.is_available() and not cfg.debug:
    #     model = torch.compile(model)  # defaults to mode="reduce-overhead"

    if not cfg.debug:
        ema_model = EMAModel(
            model,
            inv_gamma=cfg.model.ema_inv_gamma,
            power=cfg.model.ema_power,
            max_value=cfg.model.ema_max_decay,
        )

    start_epoch, start_iter = 0, 0
    if cfg.train.resume_train:
        accelerator.load_state(cfg.train.resume_model)
        path = os.path.basename(cfg.train.resume_model)
        # start_epoch, start_iter = int(path.split("-")[1]), int(path.split("-")[2])

    now = datetime.datetime.now()
    formatted_datetime = now.strftime("%B-%d-%H-%M")
    eval_dir = Path(cfg.log_dir) / (cfg.model.scheduler_type + " " + formatted_datetime)

    img_vae, vid_vae = img_vae.module if hasattr(img_vae, "module") else img_vae, (
        vid_vae.module if hasattr(vid_vae, "module") else vid_vae
    )
    if cfg.one_sample:
        first_batch = next(iter(train_dataloader))
    global_step = 0
    grad_norm = 0.0
    val_iters = int(cfg.train.val_iters * cfg.train.gradient_accumulation_steps)
    save_model_iters = int(
        cfg.train.save_model_iters * cfg.train.gradient_accumulation_steps
    )

    for epoch in range(start_epoch, cfg.train.num_epochs):
        if accelerator.is_main_process:
            progress_bar = tqdm.tqdm(
                total=len(train_dataloader) // cfg.train.gradient_accumulation_steps,
                disable=not accelerator.is_local_main_process,
            )
            progress_bar.set_description(f"Epoch {epoch}")

        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(
            islice(train_dataloader, start_iter, None), start=start_iter
        ):
            # if step < start_iter:
            # continue
            if cfg.one_sample:
                batch = first_batch

            latents, batch = encode_batch(
                cfg, batch, accelerator, img_vae=img_vae, vid_vae=vid_vae, vae=vid_vae
            )
            noise = torch.randn(latents.shape, device=latents.device)
            bs = latents.shape[0]
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
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            batch["noisy_latents"] = noisy_latents
            model.train()
            with accelerator.accumulate(model):
                # Predict the noise residual
                # noise_pred = model(noisy_latents, timesteps, return_dict=False)[0]
                noise_pred = model(batch, timesteps, accelerator.device, use_cfg=True)
                noise_target = noise_scheduler.get_target(latents, noise, timesteps)
                loss = F.mse_loss(noise_pred, noise_target)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    grad_norm = grad_norm.detach().item()

                optimizer.step()
                lr_scheduler.step()
                if accelerator.sync_gradients and not cfg.debug:
                    ema_model.step(model)
                optimizer.zero_grad()

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
                        vae=vid_vae,
                        img_vae=img_vae,
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

                # Sample some images
                if accelerator.is_main_process and not cfg.val.skip_img_sample:
                    with torch.no_grad():
                        if "video" in cfg.gen_type.lower():
                            sample_videos, sample_grids = sampler.sample_video(
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
                            samples = sampler.sample_future_frame(
                                cfg,
                                batch=batch,
                                dataloader=val_dataloader,
                                batch_idx=0,
                                img_vae=img_vae,
                                vid_vae=vid_vae,
                                accelerator=accelerator,
                                model=model_unwrapped,
                                guidance_scale=cfg.model.cfg_scale,
                                dtype=noisy_latents.dtype,
                                device=accelerator.device,
                            )
                        elif cfg.conditioning.type == "text":
                            samples = sampler.sample_textcond_img(
                                model=model_unwrapped,
                                img_size=cfg.image_size,
                                in_channels=latent_channels,
                                prompt_file=cfg.conditioning.prompt_file,
                                text_tokenizer=conditioning_manager.get_module()[
                                    "text"
                                ],
                                device=accelerator.device,
                                dtype=noisy_latents.dtype,
                                guidance_scale=cfg.model.cfg_scale,
                            )
                        else:
                            samples = sampler.sample_uncond_img(
                                unet=model_unwrapped,
                                img_size=cfg.image_size,
                                in_channels=latent_channels,
                                device=accelerator.device,
                                dtype=noisy_latents.dtype,
                            )

                        if (
                            "image" in cfg.gen_type.lower()
                            or "frame" in cfg.gen_type.lower()
                        ):
                            rows = max(len(samples) // 4, 1)
                            cols = len(samples) // max(rows, 1)
                            image_grid = make_image_grid(samples, rows=rows, cols=cols)
                            os.makedirs(eval_dir, exist_ok=True)
                            image_grid.save(path)
                            logging.info(f"Sampled images to : {path}")
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
                                path_img0 = os.path.join(
                                    eval_dir, f"tokenizedprompt+predictions.jpeg"
                                )
                                sample_grids[0][i].save(path_img0)

                                path_img1 = os.path.join(
                                    eval_dir, f"gtprompt+predictions.jpeg"
                                )
                                sample_grids[1][i].save(path_img1)

                                path_img2 = os.path.join(
                                    eval_dir, f"gt_vs_predictions.jpeg"
                                )
                                sample_grids[2][i].save(path_img2)

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

        start_iter = 0
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
