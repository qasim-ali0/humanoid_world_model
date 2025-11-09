import numpy as np
import PIL
import PIL.Image
import torch
import tqdm
from diffusers.utils import make_image_grid
from einops import rearrange

from data import encode_batch


class Sampler:
    def __init__(
        self,
        vae,
        scheduler,
        seed,
        spatial_compression,
        latent_channels,
        num_inference_steps,
    ):
        self.vae = vae
        self.seed = seed
        self.scheduler = scheduler
        self.spatial_compression = spatial_compression
        self.latent_channels = latent_channels
        self.num_inference_steps = num_inference_steps

    def sample_video(
        self,
        cfg,
        model,
        vae,
        accelerator,
        guidance_scale,
        n_samples=4,
        dataloader=None,
        batch_idx=None,
        dtype=torch.float32,
        batch=None,
        device="cuda",
        use_progress_bar=True,
    ):
        if batch == None:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            for i in range(batch_idx):
                batch = next(dataloader_iter)
        latents, batch = encode_batch(cfg, batch, accelerator, vae)
        for key in batch.keys():
            batch[key] = batch[key][:n_samples]
        generator = torch.Generator(device=device).manual_seed(self.seed)
        latents = torch.randn(
            batch["future_latents"].shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        batch["noisy_latents"] = latents

        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)
        progress_bar = self.progress_bar(timesteps) if use_progress_bar else timesteps
        for t in progress_bar:
            # 1. predict noise model_output
            # model_output = unet(latents, t).sample
            t = t.repeat((batch["noisy_latents"].shape[0], 1))
            pred_cond = model(batch, t, device, use_cfg=False)
            pred_uncond = model(
                batch, t, device, use_cfg=False, force_drop_context=True
            )
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # 2. compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(pred, t, latents, generator=generator)

        vae = self.vae.module if hasattr(self.vae, "module") else self.vae

        all_pred_latents = torch.concatenate((batch["past_latents"], latents), 2)
        all_pred_frames = vae.decode(all_pred_latents.to(torch.bfloat16))

        all_pred_video = split_video_to_imgs(denormalize_video(all_pred_frames))
        pred_output_frames = split_video_to_imgs(
            denormalize_video(all_pred_frames[:, :, cfg.conditioning.num_past_frames :])
        )
        gt_input_frames = split_video_to_imgs(denormalize_video(batch["past_frames"]))
        gt_output_frames = split_video_to_imgs(
            denormalize_video(batch["future_frames"])
        )

        img_grid_fullvideo = []
        img_grid_fullvideo_tokenizedprompt = []
        img_grid_generated = []
        for i in range(all_pred_frames.shape[0]):
            img_grid_fullvideo.append(
                make_image_grid(all_pred_video[i], rows=1, cols=len(all_pred_video[i]))
            )
            img_grid_fullvideo_tokenizedprompt.append(
                make_image_grid(
                    gt_input_frames[i] + pred_output_frames[i],
                    rows=1,
                    cols=len(all_pred_video[i]),
                )
            )
            img_grid_generated.append(
                make_image_grid(
                    pred_output_frames[i] + gt_output_frames[i],
                    2,
                    cfg.conditioning.num_future_frames,
                )
            )
        return all_pred_video, (
            img_grid_fullvideo,
            img_grid_fullvideo_tokenizedprompt,
            img_grid_generated,
        )

    def sample_future_frame(
        self,
        cfg,
        model,
        img_vae,
        vid_vae,
        accelerator,
        guidance_scale,
        n_samples=16,
        dataloader=None,
        batch_idx=None,
        dtype=torch.float32,
        batch=None,
        device="cuda",
        use_progress_bar=True,
    ):
        if batch == None:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
            for i in range(batch_idx):
                batch = next(dataloader_iter)
        _, batch = encode_batch(
            cfg, batch, accelerator, img_vae=img_vae, vid_vae=vid_vae
        )
        for key in batch.keys():
            batch[key] = batch[key][:n_samples]
        generator = torch.Generator(device=device).manual_seed(self.seed)
        batch["noisy_latents"] = torch.randn(
            batch["future_latents"].shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.scheduler.set_timesteps(self.num_inference_steps)
        timesteps = self.scheduler.timesteps.to(device)
        progress_bar = self.progress_bar(timesteps) if use_progress_bar else timesteps
        for t in progress_bar:
            t = t.repeat((batch["noisy_latents"].shape[0], 1))
            pred_cond = model(batch, t, device, use_cfg=False)
            pred_uncond = model(
                batch, t, device, use_cfg=False, force_drop_context=True
            )
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            # 2. compute previous image: x_t -> x_t-1
            batch["noisy_latents"] = self.scheduler.step(
                pred, t, batch["noisy_latents"], generator=generator
            )
        img_vae = img_vae.module if hasattr(img_vae, "module") else img_vae
        # batch['noisy_latents'] = batch['noisy_latents'].squeeze(2)
        pred_img = img_vae.decode(batch["noisy_latents"].to(torch.bfloat16))
        pred_img = denormalize_img(pred_img.squeeze(2))
        pred_img = [PIL.Image.fromarray(s) for s in pred_img]
        return pred_img

    def sample_video_autoregressive(
        self,
        cfg,
        train_dataloader,
        model,
        batch_idx,
        vae,
        accelerator,
        guidance_scale,
        dtype=torch.float32,
        device="cuda",
    ):
        n_samples = 4

        train_dataloader_iter = iter(train_dataloader)
        batch = next(train_dataloader_iter)
        for i in range(batch_idx):
            batch = next(train_dataloader_iter)
        latents, batch = encode_batch(cfg, batch, vae, accelerator)
        for key in batch.keys():
            batch[key] = batch[key][:n_samples]
        generator = torch.Generator(device=device).manual_seed(self.seed)

        n_output_frames = 8
        n_input_frames = 9
        final_pred_latents = []
        for i in range(n_output_frames):
            latents = torch.randn(
                batch["future_latents"].shape,
                dtype=dtype,
                device=device,
                generator=generator,
            )
            batch_i = {
                "noisy_latents": latents,
                "past_actions": batch["past_actions"][:, i : n_input_frames + i],
                "future_actions": batch["past_actions"][
                    :, i : n_output_frames + i + n_input_frames
                ],
                "past_latents": vae.encode(
                    batch["past_latents"][:, i : n_input_frames + i].to(torch.bfloat16)
                )[0],
            }

            self.scheduler.set_timesteps(self.num_inference_steps)
            timesteps = self.scheduler.timesteps.to(device)
            for t in self.progress_bar(timesteps):
                # 1. predict noise model_output
                # model_output = unet(latents, t).sample
                t = t.repeat((batch_i["noisy_latents"][0].shape[0], 1))
                pred_cond = model(batch_i, t, device, use_cfg=False)
                pred_uncond = model(
                    batch_i, t, device, use_cfg=False, force_drop_context=True
                )
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                # 2. compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(pred, t, latents, generator=generator)
                final_pred_latents.append(latents[:, 0])

        final_pred_latents = torch.concat(final_pred_latents, 0).move_axis(0, 1)

        vae = self.vae.module if hasattr(self.vae, "module") else self.vae

        all_pred_latents = torch.concatenate(
            (
                vae.encode(batch["past_frames"][:n_input_frames])[0][:, n_input_frames],
                latents,
            ),
            2,
        )
        all_pred_frames = vae.decode(all_pred_latents.to(torch.bfloat16))

        all_pred_video = split_video_to_imgs(denormalize_video(all_pred_frames))
        pred_output_frames = split_video_to_imgs(
            denormalize_video(all_pred_frames[:, :, cfg.conditioning.num_past_frames :])
        )
        gt_input_frames = split_video_to_imgs(
            denormalize_video(batch["past_frames"][:, n_input_frames])
        )
        gt_output_frames = split_video_to_imgs(
            denormalize_video(batch["future_frames"])
        )

        img_grid_fullvideo = []
        img_grid_fullvideo_tokenizedprompt = []
        img_grid_generated = []
        for i in range(all_pred_frames.shape[0]):
            img_grid_fullvideo.append(
                make_image_grid(all_pred_video[i], rows=1, cols=len(all_pred_video[i]))
            )
            img_grid_fullvideo_tokenizedprompt.append(
                make_image_grid(
                    gt_input_frames[i] + pred_output_frames[i],
                    rows=1,
                    cols=len(all_pred_video[i]),
                )
            )
            img_grid_generated.append(
                make_image_grid(
                    pred_output_frames[i] + gt_output_frames[i],
                    2,
                    cfg.conditioning.num_future_frames,
                )
            )
        return all_pred_video, (
            img_grid_fullvideo,
            img_grid_fullvideo_tokenizedprompt,
            img_grid_generated,
        )

    def progress_bar(self, iterable=None, total=None):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        elif not isinstance(self._progress_bar_config, dict):
            raise ValueError(
                f"`self._progress_bar_config` should be of type `dict`, but is {type(self._progress_bar_config)}."
            )

        if iterable is not None:
            return tqdm.tqdm(iterable, **self._progress_bar_config)
        elif total is not None:
            return tqdm.tqdm(total=total, **self._progress_bar_config)
        else:
            raise ValueError("Either `total` or `iterable` has to be defined.")


def denormalize_img(preds):
    output_imgs = (preds.float() + 1.0) / 2.0
    output_imgs = rearrange(output_imgs, "b c h w -> b h w c")
    output_imgs = output_imgs.clamp(0, 1).cpu().numpy()
    _UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
    output_imgs = output_imgs * _UINT8_MAX_F + 0.5
    output_imgs = output_imgs.astype(np.uint8)
    return output_imgs


def denormalize_video(preds):
    output_video = (preds.float() + 1.0) / 2.0
    output_video = rearrange(output_video, "b c t h w -> b t c h w")
    output_video = output_video.clamp(0, 1).cpu().numpy()
    _UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)
    output_video = output_video * _UINT8_MAX_F
    output_video = output_video.astype(np.uint8)
    return output_video


def split_video_to_imgs(video):
    b, t, c, h, w = video.shape
    output_images = [[] for _ in range(b)]
    for i in range(b):
        for j in range(t):
            img = np.moveaxis(video[i][j], 0, 2)
            output_images[i].append(PIL.Image.fromarray(img))
    return output_images


def sample_hf_pipeline(pipeline, seed):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=16,
        generator=torch.Generator(device="cpu").manual_seed(
            seed
        ),  # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images
    return images
