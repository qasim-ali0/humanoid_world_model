from functools import partial

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .future_frame_dataset import (
    FutureFrameDataset,
    FutureFrameTestDataset,
    FutureVideoDataset,
)
from .future_video_dataset import FutureVideoDataset


def get_dataloaders(
    dataset_type,
    cfg,
    vae=None,
    return_datasets=False,
    val_stride=None,
    generator=None,
):
    """
    Factory function to return train and validation dataloaders based on the dataset type.
    """
    hmwm_train_dir = cfg.data.hmwm_train_dir
    hmwm_val_dir = cfg.data.hmwm_val_dir
    hmwm_test_dir = cfg.data.hmwm_test_dir
    train_batch_size = cfg.train.batch_size
    val_batch_size = cfg.val.batch_size
    test_batch_size = val_batch_size
    conditioning_type = cfg.conditioning.type
    num_past_frames = cfg.conditioning.num_past_frames
    num_future_frames = cfg.conditioning.num_future_frames

    if dataset_type == "1xgpt_future_frame":
        assert num_past_frames != None
        assert num_future_frames != None
        with_action = False
        if conditioning_type == "action":
            with_action = True
        train_dataset = FutureFrameDataset(
            hmwm_train_dir,
            cfg,
            n_input=17,
            n_output=1,
            n_intermediate=59,
            with_actions=with_action,
            stride=1,
        )
        val_dataset = FutureFrameDataset(
            hmwm_val_dir,
            cfg,
            n_input=17,
            n_output=1,
            n_intermediate=59,
            with_actions=with_action,
            stride=num_past_frames // 2 if val_stride == None else val_stride,
        )
        if return_datasets:
            return train_dataset, val_dataset
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=generator,
            **get_dataloader_kwargs(),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            generator=generator,
            shuffle=False,
            **get_dataloader_kwargs(),
        )
        return train_dataloader, val_dataloader

    elif dataset_type == "1xgpt_test":
        assert (
            conditioning_type != "text"
        ), "Conditioning must not be 'text' for 1xgpt dataset."
        assert num_past_frames != None
        assert num_future_frames != None
        with_action = False
        if conditioning_type == "action":
            with_action = True
        test_dataset = FutureFrameTestDataset(
            hmwm_test_dir,
            cfg,
            n_input=17,
            n_output=1,
            n_intermediate=59,
            with_actions=with_action,
            stride=1,
        )
        if return_datasets:
            return test_dataset
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            **get_dataloader_kwargs(),
        )
        return test_dataloader

    elif dataset_type == "1xgpt_video":
        assert (
            conditioning_type != "text"
        ), "Conditioning must not be 'text' for 1xgpt dataset."
        assert num_past_frames != None
        assert num_future_frames != None
        with_action = False
        if conditioning_type == "action":
            with_action = True
        train_dataset = FutureVideoDataset(
            hmwm_train_dir,
            cfg,
            vae,
            n_input=num_past_frames,
            n_output=num_future_frames,
            with_actions=with_action,
            stride=1,
        )
        val_dataset = FutureVideoDataset(
            hmwm_val_dir,
            cfg,
            vae,
            n_input=num_past_frames,
            n_output=num_future_frames,
            with_actions=with_action,
            stride=num_past_frames // 2 if val_stride == None else val_stride,
        )
        if return_datasets:
            return train_dataset, val_dataset
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            generator=generator,
            **get_dataloader_kwargs(),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            generator=generator,
            **get_dataloader_kwargs(),
        )

        return train_dataloader, val_dataloader

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataloader_kwargs():
    """Get kwargs for DataLoader in distributed setting"""
    return {
        "pin_memory": True,
        "num_workers": 4,  # Increased from 0
        "prefetch_factor": 2,  # Added prefetch
        "persistent_workers": True,  # Keep workers alive between epochs
    }


def no_collate_fn(batch):
    return batch


def encode_batch(cfg, batch, accelerator, vae=None, img_vae=None, vid_vae=None):
    if "video" in cfg.gen_type.lower():
        batch = encode_video_batch(cfg, batch, vae)
        return batch["future_latents"], batch
    elif "future_frame" in cfg.gen_type.lower():
        if type(batch["past_frames"]) == list:
            batch["past_frames"] = torch.concat(batch["past_frames"], 0)
        batch = encode_batch_key(cfg, batch, vid_vae, "past_frames", "past_latents")
        batch = encode_batch_key(cfg, batch, img_vae, "future_frames", "future_latents")
        return batch["future_latents"], batch
    else:
        imgs = batch["imgs"]
        with accelerator.autocast():
            imgs = imgs.to(getattr(torch, cfg.image_tokenizer.dtype))
            latents = vae.encode(imgs)[0]
        return latents, batch


def encode_video_batch(cfg, batch, vae):
    orig_dtype = batch["past_frames"].dtype
    past_frames = batch["past_frames"].to(vae._dtype)
    future_frames = batch["future_frames"].to(vae._dtype)
    device = next(vae.parameters()).device
    past_latents, _ = create_condition_latent(
        vae,
        past_frames,
        cfg.conditioning.num_past_frames,
        cfg.conditioning.num_future_frames,
        cfg.conditioning.num_past_latents,
        cfg.conditioning.num_future_latents,
        device,
    )
    _, future_latents = create_label_latent(
        vae,
        past_frames,
        future_frames,
        cfg.conditioning.num_past_latents,
        cfg.conditioning.num_future_latents,
        device,
    )
    batch["past_latents"] = past_latents.to(orig_dtype)
    batch["future_latents"] = future_latents.to(orig_dtype)
    return batch


def encode_batch_key(cfg, batch, vae, frame_key, latent_key):
    orig_dtype = batch[frame_key].dtype
    frames = batch[frame_key].to(vae._dtype)
    device = next(vae.parameters()).device
    (latent,) = vae.encode(frames.to(device))
    batch[latent_key] = latent.to(orig_dtype)
    return batch


def create_condition_latent(
    tokenizer,
    past_frames,
    num_past_frames,
    num_future_frames,
    num_past_latents,
    num_future_latent,
    device,
):
    B, C, T, H, W = past_frames.shape

    padding_frames = past_frames.new_zeros(B, C, num_future_frames, H, W)
    encode_past_frames = torch.cat([past_frames, padding_frames], dim=2)
    (latent,) = tokenizer.encode(encode_past_frames.to(device))

    past_latent = latent[:, :, :num_past_latents]
    future_latent = latent[:, :, num_past_latents:]
    assert future_latent.shape[2] == num_future_latent
    return past_latent, future_latent


def create_label_latent(
    tokenizer, past_frames, future_frames, num_past_latents, num_future_latent, device
):
    B, C, T, H, W = past_frames.shape
    all_frames = torch.concatenate((past_frames, future_frames), 2)
    (latent,) = tokenizer.encode(all_frames.to(device))
    past_latent = latent[:, :, :num_past_latents]
    future_latent = latent[:, :, num_past_latents:]
    assert future_latent.shape[2] == num_future_latent
    return past_latent, future_latent
