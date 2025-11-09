from functools import partial

import torch
import torchvision.transforms as transforms
from einops import pack, unpack
from torch.utils.data import DataLoader

from .video_dataset import (
    FutureFrameDataset,
    FutureFrameTestSet,
    RawVideoDataset,
    encode_batch_key,
    encode_video_batch,
)


def get_dataloaders(
    data_type,
    cfg,
    hmwm_train_dir,
    hmwm_val_dir,
    coco_train_imgs,
    coco_train_ann,
    coco_val_imgs,
    coco_val_ann,
    conditioning_type,
    image_size,
    train_batch_size,
    val_batch_size,
    hmwm_test_dir=None,
    test_batch_size=None,
    num_past_frames=None,
    num_future_frames=None,
    vae=None,
    img_vae=None,
    vid_vae=None,
    conditioning_manager=None,
    return_datasets=False,
    val_stride=None,
    generator=None,
):
    """Factory function to return train and validation dataloaders based on the dataset type."""
    if data_type == "1xgpt_image":
        assert (
            conditioning_type != "text"
        ), "Conditioning must not be 'text' for 1xgpt dataset."
        train_dataset = RawImageDataset(hmwm_train_dir)
        val_dataset = RawImageDataset(hmwm_val_dir)
        if return_datasets:
            return train_dataset, val_dataset
    elif data_type == "1xgpt_future_frame":
        assert (
            conditioning_type != "text"
        ), "Conditioning must not be 'text' for 1xgpt dataset."
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
        )  # num_past_frames // 2
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
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae),
            **get_dataloader_kwargs(),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            generator=generator,
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae),
            shuffle=False,
            **get_dataloader_kwargs(),
        )
        return train_dataloader, val_dataloader
    elif data_type == "1xgpt_test":
        assert (
            conditioning_type != "text"
        ), "Conditioning must not be 'text' for 1xgpt dataset."
        assert num_past_frames != None
        assert num_future_frames != None
        with_action = False
        if conditioning_type == "action":
            with_action = True
        test_dataset = FutureFrameTestSet(
            hmwm_test_dir,
            cfg,
            n_input=17,
            n_output=1,
            n_intermediate=59,
            with_actions=with_action,
            stride=1,
        )  # num_past_frames // 2
        if return_datasets:
            return test_dataset
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae),
            **get_dataloader_kwargs(),
        )
        return test_dataloader
    elif data_type == "1xgpt_video":
        assert (
            conditioning_type != "text"
        ), "Conditioning must not be 'text' for 1xgpt dataset."
        assert num_past_frames != None
        assert num_future_frames != None
        with_action = False
        if conditioning_type == "action":
            with_action = True
        train_dataset = RawVideoDataset(
            hmwm_train_dir,
            cfg,
            vae,
            n_input=num_past_frames,
            n_output=num_future_frames,
            with_actions=with_action,
            stride=1,
        )  # num_past_frames // 2
        val_dataset = RawVideoDataset(
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
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae),
            **get_dataloader_kwargs(),
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            # collate_fn=partial(RawVideoDataset_collate_fn, cfg, vae),
            shuffle=False,
            generator=generator,
            **get_dataloader_kwargs(),
        )

        return train_dataloader, val_dataloader
    elif data_type.lower() == "coco":
        assert (
            conditioning_type == "text"
        ), "Conditioning must be 'text' for COCO dataset."
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        train_dataset = CustomCoco(
            root=coco_train_imgs,
            annFile=coco_train_ann,
            text_tokenizer=conditioning_manager.get_module()["text"],
            transform=transform,
        )
        val_dataset = CustomCoco(
            root=coco_val_imgs,
            annFile=coco_val_ann,
            text_tokenizer=conditioning_manager.get_module()["text"],
            transform=transform,
        )
        if return_datasets:
            return train_dataset, val_dataset
    else:
        raise ValueError(f"Unknown dataset type: {data_type}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        generator=generator,
        # collate_fn=no_collate_fn,
        **get_dataloader_kwargs(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        # collate_fn=no_collate_fn,
        shuffle=False,
        generator=generator,
        **get_dataloader_kwargs(),
    )

    return train_dataloader, val_dataloader


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
        # batch['future_frames'] = batch['future_frames'].squeeze(2)
        batch = encode_batch_key(cfg, batch, img_vae, "future_frames", "future_latents")
        # batch['future_latents'] = batch['future_latents'].unsqueeze(2)
        return batch["future_latents"], batch
    else:
        imgs = batch["imgs"]
        with accelerator.autocast():
            imgs = imgs.to(getattr(torch, cfg.image_tokenizer.dtype))
            latents = vae.encode(imgs)[0]
        return latents, batch
