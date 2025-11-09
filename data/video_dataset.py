import json
import math
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class RawVideoDataset(TorchDataset):
    def __init__(
        self,
        data_dir,
        cfg,
        vae,
        n_input=8,
        n_output=1,
        stride=4,
        skip_frame=1,
        with_actions=True,
        down_sample=2,
    ):
        super().__init__()
        self._UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

        # mp.set_start_method('spawn')

        self.data_dir = Path(data_dir)
        self.vae = vae
        self.cfg = cfg
        self.down_sample = down_sample
        self.image_size = cfg.image_size

        # Load main metadata
        with open(self.data_dir / "metadata.json") as f:
            metadata = json.load(f)
            self.num_shards = metadata["num_shards"]
            self.query = metadata["query"]
            self.hz = metadata["hz"]
            self.num_images = metadata["num_images"]

        # Load shard-specific metadata
        self.shard_sizes = []
        for shard in range(self.num_shards):
            with open(self.data_dir / f"metadata/metadata_{shard}.json") as f:
                shard_metadata = json.load(f)
                self.shard_sizes.append(shard_metadata["shard_num_frames"])

        # Calculate cumulative shard sizes for index mapping
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        assert (
            self.cumulative_sizes[-1] == self.num_images
        ), "Metadata mismatch in total number of frames"

        # Store video paths instead of keeping captures open
        self.video_paths = [
            str(self.data_dir / f"videos/video_{shard}.mp4")
            for shard in range(self.num_shards)
        ]

        # Store action paths, and open them up if with_actions
        self.action_paths = [
            str(self.data_dir / f"states/states_{shard}.bin")
            for shard in range(self.num_shards)
        ]
        self.with_actions = with_actions
        if self.with_actions:
            self.action_shards = []
            for i, path in enumerate(self.action_paths):
                action_shard = np.memmap(
                    path, dtype=np.float32, mode="r", shape=(self.shard_sizes[i], 25)
                )
                self.action_shards.append(action_shard)
            self.action_shards = np.concatenate(self.action_shards, 0)
            self.action_shards = (
                self.action_shards - np.mean(self.action_shards, 0)
            ) / np.std(self.action_shards, 0)

        # Store action paths, and open them up if with_actions
        self.segment_paths = [
            str(self.data_dir / f"segment_idx/segment_idx_{shard}.bin")
            for shard in range(self.num_shards)
        ]

        segment_shards = []
        for i, path in enumerate(self.segment_paths):
            segment_shard = np.memmap(
                path, dtype=np.int32, mode="r", shape=(self.shard_sizes[i], 1)
            )
            segment_shards.append(segment_shard)
        self.segment_shards = np.concatenate(segment_shards).squeeze(-1)

        # Compute the valid start indexes
        self.n_input, self.n_output, self.stride, self.skip_frame = (
            n_input,
            n_output,
            stride,
            skip_frame,
        )
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (
            self.n_output + self.n_input
        ) * down_sample  # * self.skip_frame

        start_indices = np.arange(0, self.num_images - self.video_len, self.stride)
        end_indices = start_indices + self.video_len

        start_segment_ids = self.segment_shards[start_indices]  #
        end_segment_ids = self.segment_shards[end_indices]  #

        self.valid_start_inds = start_indices[start_segment_ids == end_segment_ids]

        # Verify all video files exist and are readable
        for path in self.video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video file: {path}")
            cap.release()

        # Verify all action files exist and are readable
        for path in self.action_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")

    def get_index_info(self, idx):
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side="right")
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        return shard_idx, frame_idx

    def extract_frames_opencv(self, start_shard, end_shard, start_frame, end_frame):
        if start_shard == end_shard:
            # if the shards are same then we can get all frames from the same video
            video_path = self.video_paths[start_shard]
            start_cap = cv2.VideoCapture(video_path)
            start_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Jump to start frame

            frames = []
            for frame_idx in range(start_frame, end_frame):
                ret, frame = start_cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            start_cap.release()
            return np.array(frames)  # Shape: (num_frames, H, W, 3)
        else:
            # if the shards are different then we need to get the video frames from different videos
            frames = []
            start_cap = cv2.VideoCapture(self.video_paths[start_shard])
            start_cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for frame_idx in range(start_frame, self.shard_sizes[start_shard]):
                ret, frame = start_cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)
            start_cap.release()

            end_cap = cv2.VideoCapture(self.video_paths[end_shard])
            end_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for frame_idx in range(0, end_frame):
                ret, frame = end_cap.read()
                if not ret:
                    break
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                frame = cv2.resize(frame, (self.image_size, self.image_size))
                frames.append(frame)

            end_cap.release()
            return np.array(frames)

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.valid_start_inds):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset with {self.num_images} images"
            )

        start_idx = self.valid_start_inds[idx]
        end_idx = start_idx + self.video_len
        start_shard_idx, start_frame_idx = self.get_index_info(start_idx)
        end_shard_idx, end_frame_idx = self.get_index_info(end_idx)

        assert self.segment_shards[start_idx] == self.segment_shards[end_idx]
        shard_idx = start_shard_idx

        frames = self.extract_frames_opencv(
            start_shard_idx, end_shard_idx, start_frame_idx, end_frame_idx
        )

        frames = np.moveaxis(frames, 3, 0)
        frames = frames / self._UINT8_MAX_F * 2.0 - 1.0

        frames = frames[:, :: self.down_sample]
        past_frames = frames[:, : self.n_input]
        future_frames = frames[:, self.n_input :]
        assert past_frames.shape[1] == self.n_input
        assert future_frames.shape[1] == self.n_output

        ret = {
            "past_frames": past_frames.astype(np.float32),
            "future_frames": future_frames.astype(np.float32),
            "future_frames_idxs": [
                i
                for i in range(
                    start_idx + self.n_input * self.down_sample,
                    start_idx + self.video_len,
                    self.down_sample,
                )
            ],
        }

        if self.with_actions:
            ret["past_actions"] = self.action_shards[
                start_idx : start_idx + self.n_input
            ].astype(np.float32)
            ret["future_actions"] = self.action_shards[
                start_idx + self.n_input : start_idx + self.n_input + self.n_output
            ].astype(np.float32)

        return ret

    def get_frame_info(self, idx):
        """Helper method to debug frame locations"""
        shard_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side="right")
        frame_idx = idx - self.cumulative_sizes[shard_idx]
        return {
            "global_index": idx,
            "shard_index": shard_idx,
            "frame_index": frame_idx,
            "video_path": self.video_paths[shard_idx],
        }


class FutureFrameDataset(RawVideoDataset):
    def __init__(
        self,
        data_dir,
        cfg,
        n_input=17,
        n_intermediate=59,
        n_output=1,
        stride=1,
        skip_frame=1,
        with_actions=True,
    ):
        TorchDataset.__init__(self)
        # mp.set_start_method('spawn')
        self._UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.image_size = cfg.image_size

        # Load main metadata
        with open(self.data_dir / "metadata.json") as f:
            metadata = json.load(f)
            self.num_shards = metadata["num_shards"]
            self.query = metadata["query"]
            self.hz = metadata["hz"]
            self.num_images = metadata["num_images"]

        # Load shard-specific metadata
        self.shard_sizes = []
        for shard in range(self.num_shards):
            with open(self.data_dir / f"metadata/metadata_{shard}.json") as f:
                shard_metadata = json.load(f)
                self.shard_sizes.append(shard_metadata["shard_num_frames"])

        # Calculate cumulative shard sizes for index mapping
        self.cumulative_sizes = np.cumsum([0] + self.shard_sizes)
        assert (
            self.cumulative_sizes[-1] == self.num_images
        ), "Metadata mismatch in total number of frames"

        # Store video paths instead of keeping captures open
        self.video_paths = [
            str(self.data_dir / f"videos/video_{shard}.mp4")
            for shard in range(self.num_shards)
        ]

        # Store action paths, and open them up if with_actions
        self.action_paths = [
            str(self.data_dir / f"states/states_{shard}.bin")
            for shard in range(self.num_shards)
        ]
        self.with_actions = with_actions
        if self.with_actions:
            self.action_shards = []
            for i, path in enumerate(self.action_paths):
                action_shard = np.memmap(
                    path, dtype=np.float32, mode="r", shape=(self.shard_sizes[i], 25)
                )
                self.action_shards.append(action_shard)
            self.action_shards = np.concatenate(self.action_shards, 0)
            self.action_shards = (
                self.action_shards - np.mean(self.action_shards, 0)
            ) / np.std(self.action_shards, 0)

        # Store action paths, and open them up if with_actions
        self.segment_paths = [
            str(self.data_dir / f"segment_idx/segment_idx_{shard}.bin")
            for shard in range(self.num_shards)
        ]

        segment_shards = []
        for i, path in enumerate(self.segment_paths):
            segment_shard = np.memmap(
                path, dtype=np.int32, mode="r", shape=(self.shard_sizes[i], 1)
            )
            segment_shards.append(segment_shard)
        self.segment_shards = np.concatenate(segment_shards).squeeze(-1)

        # Compute the valid start indexes
        self.n_input, self.n_output, self.stride, self.skip_frame = (
            n_input,
            n_output,
            stride,
            skip_frame,
        )
        self.n_intermediate = n_intermediate
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (
            self.n_input + self.n_intermediate + self.n_output
        )  # * self.skip_frame

        start_indices = np.arange(0, self.num_images - self.video_len, self.stride)
        end_indices = start_indices + self.video_len

        start_segment_ids = self.segment_shards[start_indices]  #
        end_segment_ids = self.segment_shards[end_indices]  #

        self.valid_start_inds = start_indices[start_segment_ids == end_segment_ids]

        # Verify all video files exist and are readable
        for path in self.video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video file: {path}")
            cap.release()

        # Verify all action files exist and are readable
        for path in self.action_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.valid_start_inds):
            raise IndexError(
                f"Index {idx} is out of bounds for dataset with {self.num_images} images"
            )

        start_idx = self.valid_start_inds[idx]
        end_idx = start_idx + self.n_input
        start_shard_idx, start_frame_idx = self.get_index_info(start_idx)
        end_shard_idx, end_frame_idx = self.get_index_info(end_idx)

        assert self.segment_shards[start_idx] == self.segment_shards[end_idx]
        in_frames = self.extract_frames_opencv(
            start_shard_idx, end_shard_idx, start_frame_idx, end_frame_idx
        )
        in_frames = np.moveaxis(in_frames, 3, 0)
        in_frames = in_frames / self._UINT8_MAX_F * 2.0 - 1.0

        ret = {}
        if self.with_actions:
            ret["past_actions"] = self.action_shards[
                start_idx : start_idx
                + self.n_input
                + self.n_intermediate
                + self.n_output
            ].astype(np.float32)

        start_idx = self.valid_start_inds[idx] + self.n_input + self.n_intermediate + 1
        # end_idx = start_idx + self.n_output
        start_shard_idx, start_frame_idx = self.get_index_info(start_idx)
        # end_shard_idx, end_frame_idx = self.get_index_info(end_idx)
        # assert self.segment_shards[start_idx] == self.segment_shards[end_idx], f"Segment mismatch, {self.segment_shards[start_idx]} != {self.segment_shards[end_idx]}"
        out_frame = self.extract_frames_opencv(
            start_shard_idx, start_shard_idx, start_frame_idx, start_frame_idx + 1
        )
        out_frame = np.moveaxis(out_frame, 3, 0)
        out_frame = out_frame / self._UINT8_MAX_F * 2.0 - 1.0

        ret["past_frames"] = (in_frames.astype(np.float32),)
        ret["future_frames"] = out_frame.astype(np.float32)
        return ret


class FutureFrameTestSet(RawVideoDataset):
    def __init__(
        self,
        data_dir,
        cfg,
        n_input=17,
        n_intermediate=60,
        n_output=1,
        stride=1,
        skip_frame=1,
        with_actions=True,
    ):
        TorchDataset.__init__(self)
        self._UINT8_MAX_F = float(torch.iinfo(torch.uint8).max)

        # mp.set_start_method('spawn')

        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.image_size = cfg.image_size
        self.n_videos = 250

        # Store video paths instead of keeping captures open
        self.video_paths = [
            str(self.data_dir / f"videos/video_{shard}.mp4")
            for shard in range(self.n_videos)
        ]

        # Store action paths, and open them up if with_actions
        self.action_paths = [
            str(self.data_dir / f"robot_states/states_{shard}.bin")
            for shard in range(self.n_videos)
        ]
        self.with_actions = with_actions
        if self.with_actions:
            self.action_means = np.array(
                [
                    1.17671257e-03,
                    -6.75296551e-03,
                    -2.30483219e-01,
                    2.75664300e-01,
                    4.71933139e-03,
                    -3.87086980e-02,
                    1.64337859e-01,
                    1.90418631e-01,
                    -2.70267516e-01,
                    -1.33299148e00,
                    -9.99172330e-02,
                    1.23379976e-01,
                    5.50873131e-02,
                    3.17352265e-02,
                    -2.34926492e-01,
                    3.47736299e-01,
                    -1.31072414e00,
                    2.11133912e-01,
                    1.13519236e-01,
                    -4.80192229e-02,
                    3.62483323e-01,
                    1.76630586e-01,
                    3.53241622e-01,
                    7.09157884e-02,
                    4.24408354e-03,
                ]
            )
            self.action_stds = np.array(
                [
                    0.05491458,
                    0.02928038,
                    0.1603748,
                    0.2618789,
                    0.01833516,
                    0.11222003,
                    0.33955425,
                    0.1963473,
                    0.21134913,
                    0.35439548,
                    0.2956042,
                    0.2451176,
                    0.23883468,
                    0.39805043,
                    0.2169766,
                    0.23254885,
                    0.45150614,
                    0.30952364,
                    0.2632832,
                    0.27709484,
                    0.18423752,
                    0.35578346,
                    0.46318832,
                    0.24481903,
                    0.22888942,
                ]
            )
            self.action_shards = []
            for i, path in enumerate(self.action_paths):
                action_shard = np.memmap(
                    path, dtype=np.float32, mode="r", shape=(97, 25)
                )[:77]
                self.action_shards.append(action_shard)

        self.n_input, self.n_output, self.stride, self.skip_frame = (
            n_input,
            n_output,
            stride,
            skip_frame,
        )
        self.n_intermediate = n_intermediate
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (
            self.n_input + self.n_intermediate + self.n_output
        )  # * self.skip_frame

        # Verify all video files exist and are readable
        for path in self.video_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                raise IOError(f"Could not open video file: {path}")
            cap.release()

        # Verify all action files exist and are readable
        for path in self.action_paths:
            if not Path(path).exists():
                raise FileNotFoundError(f"Video file not found: {path}")

    def __len__(self):
        return self.n_videos

    def __getitem__(self, idx):
        in_frames = self.extract_frames_opencv(idx, idx, 0, 17)
        in_frames = np.moveaxis(in_frames, 3, 0)
        in_frames = in_frames / self._UINT8_MAX_F * 2.0 - 1.0

        ret = {}
        if self.with_actions:
            ret["past_actions"] = (
                self.action_shards[idx] - self.action_means
            ) / self.action_stds

        out_frame = np.zeros(
            (3, in_frames.shape[-2], in_frames.shape[-1]), dtype=np.float32
        )

        ret["past_frames"] = (in_frames.astype(np.float32),)
        ret["future_frames"] = out_frame.astype(np.float32)
        ret["sample_id"] = (
            self.video_paths[idx].split("/")[-1].split(".")[0].split("_")[1]
        )
        return ret


def RawVideoDataset_collate_fn(cfg, vae, samples):
    """
    We encode the batches in the collate_fn to speed up the dataloader
    """
    # Extract past and future frames from each sample
    past_frames_list = np.stack([s["past_frames"] for s in samples], 0)
    future_frames_list = np.stack([s["future_frames"] for s in samples], 0)
    past_actions_list = np.stack([s["past_actions"] for s in samples], 0)
    future_actions_list = np.stack([s["future_actions"] for s in samples], 0)

    # Create batch dictionary
    batch = {
        "past_frames": torch.from_numpy(past_frames_list).to(vae._dtype),
        "future_frames": torch.from_numpy(future_frames_list).to(vae._dtype),
        "past_actions": torch.from_numpy(past_actions_list).to(vae._dtype),
        "future_actions": torch.from_numpy(future_actions_list).to(vae._dtype),
    }

    # Encode the batch using the provided encode_batch function
    batch = encode_video_batch(cfg, batch, vae)
    return batch


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


if __name__ == "__main__":
    dataset = RawVideoDataset(
        data_dir="/pub0/qasim/1xgpt/data/data_v2_raw/train_v2.0_raw",
        n_input=8,
        n_output=1,
        stride=4,
        skip_frame=1,
    )
    from cosmos_tokenizer.image_lib import ImageTokenizer
    from cosmos_tokenizer.networks import TokenizerConfigs
    from cosmos_tokenizer.video_lib import CausalVideoTokenizer

    tokenizer_path = "/pub0/qasim/1xgpt/Cosmos-Tokenizer/pretrained_ckpts"
    spatial_compression = 8  # cfg.image_tokenizer.spatial_compression
    temporal_compression = 4  # cfg.image_tokenizer.temporal_compression
    tokenizer_type = "CV"  # cfg.image_tokenizer.tokenizer_type

    tokenizer_config = TokenizerConfigs["CV"].value
    tokenizer_config.update(dict(spatial_compression=spatial_compression))
    if "I" in tokenizer_type:
        model_name = f"Cosmos-Tokenizer-{tokenizer_type}{spatial_compression}x{spatial_compression}"
    else:
        model_name = f"Cosmos-Tokenizer-{tokenizer_type}{temporal_compression}x{spatial_compression}x{spatial_compression}"

    vae = CausalVideoTokenizer(
        checkpoint=Path(tokenizer_path) / model_name / "autoencoder.jit",
        checkpoint_enc=Path(tokenizer_path) / model_name / "encoder.jit",
        checkpoint_dec=Path(tokenizer_path) / model_name / "decoder.jit",
        tokenizer_config=tokenizer_config,
        device=None,
        dtype="bfloat16",
    ).to("cuda")
    input_tensor = torch.randn(1, 3, 9, 512, 512).to("cuda").to(torch.bfloat16)
    (latent,) = vae.encode(input_tensor)
    print(latent.shape)

    print(len(dataset))
    print(dataset[0])
    print(dataset[6050])
    print("Compeleted")
    vae.autoencode()
