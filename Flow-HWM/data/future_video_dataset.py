import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


class FutureVideoDataset(TorchDataset):
    """
    Dataset class for training a model to predict one future video sequences.
    e.g. predict 8 future frames from 9 past video frames
    """

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
