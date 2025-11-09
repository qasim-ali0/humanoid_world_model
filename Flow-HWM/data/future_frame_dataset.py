import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from data.future_video_dataset import FutureVideoDataset


class FutureFrameDataset(FutureVideoDataset):
    """
    Dataset class for training a model to predict one future frame
    e.g. predict the 77th frame after the past frames
    """

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


class FutureFrameTestDataset(FutureVideoDataset):
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
