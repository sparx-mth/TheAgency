"""Feeding the trainer: labels from the dataset, pixels (or tokens) from disk.

The dataset built by :mod:`.build_dataset` stores pointers, not pictures, so
this is where a sample is actually assembled. Two modes, same output contract
apart from the vision fields:

``live``    reads the RGB and depth frames and preprocesses them exactly as
            NavDP's agent does. Correct in every configuration, and the only
            option once the depth trunk is being trained.
``cached``  reads pre-computed frozen-ViT patch tokens instead. With both DINOv2
            trunks frozen their output for a frame never changes, so ~99 % of
            the forward pass can be done once and reused -- see
            :mod:`.cache_features`.

**The memory is real.** NavDP conditions on eight frames, and the earlier
pixel-goal dataset filled all eight with copies of the current one, which threw
away every cue about motion the architecture was built to use. Here the eight
slots are eight actual frames, left-zero-padded at the start of a recording
exactly as ``NavDP_Agent`` pads its queue.

Torch, but only ``torch.utils.data``; the heavy imports stay in the trainer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import (
    load_recording,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.preprocess import (
    IMAGE_SIZE, memory_indices, preprocess_depth, preprocess_rgb,
)

PATCH_TOKENS = 256
TOKEN_DIM = 384


@dataclass(frozen=True)
class DatasetConfig:
    """How samples are assembled.

    Attributes:
        memory_size: NavDP's queue length. Do not change (8 is baked into the
            Q-Former's positional table, which is sized ``(8 + 1) * 256``).
        memory_stride: Frames skipped between memory slots. 1 = consecutive
            (0.8 s of history at 10 Hz).
        color_order: Channel order fed to the encoder; see :mod:`.preprocess`.
        cache_dir: Feature cache to read tokens from. ``None`` = live pixels.
    """

    memory_size: int = 8
    memory_stride: int = 1
    color_order: str = "bgr"
    cache_dir: Optional[str] = None


class WorldGoalDataset(Dataset):
    """One split of a built dataset."""

    def __init__(self, root, split: str, config: Optional[DatasetConfig] = None) -> None:
        self.root = Path(root).expanduser()
        self.split = split
        self.config = config or DatasetConfig()
        self.index = json.loads((self.root / "index.json").read_text())

        archive = np.load(self.root / split / "samples.npz")
        self.samples = {key: archive[key] for key in archive.files}
        self.count = int(self.samples["frame"].size)
        self.recordings = self.index["recordings"]

        self._readers: Dict[int, object] = {}
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}
        self.cache_dir = Path(self.config.cache_dir).expanduser() if self.config.cache_dir else None
        self.cache_meta = (json.loads((self.cache_dir / "meta.json").read_text())
                           if self.cache_dir else None)
        if self.cache_meta and self.cache_meta.get("color_order") != self.config.color_order:
            raise ValueError(
                f"feature cache was built with color_order="
                f"{self.cache_meta.get('color_order')!r} but this run asks for "
                f"{self.config.color_order!r}; rebuild the cache or match it")

    def __len__(self) -> int:
        return self.count

    # ------------------------------------------------------------------ readers
    def recording(self, index: int):
        """The reader for one recording, opened lazily once per worker."""
        if index not in self._readers:
            self._readers[index] = load_recording(self.recordings[index]["path"])
        return self._readers[index]


    def _tokens(self, index: int) -> Dict[str, np.ndarray]:
        """Lazily memory-map one recording's cached tokens."""
        if index not in self._cache:
            base = self.cache_dir / str(index)
            self._cache[index] = {
                "rgb": np.load(base / "rgb.npy", mmap_mode="r"),
                "depth": np.load(base / "depth.npy", mmap_mode="r"),
                "row": np.load(base / "row_of_frame.npy"),
            }
        return self._cache[index]

    # ------------------------------------------------------------------- items
    def _vision_live(self, recording, frames, valid) -> Dict[str, torch.Tensor]:
        cfg = self.config
        images = np.zeros((cfg.memory_size, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        for slot, (frame, ok) in enumerate(zip(frames, valid)):
            if not ok:
                continue                                   # left zero-pad, as NavDP does
            rgb = recording.rgb(frame)
            if rgb is None:
                raise FileNotFoundError(
                    f"no RGB frame {frame:06d} in {recording.root}; NavDP needs colour")
            images[slot] = preprocess_rgb(rgb, cfg.color_order)
        depth = preprocess_depth(recording.depth(frames[-1]))
        return {"images": torch.from_numpy(images),
                "depth": torch.from_numpy(depth)}

    def _vision_cached(self, index: int, frames, valid) -> Dict[str, torch.Tensor]:
        cfg = self.config
        store = self._tokens(index)
        rows = store["row"]
        tokens = np.zeros((cfg.memory_size, PATCH_TOKENS, TOKEN_DIM), dtype=np.float16)
        for slot, (frame, ok) in enumerate(zip(frames, valid)):
            if not ok:
                continue
            row = int(rows[frame])
            if row < 0:
                raise KeyError(
                    f"frame {frame} of recording {index} is not in the feature cache; "
                    f"rebuild it with the same --frame-stride and --memory-stride")
            tokens[slot] = store["rgb"][row]
        depth_row = int(rows[frames[-1]])
        # np.array(copy=True): a slice of a read-only memmap makes torch warn
        # about non-writable storage on every single sample.
        return {"rgb_tokens": torch.from_numpy(tokens),
                "depth_tokens": torch.from_numpy(
                    np.array(store["depth"][depth_row], dtype=np.float16))}

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        cfg = self.config
        sample = self.samples
        recording_index = int(sample["recording"][item])
        frame = int(sample["frame"][item])
        frames, valid = memory_indices(frame, cfg.memory_size, cfg.memory_stride)

        if self.cache_dir is None:
            vision = self._vision_live(self.recording(recording_index), frames, valid)
        else:
            vision = self._vision_cached(recording_index, frames, valid)

        goal = np.zeros(3, dtype=np.float32)
        goal[:2] = sample["goal_token"][item]
        out = {
            "goal": torch.from_numpy(goal),
            "action": torch.from_numpy(sample["action"][item].astype(np.float32)),
            "pose": torch.from_numpy(sample["pose"][item].astype(np.float32)),
            "goal_world": torch.from_numpy(sample["goal_world"][item].astype(np.float32)),
            "scene": torch.tensor(int(sample["scene"][item]), dtype=torch.long),
            "turn_deg": torch.tensor(float(sample["turn_deg"][item])),
            "goal_kind": torch.tensor(int(sample["goal_kind"][item]), dtype=torch.long),
        }
        out.update(vision)
        return out

    # -------------------------------------------------------------- inspection
    def frames_needed(self) -> Dict[int, np.ndarray]:
        """Every ``(recording -> frame indices)`` this split will actually read.

        Includes the memory window of each sample, which is what the feature
        cache has to cover: a sample at frame 300 reads frames 293..300.
        """
        cfg = self.config
        needed: Dict[int, set] = {}
        for recording_index, frame in zip(self.samples["recording"].tolist(),
                                          self.samples["frame"].tolist()):
            frames, valid = memory_indices(frame, cfg.memory_size, cfg.memory_stride)
            bucket = needed.setdefault(int(recording_index), set())
            bucket.update(f for f, ok in zip(frames, valid) if ok)
        return {key: np.array(sorted(value), dtype=np.int32)
                for key, value in sorted(needed.items())}

    def describe(self) -> str:
        """One-line summary for the run log."""
        turn = self.samples["turn_deg"]
        return (f"{self.split}: {self.count} samples, "
                f"{int(np.unique(self.samples['recording']).size)} recordings, "
                f"turn>=40deg {int((turn >= 40).sum())} "
                f"({100.0 * float((turn >= 40).mean()):.0f}%), "
                f"arrivals {int(self.samples['reaches'].sum())}")


def merged_frames_needed(datasets: List[WorldGoalDataset]) -> Dict[int, np.ndarray]:
    """Union of :meth:`WorldGoalDataset.frames_needed` over several splits."""
    merged: Dict[int, set] = {}
    for dataset in datasets:
        for recording_index, frames in dataset.frames_needed().items():
            merged.setdefault(recording_index, set()).update(frames.tolist())
    return {key: np.array(sorted(value), dtype=np.int32)
            for key, value in sorted(merged.items())}
