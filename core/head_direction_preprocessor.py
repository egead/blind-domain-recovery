import numpy as np
import os
import pandas as pd
from typing import Tuple, Optional

import pynapple as nap


class HeadDirectionPreprocessor:
    CACHE_DIR = "th1_data"
    TARGET_STRUCTURE = "adn"

    DEFAULT_BIN_SIZE = 0.025
    DEFAULT_SMOOTHING_STD = 0.05
    DEFAULT_MIN_RATE = 0.5
    DEFAULT_HD_INFO_THRESHOLD = 0.2
    DEFAULT_GRID_RESOLUTION = 32

    def __init__(self,
                 session_path: str,
                 bin_size: float = DEFAULT_BIN_SIZE,
                 smoothing_std: float = DEFAULT_SMOOTHING_STD,
                 target_structure: str = TARGET_STRUCTURE,
                 cache_dir: str = CACHE_DIR):

        self.session_path = session_path
        self.bin_size = bin_size
        self.smoothing_std = smoothing_std
        self.target_structure = target_structure
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        print(f"Loading th-1 session from: {session_path}...")
        self.data = nap.load_session(session_path, "neurosuite")

        self.spikes = self.data.spikes
        self.angle = self.data.position["ry"]
        self.wake_epoch = self.data.epochs["wake"]

        self.units = self._select_units()
        self.num_neurons = len(self.units)
        print(f"   -> Using {self.num_neurons} units from {self.target_structure}.")

    def generate_dataset(self,
                         fixed_orientation: Optional[float] = None,
                         fixed_frequency: Optional[float] = None,
                         grid_resolution: int = DEFAULT_GRID_RESOLUTION
                         ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

        print("Generating head-direction dataset...")
        print(f"   -> Bin size: {self.bin_size}s | Smoothing std: {self.smoothing_std}s")

        angle_wake = self.angle.restrict(self.wake_epoch)

        count = self.units.count(self.bin_size, self.wake_epoch)
        rates = count / self.bin_size

        if self.smoothing_std is not None and self.smoothing_std > 0:
            rates = rates.smooth(self.smoothing_std)

        rate_centers = rates.index.values
        head_angle = np.interp(rate_centers,
                               angle_wake.index.values,
                               np.unwrap(angle_wake.values))
        head_angle = np.mod(head_angle, 2.0 * np.pi)

        spks = np.asarray(rates.values, dtype=np.float32)

        valid = np.all(np.isfinite(spks), axis=1) & np.isfinite(head_angle)
        spks = spks[valid]
        head_angle = head_angle[valid]

        spks = spks[:, self._hd_info_order]

        num_samples = spks.shape[0]
        if num_samples == 0:
            print("   -> ❌ No valid samples produced!")
            return None, None, None

        print(f"   -> Produced {num_samples} samples of dimension {self.num_neurons}.")

        mov = self._render_circular_bumps(head_angle, grid_resolution)

        params = pd.DataFrame({
            "orientation": np.rad2deg(head_angle).astype(np.float32),
            "spatial_frequency": np.zeros(num_samples, dtype=np.float32),
            "phase": np.zeros(num_samples, dtype=np.float32),
        })

        return spks, mov, params

    def _select_units(self):
        units = self.spikes
        if self.target_structure is not None and "location" in units.metadata_columns:
            units = units[units.location == self.target_structure]

        units = units.getby_threshold("rate", self.DEFAULT_MIN_RATE)

        angle_wake = self.angle.restrict(self.wake_epoch)
        tuning = nap.compute_1d_tuning_curves(units,
                                              angle_wake,
                                              120,
                                              ep=self.wake_epoch,
                                              minmax=(0.0, 2.0 * np.pi))

        info = self._head_direction_information(tuning)
        keep = info.values > self.DEFAULT_HD_INFO_THRESHOLD
        if not np.any(keep):
            print("   -> ⚠ No units passed the HD-information threshold; keeping all.")
            keep = np.ones(len(info), dtype=bool)

        units = units[keep]
        self._hd_info_order = np.argsort(info.values[keep])[::-1]
        return units

    def _head_direction_information(self, tuning):
        eps = 1e-12
        rates = tuning.values
        p = 1.0 / rates.shape[0]
        mean_rate = np.sum(rates * p, axis=0) + eps
        ratio = rates / mean_rate
        info = np.sum(p * ratio * np.log2(ratio + eps), axis=0)
        return pd.Series(info, index=tuning.columns)

    def _render_circular_bumps(self, head_angle: np.ndarray, resolution: int,
                               concentration: float = 4.0) -> np.ndarray:
        grid = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
        delta = head_angle[:, None] - grid[None, :]
        bumps = np.exp(concentration * np.cos(delta))
        bumps /= np.max(bumps, axis=1, keepdims=True)
        return bumps.astype(np.float32)
