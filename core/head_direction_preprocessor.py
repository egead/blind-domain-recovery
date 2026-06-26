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
        self.angle = self._load_head_angle(session_path)
        self.wake_epoch = self._load_wake_epoch(session_path)

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
            std_bins = max(1, int(round(self.smoothing_std / self.bin_size)))
            window_bins = max(std_bins, int(6 * std_bins) | 1)
            rates = rates.smooth(std_bins, window_bins)

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

    def _session_basename(self, session_path: str) -> str:
        return os.path.join(session_path, os.path.basename(os.path.normpath(session_path)))

    def _whl_rate(self, session_path: str) -> float:
        xml_path = self._session_basename(session_path) + ".xml"
        sampling_rate = 20000.0
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(xml_path).getroot()
            node = root.find(".//acquisitionSystem/samplingRate")
            if node is None:
                node = root.find(".//samplingRate")
            if node is not None:
                sampling_rate = float(node.text)
        except Exception as e:
            print(f"   -> ⚠ Could not read samplingRate from XML ({e}); assuming 20000 Hz.")
        return sampling_rate / 512.0

    def _load_head_angle(self, session_path: str):
        whl_path = self._session_basename(session_path) + ".whl"
        print(f"   -> Reading head angle from {whl_path}")
        whl = np.loadtxt(whl_path, dtype=np.float32)
        if whl.ndim != 2 or whl.shape[1] < 4:
            raise ValueError(f"Expected a 4-column .whl file, got shape {whl.shape}.")

        x1, y1, x2, y2 = whl[:, 0], whl[:, 1], whl[:, 2], whl[:, 3]

        tracked = (x1 >= 0) & (y1 >= 0) & (x2 >= 0) & (y2 >= 0)

        angle = np.mod(np.arctan2(y2 - y1, x2 - x1), 2.0 * np.pi)
        angle[~tracked] = np.nan

        rate = self._whl_rate(session_path)
        t = np.arange(whl.shape[0], dtype=np.float64) / rate

        valid = np.isfinite(angle)
        return nap.Tsd(t=t[valid], d=angle[valid].astype(np.float64))

    def _load_wake_epoch(self, session_path: str):
        states_path = self._session_basename(session_path) + ".states.Wake"
        print(f"   -> Reading wake epoch from {states_path}")
        intervals = np.loadtxt(states_path, dtype=np.float64, ndmin=2)
        if intervals.shape[1] < 2:
            raise ValueError(f"Expected start/end columns in {states_path}, got shape {intervals.shape}.")
        return nap.IntervalSet(start=intervals[:, 0], end=intervals[:, 1])

    def _select_units(self):
        units = self.spikes
        if self.target_structure is not None and "location" in units.metadata_columns:
            loc_vals = units.get_info("location")
            loc = np.array([str(v).lower() for v in loc_vals])
            keep_locs = np.array(units.keys())[loc == self.target_structure.lower()]
            units = units[keep_locs]

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

        keep_keys = np.asarray(info.index)[keep]
        units = units[keep_keys]
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
