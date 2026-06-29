import numpy as np
import os
import pandas as pd
from scipy.stats import poisson
from typing import Tuple, Optional
  
class NeuropixelPreprocessor:
    """
    A unified preprocessor for Allen Neuropixel data.
    Adapted for Static Gratings: Generates 32x32 grayscale diffraction patterns.
    """
    
    # Configuration Constants
    CACHE_DIR = "/Users/onur/Desktop/Code/GroupDenseLayer/allen_data"
    MANIFEST_FILE = "manifest.json"
    TARGET_STRUCTURE = "VISp"
    
    ISI_VIOLATION_THRESHOLD = 0.75 
    CALIBRATION_BLOCK = "spontaneous"
    
    # Adjusted defaults for Static Gratings (often longer duration)
    DEFAULT_DURATION = 0.25 

    def __init__(self, 
                 session_id: int, 
                 window_duration: float = DEFAULT_DURATION, 
                 window_latency: float = 0.05,
                 cache_dir: str = CACHE_DIR):
        
        self.session_id = session_id
        self.window_duration = window_duration
        self.window_latency = window_latency
        
        # 1. Initialize Cache & Load Session
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = self._initialize_cache(cache_dir)
        
        print(f"Loading Session Data for ID: {session_id}...")
        self.session = self.cache.get_session_data(session_id)
        
        # 2. Filter Units
        self.units = self._get_quality_units()
        self.unit_ids = self.units.index.values
        self.num_neurons = len(self.unit_ids)
        print(f"   -> Found {self.num_neurons} quality units in {self.TARGET_STRUCTURE}.")
        
        # 3. Calibrate Noise Models
        self.noise_lambdas = self._calibrate_noise_baselines()
    
    def generate_dataset(self, 
                         stimulus_name: str = 'static_gratings', 
                         fixed_orientation: Optional[float] = None,
                         fixed_frequency: Optional[float] = None, # <--- NEW PARAMETER
                         include_phase: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        
        print(f"Generating Grating Dataset for stimulus: {stimulus_name}...")
        print(f"   -> Latency: {self.window_latency}s | Duration: {self.window_duration}s")
        
        full_table = self.session.get_stimulus_table(stimulus_name)
        
        # 1. Force columns to numeric, turning string "null" into NaN
        full_table['orientation'] = pd.to_numeric(full_table['orientation'], errors='coerce')
        full_table['spatial_frequency'] = pd.to_numeric(full_table['spatial_frequency'], errors='coerce')
        
        if 'phase' in full_table.columns:
            full_table['phase'] = pd.to_numeric(full_table['phase'], errors='coerce')

        # 2. Filter for valid trials
        valid_trials = full_table[
            full_table['orientation'].notnull() & 
            full_table['spatial_frequency'].notnull()
        ].copy()
        
        # 3. Safe Casting
        valid_trials['orientation'] = valid_trials['orientation'].astype(float)
        valid_trials['spatial_frequency'] = valid_trials['spatial_frequency'].astype(float)
        
        # Handle Phase
        if 'phase' in valid_trials.columns and include_phase:
            valid_trials['phase'] = valid_trials['phase'].fillna(0.0).astype(float)
        else:
            valid_trials['phase'] = 0.0
        
        # --- FILTER 1: Fixed Orientation ---
        if fixed_orientation is not None:
            print(f"   -> Filtering for orientation: {fixed_orientation}")
            valid_trials = valid_trials[
                np.isclose(valid_trials['orientation'], fixed_orientation, atol=1.0)
            ]

        # --- FILTER 2: Fixed Frequency (NEW) ---
        if fixed_frequency is not None:
            print(f"   -> Filtering for frequency: {fixed_frequency}")
            # Use isclose for float comparison (e.g. 0.04 vs 0.0399999)
            valid_trials = valid_trials[
                np.isclose(valid_trials['spatial_frequency'], fixed_frequency, atol=1e-5)
            ]
            
        num_trials = len(valid_trials)
        if num_trials == 0:
            print("   -> ❌ No trials found after filtering!")
            # Helpful debug print to show what IS available
            all_freqs = sorted(full_table['spatial_frequency'].dropna().unique())
            all_oris = sorted(full_table['orientation'].dropna().unique())
            print(f"      Available Frequencies: {all_freqs}")
            print(f"      Available Orientations: {all_oris}")
            return None, None, None
            
        print(f"   -> Processing {num_trials} valid trials...")

        # A. Compute Neural Probabilities
        probs = self._compute_probabilities(valid_trials)

        # B. Generate Grating Images
        images = self._generate_grating_images(valid_trials, resolution=32, fov_degrees=40.0)

        # C. Extract Metadata
        param_cols = ['orientation', 'spatial_frequency', 'phase', 'contrast', 'color']
        available_cols = [c for c in param_cols if c in valid_trials.columns]
        params = valid_trials[available_cols].reset_index(drop=True)
        
        # Add Eye Tracking (Optional, kept commented out as in your snippet)
        # params = self._add_eye_tracking_to_params(valid_trials, params)
        return probs, images, params

    # --- Internal Helpers: Computation ---
    def _generate_grating_images(self, table: pd.DataFrame, resolution: int = 32, fov_degrees: float = 40.0) -> np.ndarray:
        """
        Generates 32x32 grayscale sine-wave grating images.
        Formula: sin( 2*pi * f * (x*cos(t) + y*sin(t)) + phi )
        """
        print(f"   -> Synthesizing {resolution}x{resolution} grating images (FOV: {fov_degrees} deg)...")
        
        num_trials = len(table)
        images = np.zeros((num_trials, resolution, resolution), dtype=np.float32)
        
        # 1. Create Grid coordinates (Centered at 0)
        # Range: -FOV/2 to +FOV/2
        limit = fov_degrees / 2.0
        x = np.linspace(-limit, limit, resolution)
        y = np.linspace(-limit, limit, resolution)
        X, Y = np.meshgrid(x, y) # Shape (32, 32)
        
        # 2. Extract parameters as vectors
        # Convert Orientation to Radians
        # Allen: 0 deg = Horizontal, 90 deg = Vertical
        thetas = np.deg2rad(table['orientation'].values)
        freqs = table['spatial_frequency'].values
        # Allen phase is [0, 1) cycle -> Convert to Radians [0, 2pi)
        phases = table['phase'].values * (2 * np.pi) 

        # 3. Generate Images
        for i in range(num_trials):
            theta = thetas[i]
            f = freqs[i]
            phi = phases[i]
            
            # Calculate Wave
            # X_rot = X*cos(theta) + Y*sin(theta)
            wave = np.sin(2 * np.pi * f * (X * np.cos(theta) + Y * np.sin(theta)) + phi)
            images[i] = wave
            
        return images
    
    def _add_eye_tracking_to_params(self, trials_df: pd.DataFrame, params_df: pd.DataFrame) -> pd.DataFrame:
        """Retrieves eye tracking data and averages it per trial window."""
        print("   -> Extracting synchronized eye tracking data...")
        try:
            pupil_data = self.session.get_pupil_data()
        except Exception:
            # Return with NaNs if data missing
            n = len(params_df)
            params_df['pupil_area'] = np.nan
            params_df['eye_x'] = np.nan
            params_df['eye_y'] = np.nan
            return params_df

        # Pre-allocate
        n_trials = len(trials_df)
        areas = np.full(n_trials, np.nan)
        eye_x = np.full(n_trials, np.nan)
        eye_y = np.full(n_trials, np.nan)
        
        starts = trials_df['start_time'].values + self.window_latency
        ends = starts + self.window_duration
        
        pupil_times = pupil_data.index.values
        p_area_vals = pupil_data['pupil_area'].values
        p_x_vals = pupil_data['eye_center_position_x'].values
        p_y_vals = pupil_data['eye_center_position_y'].values

        for i in range(n_trials):
            t_start, t_end = starts[i], ends[i]
            idx_start = np.searchsorted(pupil_times, t_start)
            idx_end = np.searchsorted(pupil_times, t_end)
            
            if idx_end > idx_start:
                areas[i] = np.nanmean(p_area_vals[idx_start:idx_end])
                eye_x[i] = np.nanmean(p_x_vals[idx_start:idx_end])
                eye_y[i] = np.nanmean(p_y_vals[idx_start:idx_end])

        params_df['pupil_area'] = areas
        params_df['eye_x'] = eye_x
        params_df['eye_y'] = eye_y
        return params_df

    # --- Standard Helpers (Unchanged) ---
    def _initialize_cache(self, directory: str):
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
        manifest_path = os.path.join(directory, self.MANIFEST_FILE)
        return EcephysProjectCache.from_warehouse(manifest=manifest_path)

    def _get_quality_units(self) -> pd.DataFrame:
        units = self.session.units
        return units[
            (units['ecephys_structure_acronym'] == self.TARGET_STRUCTURE) & 
            (units['isi_violations'] < self.ISI_VIOLATION_THRESHOLD)
        ]

    def _calibrate_noise_baselines(self) -> np.ndarray:
        print("   -> Calibrating noise models from spontaneous activity...")
        try:
            spontaneous_epochs = self.session.get_stimulus_table(self.CALIBRATION_BLOCK)
        except KeyError:
            return np.ones(self.num_neurons) * 1e-6
            
        baselines = np.zeros(self.num_neurons)
        for i, unit_id in enumerate(self.unit_ids):
            spikes = self.session.spike_times[unit_id]
            total_spikes = 0
            total_duration = 0
            for _, epoch in spontaneous_epochs.iterrows():
                start, end = epoch['start_time'], epoch['stop_time']
                count = np.searchsorted(spikes, end) - np.searchsorted(spikes, start)
                total_spikes += count
                total_duration += (end - start)
            if total_duration > 0:
                avg_rate_hz = total_spikes / total_duration
                lambda_val = avg_rate_hz * self.window_duration
                baselines[i] = max(lambda_val, 1e-6)
            else:
                baselines[i] = 1e-6
        return baselines
    
    def _compute_probabilities(self, trials_df: pd.DataFrame) -> np.ndarray:
        """
        Computes Z-Scores (Gaussian Statistics) instead of raw Probabilities.
        Z = How many standard deviations the signal is above the noise floor.
        """
        from scipy.stats import norm # Import normal distribution
        
        window_starts = trials_df['start_time'].values + self.window_latency
        window_ends = window_starts + self.window_duration
        
        num_trials = len(trials_df)
        z_score_matrix = np.zeros((num_trials, self.num_neurons), dtype=np.float32)

        for i, unit_id in enumerate(self.unit_ids):
            spikes = self.session.spike_times[unit_id]
            
            # 1. Count Spikes
            idx_starts = np.searchsorted(spikes, window_starts)
            idx_ends = np.searchsorted(spikes, window_ends)
            counts = idx_ends - idx_starts
            
            # 2. Get Poisson Probability (CDF)
            # The probability that this count came from noise
            cdf_probs = poisson.cdf(counts, self.noise_lambdas[i])
            
            # 3. Clip to avoid Infinity
            # norm.ppf(1.0) is Inf. We clip to a number effectively representing "Certitude"
            # 1 - 1e-9 maps to a Z-score of approx 6.0 (6-sigma event)
            cdf_probs = np.clip(cdf_probs, a_min=None, a_max=0.999999999)
            
            # 4. Map to Gaussian Z-Score
            # 0.5 -> 0.0
            # 0.99 -> 2.32
            # 0.999999 -> 4.75
            z_score_matrix[:, i] = norm.ppf(cdf_probs)
            
        return z_score_matrix