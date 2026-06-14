import tensorflow as tf
import numpy as np
import time
import sys
import os
import json
import multiprocessing
import contextlib
import argparse
import platform
from shutil import rmtree
from os import makedirs
from os.path import exists, join
from git import Repo
import random

# Import your newly refactored TICA script
from train import run_tica_experiment

# ==============================================================================
# Platform Detection & Configuration
# ==============================================================================
IS_APPLE_SILICON = (platform.system() == 'Darwin') and ('arm' in platform.machine().lower())

def initialize_hardware_acceleration(gpu_indices):
    """
    Configures GPU visibility based on the architecture (NVIDIA vs Apple Silicon).
    """
    gpus = tf.config.list_physical_devices("GPU")
    
    if IS_APPLE_SILICON:
        if not gpus:
            print("WARNING: No GPU found. Ensure 'tensorflow-metal' is installed for Mac acceleration.")
            return
        
        # On Apple Silicon, Metal manages memory dynamically.
        # We don't restrict visible devices or set memory growth.
        print(f"Process {os.getpid()} using Apple Silicon/Metal GPU.")
        
    else:
        # Standard NVIDIA Workstation Logic
        if not gpus:
            raise RuntimeError("No GPUs found.")

        gpu_indices = [int(i) for i in gpu_indices]
        selected_gpus = [gpus[idx] for idx in gpu_indices]
        
        try:
            tf.config.set_visible_devices(selected_gpus, "GPU")
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Process {os.getpid()} using NVIDIA GPUs: {gpu_indices}")
        except RuntimeError as e:
            print(f"RuntimeError in initialize_hardware_acceleration: {e}")

# ==============================================================================
# Utility Functions
# ==============================================================================
try:
    repo = Repo(".")
    commit_hash = repo.git.rev_parse("HEAD")
except:
    commit_hash = "unknown"

def get_exp_dir(exp_name):
    return join("experiments", exp_name)

def get_model_weights_saving_dir(exp_name):
    return join("experiments", exp_name, "epochs")

def save_experiment_specs(exp_name, exp_specs, base_folder="experiments"):
    exp_folder = os.path.join(base_folder, exp_name)
    exp_file_path = os.path.join(exp_folder, "specs.json")
    makedirs(exp_folder, exist_ok=True)
    
    with open(exp_file_path, "w") as f:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.float64):
                return float(obj)
            return obj
        exp_specs["commit_hash"] = commit_hash
        json.dump(exp_specs, f, default=convert, indent=4)

def distribute_experiments(experiments):
    tasks = []
    for exp_name, exp_specs in experiments.items():        
        gpus = exp_specs.get("gpus", [0])
        tasks.append((exp_name, exp_specs, gpus))
    return tasks

def set_seed(seed:int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    tf.random.set_seed(seed)

# ==============================================================================
# Main Training Loop
# ==============================================================================
def train_model(exp_name, exp_specs, gpu_indices):
    # Setup logging
    log_folder = os.path.join("experiments", exp_name)
    makedirs(log_folder, exist_ok=True)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.path.join(log_folder, "training_log.txt"), "w", buffering=1)
    sys.stderr = open(os.path.join(log_folder, "training_error_log.txt"), "w", buffering=1)

    try:
        set_seed(exp_specs.get("seed", 0))
        initialize_hardware_acceleration(gpu_indices)
        save_experiment_specs(exp_name, exp_specs)

        # Execute TICA Training
        run_tica_experiment(
            exp_name=exp_name,
            exp_specs=exp_specs,
            save_dir=get_model_weights_saving_dir(exp_name)
        )

        print(f"Experiment {exp_name} finished successfully.")
        
    except Exception as e:
        print(f"Error in experiment {exp_name}: {e}")
        raise e
    finally:
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr

# ==============================================================================
# Execution Entry Point
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments from a JSON config.")
    parser.add_argument("--config", type=str, default="all_experiments.json", help="Path to the experiments JSON file")
    args = parser.parse_args()

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"Loading experiments from: {args.config}")
    with open(args.config, "r") as f:
        experiments = json.load(f)

    # Reset experiment folders
    for exp_name in experiments.keys():
        exp_dir = get_exp_dir(exp_name)
        if exists(exp_dir):
            try:
                rmtree(exp_dir)
            except OSError as e:
                print(f"Error removing directory {exp_dir}: {e}")
            
        makedirs(exp_dir, exist_ok=True)
        makedirs(get_model_weights_saving_dir(exp_name), exist_ok=True)

    tasks = distribute_experiments(experiments)
    
    # Handle Process Count based on Architecture
    if IS_APPLE_SILICON:
        print("Apple Silicon detected. Enforcing serial execution to prevent memory contention.")
        num_processes = 1
    else:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
    
    print(f"Starting {len(tasks)} experiments with {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(train_model, tasks)
        
    print("All experiments have been completed.")