import tensorflow as tf
import numpy as np
import sys
import os
import json
import multiprocessing
import platform
import argparse
import random
import traceback
from shutil import rmtree
from os import makedirs
from os.path import exists, join
from git import Repo

# Assuming these imports exist in your project structure
from core.train_utils import *

# ==============================================================================
# Platform Detection
# ==============================================================================

# Detect Apple Silicon (M1/M2/M3/etc.)
IS_APPLE_SILICON = (platform.system() == 'Darwin') and ('arm' in platform.machine().lower())

# ==============================================================================
# Utility Functions
# ==============================================================================

# Safely grab the git commit hash
try:
    repo = Repo(".")
    commit_hash = repo.git.rev_parse("HEAD")
except:
    commit_hash = "unknown"

def get_exp_dir(exp_name):
    return join("experiments", exp_name)

def get_model_weights_saving_dir(exp_name):
    return join("experiments", exp_name, "epochs")

def set_seed(seed: int = 42):
    """Ensures deterministic behavior across numpy, python, and tensorflow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def save_experiment_specs(exp_name, exp_specs, base_folder="experiments"):
    exp_folder = os.path.join(base_folder, exp_name)
    exp_file_path = os.path.join(exp_folder, "specs.json")
    
    makedirs(exp_folder, exist_ok=True)
    
    with open(exp_file_path, "w") as f:
        def convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.float64): return float(obj)
            return obj
        exp_specs["commit_hash"] = commit_hash
        json.dump(exp_specs, f, default=convert, indent=4)

def distribute_experiments(experiments):
    tasks = []
    for exp_name, exp_specs in experiments.items():        
        # Default to GPU 0 if missing from JSON
        gpus = exp_specs.get("gpus", [0]) 
        tasks.append((exp_name, exp_specs, gpus))
    return tasks

# ==============================================================================
# Hardware Sandboxing
# ==============================================================================

def initialize_hardware_acceleration(gpu_indices):
    """
    Configures GPU visibility based on the architecture.
    Crucial for multiprocessing: Sandboxes the process to ONLY see its assigned GPU.
    """
    gpus = tf.config.list_physical_devices("GPU")
    
    if IS_APPLE_SILICON:
        if not gpus:
            print("WARNING: No GPU found. Ensure 'tensorflow-metal' is installed.")
            return
        
        print(f"Process {os.getpid()} using Apple Metal GPU.")
        
    else:
        # Standard NVIDIA Workstation Logic
        if not gpus:
            raise RuntimeError("No GPUs found.")

        # Convert indices to integer
        gpu_indices = [int(i) for i in gpu_indices]
        selected_gpus = [gpus[idx] for idx in gpu_indices]
        
        try:
            # Hide all other GPUs from this specific worker process
            tf.config.set_visible_devices(selected_gpus, "GPU")
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Process {os.getpid()} sandboxed to NVIDIA GPUs: {gpu_indices}")
        except RuntimeError as e:
            print(f"RuntimeError in set_visible_gpus: {e}")

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
        # 1. Apply reproducibility seeds
        seed = exp_specs.get("seed", 0)
        set_seed(seed)
        
        # 2. Restrict process to assigned GPU BEFORE building the model
        initialize_hardware_acceleration(gpu_indices)
        save_experiment_specs(exp_name, exp_specs)

        # 3. Build model (automatically placed on the single visible GPU)
        model = create_model(**exp_specs["model_params"])

        # 4. Generate Data
        data_generator = make_data_generator(seed=seed, **exp_specs["data_generator_params"])
        batches = []
        data_generator.reset_batch_counter()
        
        for b in range(exp_specs["num_training_batches"]):
            batches.append(data_generator.sample_batch_of_data())
            print(f"Processed batch:{b}")
        
        if not batches:
            raise ValueError("No batches were generated.")
            
        dataset_np = np.concatenate(batches, axis=0)

        # 5. Start Training.
        train(
            model,
            dataset_np=dataset_np,
            saving_dir=get_model_weights_saving_dir(exp_name),
            exp_specs=exp_specs
        )

        print(f"Experiment {exp_name} completed.")
        
    except Exception as e:
        print(f"Error in experiment {exp_name}: {e}")
        traceback.print_exc()
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
    parser = argparse.ArgumentParser(description="Run GPU experiments from a JSON config.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="all_experiments.json", 
        help="Path to the experiments JSON file"
    )
    args = parser.parse_args()

    # Safely set start method for Multiprocessing and TF
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print(f"Loading experiments from: {args.config}")
    with open(args.config, "r") as f:
        experiments = json.load(f)

    # Clean previous runs
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
    
    # Handle Process Count Based on Architecture
    if IS_APPLE_SILICON:
        print("Apple Silicon detected. Enforcing serial execution (1 process) to prevent unified memory contention.")
        num_processes = 1
    else:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
    
    print(f"Starting {len(tasks)} experiments with {num_processes} processes...")
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(train_model, tasks)
        
    print("All experiments have been completed.")