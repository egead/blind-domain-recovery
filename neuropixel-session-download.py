import os
import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# --- 1. CONFIGURATION ---
# Define where you want to save the data on your computer.
# The SDK will create a 'manifest.json' here to manage downloads.
output_dir = './allen_data' 
os.makedirs(output_dir, exist_ok=True)

manifest_path = os.path.join(output_dir, "manifest.json")

# --- 2. INITIALIZE CACHE ---
# This object manages the connection to the Allen Brain Observatory servers.
print(f"Initializing cache at: {manifest_path}")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# --- 3. SELECT A SESSION ---
# We need to find a session that actually contains the "Locally Sparse Noise" stimulus.
# We download the metadata table of all available sessions first.
print("Fetching session table (metadata)...")
sessions = cache.get_session_table()

# Filter for sessions that:
# A) Have the 'locally_sparse_noise_4deg' stimulus
# B) Have data from the Primary Visual Cortex (VISp)
print(sessions.columns)
print(sessions["session_type"])

filtered_sessions = sessions[
    (sessions['session_type'] == 'brain_observatory_1.1') &
    (sessions['ecephys_structure_acronyms'].apply(lambda x: 'VISp' in x))
]

if len(filtered_sessions) == 0:
    raise ValueError("No matching sessions found! Check your filter criteria.")

# We pick the first valid session found
session_id = filtered_sessions.index[3]
print(f"Selected Session ID: {session_id}")

# --- 4. CREATE THE SESSION OBJECT ---
# This command downloads the actual NWB data file (~2GB) if it's not already cached.
print("Downloading session data (this may take a few minutes)...")
session = cache.get_session_data(session_id)

print("\nSUCCESS: 'session' object created.")
print(f"Session Specimen Name: {session.specimen_name}")
print(f"Number of Units (Neurons): {len(session.units)}")

print(session.stimulus_names)
# --- 5. VERIFY DATA AVAILABILITY ---
# Check if the sparse noise stimulus is accessible
try:
    stim_table = session.get_stimulus_table('locally_sparse_noise_4deg')
    print(f"Verified: Found {len(stim_table)} frames of Locally Sparse Noise.")
except KeyError:
    print("Error: The requested stimulus was not found in this session.")