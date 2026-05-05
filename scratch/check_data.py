import numpy as np
import os

data_path = r'e:\GitHub\brainwave-surfers\PythonBCI\data\raw\gaci-oscar-26-05-04-18-56\batch_0.npz'
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found.")
    # Try the other session ID from the terminal output
    data_path = r'e:\GitHub\brainwave-surfers\PythonBCI\data\raw\gaci-oscar-26-05-04-19-11\batch_0.npz'

if os.path.exists(data_path):
    data = np.load(data_path)
    eeg = data['eeg']
    labels = data['labels']

    print(f"File: {data_path}")
    print(f"EEG Shape: {eeg.shape}")
    print(f"Labels: {labels}")
    print(f"Mean: {np.mean(eeg):.6f}")
    print(f"Std: {np.std(eeg):.6f}")
    print(f"Max: {np.max(np.abs(eeg)):.6f}")
    print(f"First 10 samples (Ch 0, Trial 0): {eeg[0, :10, 0]}")
else:
    print("Could not find the NPZ file.")
