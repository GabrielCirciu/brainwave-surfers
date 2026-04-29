import numpy as np
import glob
import os
import mne

session_dir = r"PythonBCI\data\raw\gaci-26-04-29-09-06"
session_files = sorted(glob.glob(os.path.join(session_dir, "batch_*.npz")))
eeg_list = []
for f in session_files:
    if "merged" in f: continue
    eeg_list.append(np.load(f)['eeg'])
sess_eeg = np.concatenate(eeg_list, axis=0)

# sess_eeg is (100, 1000, 8)
# Compute std of the RAW data after removing DC offset
raw_std = np.std(sess_eeg - np.mean(sess_eeg, axis=1, keepdims=True), axis=1)
print(f"RAW AC signal std per channel (mean across trials): \n{np.mean(raw_std, axis=0)}")

# Apply 8-30 Hz bandpass filter to see actual EEG band
# mne filter expects (trials, channels, time)
epochs_data = np.transpose(sess_eeg, (0, 2, 1)).astype(np.float64)
# Filter
epochs_data = mne.filter.filter_data(epochs_data, sfreq=250.0, l_freq=8.0, h_freq=30.0, verbose=False)

band_std = np.std(epochs_data, axis=2)
print(f"8-30Hz signal std per channel (mean across trials): \n{np.mean(band_std, axis=0)}")
