import numpy as np
import mne
from mne.datasets import eegbci
import os

def main():
    print("Downloading PhysioNet EEG Motor Imagery 'Gold Standard' Dataset...")
    subject = 1
    # Run 4, 8, and 12 are specifically Left/Right fist imagery runs
    runs = [4, 8, 12]
    
    # Download data (this will download ~5MB of EEG recordings)
    raw_fnames = eegbci.load_data(subject, runs)
    
    print("\nReading downloaded files...")
    raws = []
    for f in raw_fnames:
        raw = mne.io.read_raw_edf(f, preload=True)
        mne.datasets.eegbci.standardize(raw)  # standardize channel names
        raws.append(raw)
        
    raw = mne.concatenate_raws(raws)
    
    print("\nExtracting Left/Right Motor Imagery trials...")
    # T1 = Left fist imagery, T2 = Right fist imagery, T0 = Rest
    events, event_dict = mne.events_from_annotations(raw, event_id={'T1': 0, 'T2': 1, 'T0': 2})
    
    # The Unicorn Headset has 8 channels. PhysioNet has 64. 
    # We will pick 8 standard 10-20 channels close to the motor cortex to mimic your hardware setup:
    ch_names_to_pick = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'P3', 'P4', 'Oz']
    raw.pick(ch_names_to_pick)
    
    # Resample to 250Hz to match your game's preferred Unicorn Hz limit
    raw.resample(250)
    fs = 250
    
    # Epoching: Slice out the 4 seconds after the user is told to imagine moving
    epochs = mne.Epochs(raw, events, event_id={'Left': 0, 'Right': 1, 'Rest': 2}, tmin=0, tmax=4.0, baseline=None, preload=True)
    
    # `raw.get_data()` outputs absolute Volts. Your script expects MicroVolts from hardware.
    # We multiply by 1e6 to simulate the raw integers dumped by the Unicorn hardware.
    epochs_data = epochs.get_data(copy=True) * 1e6 
    labels = epochs.events[:, -1]
    
    # Make sure we have exactly 4 seconds of data (250 Hz * 4 = 1000 samples)
    epochs_data = epochs_data[:, :, :1000]
    
    print(f"\nFinal Extracted 'Gold' Dataset Shape: {epochs_data.shape}")
    print(f"Total Trials: {epochs_data.shape[0]}")
    
    # Save it identically to how calibrate.py would save it!
    np.savez('calib_data.npz', epochs=epochs_data, labels=labels, fs=fs)
    print("Saved successfully to calib_data.npz! It forcefully overwrote any existing data.")
    print("You can now instantly run `train.py`, followed by `mock_eeg_stream.py` to watch your ship steer perfectly!")

if __name__ == "__main__":
    main()
