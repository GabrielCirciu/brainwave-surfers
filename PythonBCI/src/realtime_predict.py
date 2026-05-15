import numpy as np
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, resolve_streams
import time
import pickle
import mne
import sys
import pydirectinput
import os

import argparse

def apply_oscar(eeg_data, fs):
    """
    Online Signal Conditioning and Artifact Removal (OSCAR-like).
    eeg_data: (samples, channels)
    """
    X = eeg_data.T
    
    X = mne.filter.filter_data(X.astype(np.float64), fs, 1.0, None, method='iir', verbose=False)
    X = mne.filter.notch_filter(X, fs, [50], method='iir', verbose=False)
    
    cov = np.cov(X)
    evals, evecs = np.linalg.eigh(cov)
    threshold = np.median(evals) * 15.0
    evals_capped = np.minimum(evals, threshold)
    whitening_mat = evecs @ np.diag(np.sqrt(evals_capped / (evals + 1e-9))) @ evecs.T
    X_clean = whitening_mat @ X
    
    return X_clean.T

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.path.join("PythonBCI", "models", "model.pkl"))
    args = parser.parse_args()
    
    model_path = args.model_path
    try:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        print(f"Loaded {model_path} successfully.")
    except FileNotFoundError:
        print(f"Error: {model_path} not found!")
        sys.exit(1)


    print("Looking for EEG stream...")
    eeg_streams = []
    while not eeg_streams:
        streams = resolve_streams()
        
        print("\nFound the following streams on the network")
        for i, s in enumerate(streams):
            print(f"Stream {i}: {s.name()} | Type: {s.type()} | Channels: {s.channel_count()}")
        
        print("\nFiltering for Data streams...")
        eeg_streams = [s for s in streams if s.type() == 'Data']
        for s in eeg_streams:
            print(f"Found Data stream '{s.name()}'...")
        
        if len(eeg_streams) == 0:
            print("Still waiting for the Data stream... (Make sure LSL is set to 'send all signals in one stream')")
            time.sleep(1)

    target_stream = None
    eeg_inlet = None

    print("\nTesting streams for active data...")
    for s in reversed(eeg_streams):
        print(f"Testing stream '{s.name()}'...")
        inlet = StreamInlet(s)
        chunk, test_timestamp = inlet.pull_chunk(timeout=0.1, max_samples=250)
        
        if chunk:
            print("Success! Data is flowing.")
            target_stream = s
            eeg_inlet = inlet
            break
        else:
            print("No data. This is a zombie stream from earlier.")
            
    if target_stream is None:
        print("\nERROR: All located 8-channel streams are dead. Please restart the EEG, LSL stream, and then restart this process!")
        return
        
    stream_channels = eeg_inlet.info().channel_count()
    fs = int(eeg_inlet.info().nominal_srate())
    
    print(f"\nConnected to active stream! channels={stream_channels}, fs={fs}Hz")

    print("\nLooking for UnityMarkers stream... (Make sure Unity is Playing!)")
    marker_streams = []
    while not marker_streams:
        marker_streams = resolve_byprop('name', 'UnityMarkersObstacles', 1, 3.0)
        if len(marker_streams) == 0:
            print("Still waiting for UnityMarkers stream...")
            time.sleep(1)
            
    marker_inlets = [StreamInlet(info) for info in marker_streams]
    print(f"Connected to {len(marker_inlets)} UnityMarkers stream(s)!")
    
    BUFFER_DUR = 4.0
    BUFFER_SAMPLES = int(fs * BUFFER_DUR)
    EEG_CHANNELS = 8
    
    if EEG_CHANNELS == 8:
        ch_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    else:
        ch_names = [f'EEG {i+1}' for i in range(EEG_CHANNELS)]
    mne_info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg'] * EEG_CHANNELS)

    print("Waiting for marker from Unity to start 4-second recording...")
    
    is_collecting = False
    trial_chunks = []
    
    try:
        while True:
            for inlet in marker_inlets:
                try:
                    marker, _ = inlet.pull_sample(timeout=0.0)
                except Exception as e:
                    continue

                if marker:
                    cmd = marker[0]
                    break

            if marker:
                cmd = marker[0]
                if cmd == "OBSTACLE_LEFT" or cmd == "OBSTACLE_RIGHT":
                    print(f"Received marker: {cmd} - Starting 4-second collection!")
                    is_collecting = True
                    trial_chunks = []
                    trial_timestamps = []
                    eeg_inlet.flush()
                    collection_start_time = time.time()

            if is_collecting:
                chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1)
                if chunk:
                    chunk_arr = np.array(chunk)[:, :stream_channels]
                    ts_arr = np.array(timestamps)
                    trial_chunks.append(chunk_arr)
                    trial_timestamps.append(ts_arr)
                    
                if time.time() - collection_start_time >= 4.0:
                    is_collecting = False
                    
                    if len(trial_chunks) > 0:
                        trial_data = np.concatenate(trial_chunks, axis=0)
                        trial_ts = np.concatenate(trial_timestamps, axis=0)
                        actual_length = trial_data.shape[0]
                        
                        if actual_length < BUFFER_SAMPLES * 0.9:
                            print(f"WARNING: Only collected {actual_length} samples ({actual_length/fs:.2f}s) instead of {BUFFER_SAMPLES} samples (4.0s). LSL stream might be dropping packets!")
                            
                        if actual_length >= BUFFER_SAMPLES:
                            trial_data = trial_data[:BUFFER_SAMPLES, :]
                            trial_ts = trial_ts[:BUFFER_SAMPLES]
                        else:
                            pad_width = BUFFER_SAMPLES - actual_length
                            trial_data = np.pad(trial_data, ((0, pad_width), (0, 0)), mode='edge')
                            trial_ts = np.pad(trial_ts, (0, pad_width), mode='edge')
                    
                        eeg_data = trial_data[:, :8].astype(np.float64)
                        aux_data = trial_data[:, 8:17]
                        
                        eeg_data = apply_oscar(eeg_data, fs)
                        
                        trial_std = np.std(eeg_data)
                        if trial_std > 0:
                            eeg_data = eeg_data / trial_std
                        
                        ts_col = trial_ts.reshape(-1, 1)
                        aux_data_with_ts = np.hstack((aux_data, ts_col))
                        
                        print(f"4 seconds elapsed! EEG Shape: {eeg_data.shape} | AUX: {aux_data_with_ts.shape}. Running Inference...")
                        
                        X_raw_uV = eeg_data.T.reshape(1, EEG_CHANNELS, BUFFER_SAMPLES)
                        
                        for freq in np.arange(50, fs / 2, 50):
                            X_raw_uV = mne.filter.notch_filter(
                                X_raw_uV.astype(np.float64), 
                                fs, 
                                freq, 
                                method='iir',
                                verbose=False
                            )
                            
                        X_raw = X_raw_uV * 1e-6
                        X_epochs = mne.EpochsArray(X_raw, mne_info, verbose=False)
                        
                        X_epochs.filter(8., 30., fir_design='firwin', verbose=False)
                        
                        try:
                            X_epochs.set_montage("standard_1020")
                            X_epochs = mne.preprocessing.compute_current_source_density(X_epochs)
                        except Exception as e:
                            print(f"CSD failed: {e}, falling back to CAR")
                            X_epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)
                        
                        X_epochs.crop(tmin=0.5)
                        
                        X_filtered = X_epochs.get_data(copy=True)
                        
                        print(f"DEBUG: actual_length={actual_length}, X_raw var={np.var(X_raw):.2e}, X_filtered var={np.var(X_filtered):.2e}")
                        
                        probs = clf.predict_proba(X_filtered)[0] 
                        predicted_class = np.argmax(probs)
                        
                        if predicted_class == 0:
                            print(f"PREDICTION: Left  (L: {probs[0]:.2f} | R: {probs[1]:.2f}) -> Pressing 'left' key!\n")
                            pydirectinput.keyDown('left')
                            time.sleep(0.05)
                            pydirectinput.keyUp('left')
                        else:
                            print(f"PREDICTION: Right (L: {probs[0]:.2f} | R: {probs[1]:.2f}) -> Pressing 'right' key!\n")
                            pydirectinput.keyDown('right')
                            time.sleep(0.05)
                            pydirectinput.keyUp('right')
                            
                        print("Waiting for next marker...\n")
                    else:
                        print("Warning: 4 seconds passed but no EEG data was collected.")
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()