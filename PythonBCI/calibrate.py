import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import time
import os

BUFFER_DUR = 4.0

def main():
    #print("Looking for UnityMarkers stream... (Make sure Unity is Playing!)")
    #marker_streams = []
    #while not marker_streams:
    #    marker_streams = resolve_byprop('name', 'UnityMarkers', 1, 3.0)
    #    if len(marker_streams) == 0:
    #        print("Still waiting for UnityMarkers stream... (Is Unity in Play Mode?)")
            
    marker_inlet = StreamInlet(marker_streams[0])
    print("Found UnityMarkers stream!")

    print("Looking for EEG stream...")
    
    eeg_streams = []
    while not eeg_streams:
        streams = resolve_streams(3.0)
        valid_names = ['UN-2024.08.41', 'Unicorn', 'UnicornRecorderLSLStream', 'UnicornMock']
        eeg_streams = [s for s in streams if s.name() in valid_names or s.name().startswith('UN-2024.08.41') or s.type() == 'EEG']
        
        if len(eeg_streams) == 0:
            print("Still waiting for EEG stream... (Is GTec LSL running or mock stream active?)")
            
    eeg_inlet = StreamInlet(eeg_streams[-1]) # Grab the latest activated stream to avoid zombies
    
    fs = int(eeg_inlet.info().nominal_srate())
    if fs <= 0: fs = 250
    stream_channels = eeg_inlet.info().channel_count()
    print(f"Connected to streams. fs={fs}, channels={stream_channels}")

    BUFFER_SAMPLES = int(fs * BUFFER_DUR)
    buffer = np.zeros((stream_channels, BUFFER_SAMPLES))

    epochs_data = []
    labels = []
    
    # Storage for continuous EDA stream
    raw_stream = []
    eda_markers = []
    global_sample_count = 0

    print("\nStarting calibration...")
    
    current_trial_class = -1
    is_recording = False

    while True:
        # 1. Pull EEG block and update buffer
        chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.001)
        if chunk:
            chunk_arr = np.array(chunk).T[:stream_channels, :]
            raw_stream.append(chunk_arr)
            global_sample_count += chunk_arr.shape[1]
            
            n_new = chunk_arr.shape[1]
            if n_new >= BUFFER_SAMPLES:
                buffer = chunk_arr[:, -BUFFER_SAMPLES:]
            else:
                buffer = np.roll(buffer, -n_new, axis=1)
                buffer[:, -n_new:] = chunk_arr

        # 2. Check for markers
        marker, mrk_ts = marker_inlet.pull_sample(timeout=0.001)
        if marker:
            cmd = marker[0]
            print(f"Received Marker: {cmd}")
            eda_markers.append((cmd, global_sample_count))
            
            if cmd == "LEFT_START":
                is_recording = True
                current_trial_class = 0
            elif cmd == "RIGHT_START":
                is_recording = True
                current_trial_class = 1
            elif cmd == "REST_START":
                is_recording = True
                current_trial_class = 2
            elif cmd in ("LEFT_END", "RIGHT_END", "REST_END") and is_recording:
                # The trial just ended, the buffer currently holds the last 4 seconds
                trial_data = buffer.copy()
                epochs_data.append(trial_data)
                labels.append(current_trial_class)
                is_recording = False
                print(f"Epoch saved! Total epochs: {len(epochs_data)}")
                print(trial_data)
            elif cmd == "CALIBRATION_END":
                break

    print("\nCalibration Complete!")
    if len(epochs_data) > 0:
        epochs_arr = np.array(epochs_data)
        labels_arr = np.array(labels)
        output_file = 'calib_data.npz'
        np.savez(output_file, epochs=epochs_arr, labels=labels_arr, fs=fs)
        print(f"Saved dataset to {output_file}, shape: {epochs_arr.shape}")
        
        # Save continuous data for EDA
        raw_stream_arr = np.concatenate(raw_stream, axis=1)
        eda_file = 'calib_continuous_eda.npz'
        # Convert markers to something saveable (list of strings/ints)
        np.savez(eda_file, raw=raw_stream_arr, markers=np.array(eda_markers, dtype=str), fs=fs)
        print(f"Saved RAW continuous stream for EDA to {eda_file}\n - Stream Shape: {raw_stream_arr.shape}\n - Total Markers: {len(eda_markers)}")
    else:
        print("No epochs were recorded.")

if __name__ == '__main__':
    main()
