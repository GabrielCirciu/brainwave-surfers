import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import time
from datetime import datetime
import os

BUFFER_DUR = 4.0

def main():
    
    # Get unity markers stream
    print("Looking for UnityMarkers stream... (Make sure Unity is Playing!)")
    marker_streams = []
    while not marker_streams:
        marker_streams = resolve_byprop('name', 'UnityMarkers', 1, 3.0)
        if len(marker_streams) == 0:
            print("Still waiting for UnityMarkers stream... (Is Unity in Play Mode?)")
    marker_inlet = StreamInlet(marker_streams[0])
    print("Found UnityMarkers stream!")
    
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

    # We might have zombie streams that were not closed, so we must use the one that is transmitting data.
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

    # Get sampling rate and channels
    # Channels 0-7  : CF3, C3, CP3, Cz, CPz, CF4, C4, CP4
    # Channels 8-10 : Accelerometer data (XYZ axes)
    # Channels 11-13: Gyroscope data (XYZ axes)
    # Channel  14   : Battery level
    # Channel  15   : Sample counter (used to track dropped samples)
    # Channel  16   : Validation indicator
    sampling_frequency = int(eeg_inlet.info().nominal_srate())
    stream_channels = eeg_inlet.info().channel_count()
    print(f"Connected to streams. fs={sampling_frequency}, channels={stream_channels}")

    # Data storage
    BUFFER_SAMPLES = int(sampling_frequency * BUFFER_DUR)
    epochs_eeg = []
    epochs_aux = []
    epochs_timestamps = []
    labels = []
    
    global_sample_count = 0
    current_trial_class = -1
    is_recording = False
    trial_chunks = []
    trial_timestamps = []

    print("\nStarting calibration...")

    while True:

        # 1. Pull EEG block
        chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1)
        if chunk:
            chunk_arr = np.array(chunk)[:, :stream_channels] # Shape: (samples, channels)
            ts_arr = np.array(timestamps)
            
            global_sample_count += chunk_arr.shape[0]
            
            # If we are in a trial, accumulate the chunks!
            if is_recording:
                trial_chunks.append(chunk_arr)
                trial_timestamps.append(ts_arr)

        # 2. Check for markers
        marker, marker_timestamps = marker_inlet.pull_sample(timeout=0.1)

        if marker:
            cmd = marker[0]
            print(f"Received Marker: {cmd}")
            
            if cmd == "LEFT_START":
                is_recording = True
                current_trial_class = 0
                trial_chunks = [] 
                trial_timestamps = []

            elif cmd == "RIGHT_START":
                is_recording = True
                current_trial_class = 1
                trial_chunks = []
                trial_timestamps = []
            
            elif cmd in ("LEFT_END", "RIGHT_END") and is_recording:

                is_recording = False
                if len(trial_chunks) > 0:

                    # Combine all chunks collected during the trial
                    trial_data = np.concatenate(trial_chunks, axis=0)
                    trial_ts = np.concatenate(trial_timestamps, axis=0)
                    actual_length = trial_data.shape[0]
                    
                    # Ensure uniform lengths for ML models (e.g. exactly 4 seconds)
                    # If it's too long, truncate it. If it's too short, pad with the last edge value
                    if actual_length >= BUFFER_SAMPLES:
                        trial_data = trial_data[:BUFFER_SAMPLES, :]
                        trial_ts = trial_ts[:BUFFER_SAMPLES]
                    else:
                        pad_width = BUFFER_SAMPLES - actual_length
                        trial_data = np.pad(trial_data, ((0, pad_width), (0, 0)), mode='edge')
                        trial_ts = np.pad(trial_ts, (0, pad_width), mode='edge')
                
                    # Split the channels: 0-7 are EEG, 8-16 are AUX (accelerometer, gyro, battery, etc.)
                    eeg_data = trial_data[:, :8]
                    aux_data = trial_data[:, 8:17]
                    
                    # Append timestamp to the last column of aux_data (making it 10 columns)
                    ts_col = trial_ts.reshape(-1, 1)
                    aux_data_with_ts = np.hstack((aux_data, ts_col))
                    
                    epochs_eeg.append(eeg_data)
                    epochs_aux.append(aux_data_with_ts)
                    epochs_timestamps.append(trial_ts)
                    labels.append(current_trial_class)

                    print(f"Epoch saved! Total epochs: {len(epochs_eeg)}")
                    print(f"EEG Shape: {eeg_data.shape} | AUX Shape: {aux_data_with_ts.shape}")
                
                else:
                    print("Warning: Received END marker but no EEG data was collected during the trial.")
            
            elif cmd == "CALIBRATION_END":
                break

    print("\nCalibration Complete!")

    if len(epochs_eeg) > 0:

        eeg_arr = np.array(epochs_eeg)
        aux_arr = np.array(epochs_aux)
        ts_arr = np.array(epochs_timestamps)
        labels_arr = np.array(labels)

        # Check todays date and time and set current version to that.
        current_version = datetime.now().strftime("%Y%m%d_%H%M")
        output_file = 'PythonBCI/data/raw/output_data_' + current_version + '.npz'
        np.savez(output_file, eeg=eeg_arr, aux=aux_arr, labels=labels_arr)
        
        print(f"Saved dataset to {output_file}")
        print(f"Shapes - EEG: {eeg_arr.shape}, Labels: {labels_arr.shape}")
        
    else:
        print("No epochs were recorded.")

if __name__ == '__main__':
    main()
