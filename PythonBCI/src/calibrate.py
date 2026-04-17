import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import time
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

    # Get EEG stream
    # print("Looking for EEG stream...")
    # eeg_streams = []
    # while not eeg_streams:
    #     streams = resolve_streams(3.0)
    #     valid_names = ['UN-2024.08.41', 'Unicorn', 'UnicornRecorderLSLStream', 'UnicornMock']
    #     eeg_streams = [s for s in streams if s.name() in valid_names or s.name().startswith('UN-2024.08.41') or s.type() == 'Data']
    #     if len(eeg_streams) == 0:
    #         print("Still waiting for EEG stream... (Is GTec LSL running or mock stream active?)")
    # eeg_inlet = StreamInlet(eeg_streams[-1]) # Grab the latest activated stream to avoid zombies
    
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
    epochs_data = []
    labels = []
    raw_stream = []
    global_sample_count = 0
    current_trial_class = -1
    is_recording = False
    trial_chunks = []

    print("\nStarting calibration...")

    while True:

        # 1. Pull EEG block
        chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1)
        if chunk:
            chunk_arr = np.array(chunk).T[:stream_channels, :]
            # chunk_arr = np.array(chunk)
            raw_stream.append(chunk_arr)
            global_sample_count += chunk_arr.shape[1]
            
            # If we are in a trial, accumulate the chunks!
            if is_recording:
                trial_chunks.append(chunk_arr)

        # 2. Check for markers
        marker, marker_timestamps = marker_inlet.pull_sample(timeout=0.1)

        if marker:
            cmd = marker[0]
            print(f"Received Marker: {cmd}")
            
            if cmd == "LEFT_START":
                is_recording = True
                current_trial_class = 0
                trial_chunks = [] # Start a fresh new chunk

            elif cmd == "RIGHT_START":
                is_recording = True
                current_trial_class = 1
                trial_chunks = []

            elif cmd == "REST_START":
                is_recording = True
                current_trial_class = 2
                trial_chunks = []
            
            elif cmd in ("LEFT_END", "RIGHT_END", "REST_END") and is_recording:

                is_recording = False
                if len(trial_chunks) > 0:

                    # Combine all chunks collected during the trial
                    trial_data = np.concatenate(trial_chunks, axis=1)
                    actual_length = trial_data.shape[1]
                    
                    # Ensure uniform lengths for ML models (e.g. exactly 4 seconds)
                    # If it's too long, truncate it. If it's too short, pad with the last edge value
                    if actual_length >= BUFFER_SAMPLES:
                        trial_data = trial_data[:, :BUFFER_SAMPLES]
                    else:
                        pad_width = BUFFER_SAMPLES - actual_length
                        trial_data = np.pad(trial_data, ((0, 0), (0, pad_width)), mode='edge')
                
                    epochs_data.append(trial_data)
                    labels.append(current_trial_class)

                    print(f"Epoch saved! Total epochs: {len(epochs_data)}")
                    print(f"Epoch Shape: {trial_data.shape} (Adjusted from raw length {actual_length})")
                
                else:
                    print("Warning: Received END marker but no EEG data was collected during the trial.")
            
            elif cmd == "CALIBRATION_END":
                break

    print("\nCalibration Complete!")

    if len(epochs_data) > 0:

        epochs_arr = np.array(epochs_data)
        labels_arr = np.array(labels)

        with open('PythonBCI/data/config/output_data_version.txt', 'r') as f:
            current_version = f.read()

        output_file = 'PythonBCI/data/raw/output_data_' + current_version + '.npz'

        with open('PythonBCI/data/config/output_data_version.txt', 'w') as f:
            f.write(str(int(current_version) + 1))

        np.savez(output_file, epochs=epochs_arr, labels=labels_arr, fs=sampling_frequency)
        print(f"Saved dataset to {output_file}, shape: {epochs_arr.shape}")
        
    else:
        print("No epochs were recorded.")

if __name__ == '__main__':
    main()
