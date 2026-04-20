import numpy as np
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet, resolve_streams
import time
import pickle
import mne
import sys
import pyautogui

def main():
    try:
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)
        print("Loaded model.pkl successfully.")
    except FileNotFoundError:
        print("model.pkl not found!")
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

    # Setting up the UnityMarkers connection to receive triggers
    print("Looking for UnityMarkers stream... (Make sure Unity is Playing!)")
    marker_streams = []
    while not marker_streams:
        marker_streams = resolve_byprop('name', 'UnityMarkers', 1, 3.0)
        if len(marker_streams) == 0:
            print("Still waiting for UnityMarkers stream...")
    marker_inlet = StreamInlet(marker_streams[0])
    print("Found UnityMarkers stream!")
    
    # Real-time processing parameters
    # The model was trained on 4-second epochs. 
    BUFFER_DUR = 4.0
    BUFFER_SAMPLES = int(fs * BUFFER_DUR)
    EEG_CHANNELS = 8
    
    # We create an MNE info object to use their filter function 
    ch_names = [f'EEG {i+1}' for i in range(EEG_CHANNELS)]
    mne_info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg'] * EEG_CHANNELS)

    print("Waiting for marker from Unity to start 4-second recording...")
    
    is_collecting = False
    trial_chunks = []
    
    try:
        while True:
            # 1. Listen for marker from Unity
            marker, _ = marker_inlet.pull_sample(timeout=0.0)
            if marker:
                cmd = marker[0]
                if cmd == "PREDICT_START":
                    print(f"Received marker: {cmd} - Starting 4-second collection!")
                    is_collecting = True
                    trial_chunks = []
                    trial_timestamps = []
                    # Clear out any old EEG data sitting in the queue
                    eeg_inlet.flush()
                    collection_start_time = time.time()

            # 2. Collect Data if triggered
            if is_collecting:
                chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1)
                if chunk:
                    # Match calibrate.py exact fetching shape: (samples, stream_channels)
                    chunk_arr = np.array(chunk)[:, :stream_channels]
                    ts_arr = np.array(timestamps)
                    trial_chunks.append(chunk_arr)
                    trial_timestamps.append(ts_arr)
                    
                # Collect exactly 4 seconds based on actual elapsed time
                if time.time() - collection_start_time >= 4.0:
                    is_collecting = False
                    
                    if len(trial_chunks) > 0:
                        # Combine all chunks collected during the 4 seconds
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
                        
                        # 3. Preparation & Inference
                        print(f"4 seconds elapsed! EEG Shape: {eeg_data.shape} | AUX: {aux_data_with_ts.shape}. Running Inference...")
                        
                        # Convert to MNE format: (1, channels, samples) and Volts
                        X_raw = eeg_data.T.reshape(1, EEG_CHANNELS, BUFFER_SAMPLES) * 1e-6
                        
                        # 4. Filter data 8-30 Hz
                        X_epochs = mne.EpochsArray(X_raw, mne_info, verbose=False)
                        X_epochs.filter(8., 30., fir_design='firwin', verbose=False)
                        
                        # Drop the oldest 0.5 seconds of the buffer to match training curve
                        X_epochs.crop(tmin=0.5)
                        
                        X_filtered = X_epochs.get_data(copy=True)
                        
                        # 5. Predict Probabilities using trained model (clf)
                        probs = clf.predict_proba(X_filtered)[0] 
                        predicted_class = np.argmax(probs)
                        
                        # 6. Simulate Keypress
                        if predicted_class == 0:
                            print(f"PREDICTION: Left  (L: {probs[0]:.2f} | R: {probs[1]:.2f}) -> Pressing 'left' key!\n")
                            pyautogui.press('left')
                        else:
                            print(f"PREDICTION: Right (L: {probs[0]:.2f} | R: {probs[1]:.2f}) -> Pressing 'right' key!\n")
                            pyautogui.press('right')
                            
                        print("Waiting for next marker...\n")
                    else:
                        print("Warning: 4 seconds passed but no EEG data was collected.")
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()