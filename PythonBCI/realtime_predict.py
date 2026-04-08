import numpy as np
from pylsl import StreamInlet, resolve_byprop, StreamInfo, StreamOutlet
import time
import pickle
import mne
import sys

def main():
    try:
        with open('model.pkl', 'rb') as f:
            clf = pickle.load(f)
        print("Loaded model.pkl successfully.")
    except FileNotFoundError:
        print("model.pkl not found! Please run train.py first.")
        sys.exit(1)

    print("Looking for Unicorn EEG stream...")
    streams = []
    while not streams:
        streams = resolve_byprop('type', 'EEG', 1, 3.0)
        if len(streams) == 0:
            print("Still waiting for EEG stream... (Is GTec LSL running or mock_eeg_stream.py running?)")
            
    inlet = StreamInlet(streams[0])
    
    info = inlet.info()
    fs = int(info.nominal_srate())
    if fs <= 0: fs = 250
    print(f"Connected to stream: {info.name()} at {fs} Hz.")

    # Setting up the Outlet for Unity
    # We will send 2 channels: [Probability_Left, Probability_Right]
    # LSL Receiver in Unity will read this
    outlet_info = StreamInfo(name='BCIPredictor3', type='Predictor', channel_count=3, nominal_srate=0, channel_format='float32', source_id='bcipred_v3')
    outlet = StreamOutlet(outlet_info)
    print("Created BCIPredictor3 Outlet. Unity can now connect to this!")

    # Real-time processing parameters
    # The model was trained on 4-second epochs. 
    # For real-time, we maintain a sliding window 
    BUFFER_DUR = 4.0
    BUFFER_SAMPLES = int(fs * BUFFER_DUR)
    EEG_CHANNELS = 8
    
    buffer = np.zeros((EEG_CHANNELS, BUFFER_SAMPLES))
    
    # We create an MNE info object to use their filter function 
    ch_names = [f'EEG {i+1}' for i in range(EEG_CHANNELS)]
    mne_info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg'] * EEG_CHANNELS)

    print("\nStarting Real-time Prediction. Press Ctrl+C to stop.")
    
    update_interval = 0.1 # Update prediction every 0.1s
    last_update = time.time()
    
    inlet.flush()

    try:
        while True:
            # Pull chunk of data instead of single samples
            chunk, timestamps = inlet.pull_chunk(timeout=0.0)
            
            if chunk:
                # Chunk is list of lists, dimensions: (samples, all_channels)
                chunk_arr = np.array(chunk).T # shape: (all_channels, samples)
                chunk_eeg = chunk_arr[:EEG_CHANNELS, :] # shape: (8, samples)
                
                n_new_samples = chunk_eeg.shape[1]
                
                # Update rolling buffer
                if n_new_samples >= BUFFER_SAMPLES:
                    buffer = chunk_eeg[:, -BUFFER_SAMPLES:]
                else:
                    buffer = np.roll(buffer, -n_new_samples, axis=1)
                    buffer[:, -n_new_samples:] = chunk_eeg

            # Perform prediction repeatedly every update_interval
            if time.time() - last_update > update_interval:
                # 1. Prepare data for MNE (shape: (1, channels, samples))
                # Convert to volts
                X_raw = buffer.reshape(1, EEG_CHANNELS, BUFFER_SAMPLES) * 1e-6
                
                # 2. Filter data 8-30 Hz
                # We suppress verbose output because it prints continuously 
                X_epochs = mne.EpochsArray(X_raw, mne_info, verbose=False)
                X_epochs.filter(8., 30., fir_design='firwin', verbose=False)
                
                # Drop the oldest 0.5 seconds of the buffer to match the training data curve!
                X_epochs.crop(tmin=0.5)
                
                X_filtered = X_epochs.get_data(copy=True)
                
                # 3. Predict Probabilities using train model (clf)
                # clf.predict_proba returns probability for class 0 (Left) and class 1 (Right)
                probs = clf.predict_proba(X_filtered)[0] 
                
                # 4. Push to LSL
                outlet.push_sample(probs.tolist())
                
                # Print debug nicely
                # Left -> class 0, Right -> class 1, Rest -> class 2
                print(f"L {probs[0]:.2f} | Rest {probs[2]:.2f} | R {probs[1]:.2f}        ", end='\r')
                
                last_update = time.time()
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()