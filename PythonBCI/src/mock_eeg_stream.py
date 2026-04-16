import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
import os
import sys

def main():
    print("Starting Mock EEG Stream (Unicorn 8-Channel 250Hz)...")
    
    # Setup LSL Outlet to mimic the GTec headset
    info = StreamInfo(name='UnicornMock', type='EEG', channel_count=8, nominal_srate=250, channel_format='float32', source_id='mock_eeg_1')
    outlet = StreamOutlet(info)
    print("LSL Outlet created! The network now thinks a headset is connected.")
    
    # Load real calibration data to playback
    playback_data = None
    labels = None
    
    if os.path.exists('calib_data.npz'):
        data = np.load('calib_data.npz')
        epochs = data['epochs'] # Shape: (trials, channels, samples)
        labels = data['labels'] # Shape: (trials,)
        
        # Flatten out all trials into one huge continuous array: (channels, total_samples)
        playback_data = np.concatenate([epochs[i] for i in range(epochs.shape[0])], axis=1)
        print(f"Loaded {epochs.shape[0]} trials from calib_data.npz for playback.")
    else:
        print("\nWARNING: No calib_data.npz found in this folder.")
        print("Streaming random noise instead, which won't trigger the game properly.")
        print("Please run a calibration session first if you want valid data!\n")
        
    fs = 250
    sleep_duration = 1.0 / fs
    
    print("Streaming data live... Press Ctrl+C to stop.")
    
    try:
        sample_idx = 0
        trial_idx = 0
        
        while True:
            start_time = time.time()
            
            if playback_data is not None:
                # Grab the exact sample for the current microsecond
                current_sample = playback_data[:, sample_idx].tolist()
                
                # If we hit a multiple of the buffer size (1 trial duration), print what we are currently playing
                if sample_idx % (fs * 4) == 0:
                    class_names = ["Left", "Right", "Hold"]
                    current_class = class_names[labels[trial_idx]]
                    print(f"Currently broadcasting a '{current_class}' trial snippet...")
                    trial_idx = (trial_idx + 1) % len(labels)

                sample_idx = (sample_idx + 1) % playback_data.shape[1]
            else:
                # Fallback to random noise (simulate 8 channels of microvolts)
                current_sample = (np.random.randn(8) * 10).tolist()
                
            outlet.push_sample(current_sample)
            
            # Precise sleep to mimic exactly 250 records per second
            elapsed = time.time() - start_time
            if elapsed < sleep_duration:
                time.sleep(sleep_duration - elapsed)
                
    except KeyboardInterrupt:
        print("\nStopped Mock Stream.")

if __name__ == '__main__':
    main()
