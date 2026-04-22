import numpy as np
from pylsl import StreamInlet, resolve_byprop
import time
import pickle
import mne
import random
import sys
import pydirectinput
import os

def load_gold_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        sys.exit(1)
        
    data = np.load(file_path)
    eeg = data['eeg'] # Shape: (epochs, time, channels)
    labels = data['labels'] # Shape: (epochs,)
    
    # Separate by label
    left_trials = eeg[labels == 0]
    right_trials = eeg[labels == 1]
    
    print(f"Loaded {len(left_trials)} left trials and {len(right_trials)} right trials from {file_path}")
    return left_trials, right_trials

def main():
    print("Starting Mock End-to-End BCI Pipeline...")

    # Load Model
    model_path = os.path.join("PythonBCI", "models", "model.pkl")
    try:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        print(f"Loaded {model_path} successfully.")
    except FileNotFoundError:
        print(f"Error: {model_path} not found!")
        sys.exit(1)

    # Load Gold Data (defaulting to subject_1, batch_0 based on your current setup)
    gold_data_path = os.path.join('PythonBCI', 'data', 'raw', 'gold-data', 'subject_1', 'batch_0.npz')
    if not os.path.exists(gold_data_path):
        gold_data_path = input(f"Could not find {gold_data_path}.\nEnter path to gold data .npz file: ")
        
    left_trials, right_trials = load_gold_data(gold_data_path)

    # Setting up the UnityMarkers connection to receive triggers
    print("\nLooking for UnityMarkers stream... (Make sure Unity is Playing!)")
    marker_streams = []
    while not marker_streams:
        marker_streams = resolve_byprop('name', 'UnityMarkersObstacles', 1, 3.0)
        if len(marker_streams) == 0:
            print("Still waiting for UnityMarkers stream...")
            time.sleep(1)
            
    # Connect to ALL found UnityMarkers streams (in case there are multiple in the scene)
    marker_inlets = [StreamInlet(info) for info in marker_streams]
    print(f"Connected to {len(marker_inlets)} UnityMarkers stream(s)!")

    fs = 250
    EEG_CHANNELS = 8
    
    # We create an MNE info object to use their filter function 
    ch_names = [f'EEG {i+1}' for i in range(EEG_CHANNELS)]
    mne_info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=['eeg'] * EEG_CHANNELS)

    print("\nWaiting for marker from Unity (OBSTACLE_LEFT or OBSTACLE_RIGHT)...")
    
    try:
        while True:
            # We poll all active marker inlets
            for inlet in marker_inlets:
                try:
                    marker, _ = inlet.pull_sample(timeout=0.0)
                except Exception as e:
                    # If one stream dies, we just ignore it. 
                    # Unity might have destroyed an object but kept the game running.
                    continue

                if marker:
                    cmd = marker[0]
                    
                    target_trials = None
                    target_label_name = ""
                    
                    if cmd == "OBSTACLE_LEFT":
                        print(f"\nReceived marker: {cmd}")
                        print(f"Sending signal: RIGHT")
                        # Obstacle is on the left -> dodge RIGHT
                        target_trials = right_trials
                        target_label_name = "RIGHT"
                    elif cmd == "OBSTACLE_RIGHT":
                        print(f"\nReceived marker: {cmd}")
                        print(f"Sending signal: LEFT")
                        # Obstacle is on the right -> dodge LEFT
                        target_trials = left_trials
                        target_label_name = "LEFT"
                    
                    if target_trials is not None:
                        if len(target_trials) == 0:
                            print(f"Warning: No trials available for {target_label_name} in the loaded gold data!")
                            continue
                            
                        # Pick a random trial matching the required class
                        trial_data = random.choice(target_trials)
                        
                        # Ensure 4 seconds length
                        BUFFER_SAMPLES = int(fs * 4.0)
                        actual_length = trial_data.shape[0]
                        if actual_length >= BUFFER_SAMPLES:
                            trial_data = trial_data[:BUFFER_SAMPLES, :]
                        else:
                            pad_width = BUFFER_SAMPLES - actual_length
                            trial_data = np.pad(trial_data, ((0, pad_width), (0, 0)), mode='edge')
                            
                        # Convert to MNE format: (1, channels, samples) and Volts
                        scale = 1e-6 if np.max(np.abs(trial_data)) > 1e-3 else 1.0
                        X_raw = trial_data.T.reshape(1, EEG_CHANNELS, BUFFER_SAMPLES) * scale
                        
                        # Filter data 8-30 Hz
                        X_epochs = mne.EpochsArray(X_raw, mne_info, verbose=False)
                        X_epochs.filter(8., 30., fir_design='firwin', verbose=False)
                        
                        # Drop the oldest 0.5 seconds
                        X_epochs.crop(tmin=0.5)
                        
                        X_filtered = X_epochs.get_data(copy=True)
                        
                        # Predict Probabilities
                        probs = clf.predict_proba(X_filtered)[0] 
                        predicted_class = np.argmax(probs)
                        
                        # Simulate Keypress (Hold for a few ms so Unity's Update loop reliably catches it)
                        if predicted_class == 0:
                            print(f"PREDICTION: (L: {probs[0]:.2f} | R: {probs[1]:.2f}) -> Pressing 'left' key!\n")
                            pydirectinput.keyDown('left')
                            time.sleep(0.05)
                            pydirectinput.keyUp('left')
                        else:
                            print(f"PREDICTION: (L: {probs[0]:.2f} | R: {probs[1]:.2f}) -> Pressing 'right' key!\n")
                            pydirectinput.keyDown('right')
                            time.sleep(0.05)
                            pydirectinput.keyUp('right')
                            
                        print("Waiting for next marker...\n")
                
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
