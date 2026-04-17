import numpy as np
import warnings
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import MotorImagery
import os

warnings.filterwarnings("ignore")

def main():
    print("Initializing MOABB Motor Imagery paradigm...")

    # Schirrmeister2017 (High-Gamma Dataset) contains 128 channels at 500Hz
    # It includes right_hand, left_hand, rest, and feet. We drop feet.
    # MOABB MotorImagery paradigm will automatically handle fetching, 
    # epochs extraction, and filtering the data. We request our 3 specific categories:
    dataset = Schirrmeister2017()
    paradigm = MotorImagery(n_classes=3, events=['left_hand', 'right_hand', 'rest'])
    
    # Fetch data for subject 1
    # X will be shape (epochs, channels, time)
    # y will be string labels
    print(f"Downloading {dataset.code} gold dataset for Subject 1...")
    print("(This might take a minute depending on your internet connection)")
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    print(f"\nDone!")
    print(f"Raw epochs shape: {X.shape}")
    print(f"Labels: {set(y)}")
    
    # calibrate.py maps classes to integers (LEFT=0, RIGHT=1, REST=2)
    # We must format our gold data EXACTLY the same so train.py loads it smoothly.
    # We use 250Hz as it is the sampling rate of the Unicorn headset
    # In case we need to, also have convert voltage magnitudes available
    label_map = {'left_hand': 0, 'right_hand': 1, 'rest': 2}
    y_mapped = np.array([label_map[label] for label in y])
    X_microvolts = X * 1
    fs = 250 
    
    output_file = 'PythonBCI/data/raw/gold_data.npz'
    np.savez(output_file, epochs=X_microvolts, labels=y_mapped, fs=fs)
    
    print(f"\nSUCCESS! Wrote to {output_file}")
    print(f"Total Epochs: {len(y_mapped)}")
    print(f"Data Range (μV): {X_microvolts.min():.2f} to {X_microvolts.max():.2f}")
    print(f"Sampling rate: {fs} Hz")

if __name__ == '__main__':
    main()
