import numpy as np
import warnings
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import MotorImagery
import os

warnings.filterwarnings("ignore")

def main():
    print("Initializing MOABB Motor Imagery paradigm...")
    # MOABB MotorImagery paradigm will automatically handle fetching, 
    # epoching, and filtering the data. We request our 3 specific categories:
    paradigm = MotorImagery(n_classes=3, events=['left_hand', 'right_hand', 'rest'])
    
    # Schirrmeister2017 (High-Gamma Dataset) contains 128 channels at 250Hz!
    # It includes right_hand, left_hand, rest, and feet. We drop feet.
    dataset = Schirrmeister2017()
    
    print(f"Downloading/Extracting {dataset.code} gold dataset for Subject 1...")
    print("(This might take a minute depending on your internet connection)")
    
    # Fetch data for subject 1
    # X will be shape (epochs, channels, time)
    # y will be string labels
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=[1])
    
    print(f"\nExtraction complete!")
    print(f"Raw epochs shape: {X.shape}")
    print(f"Discovered labels: {set(y)}")
    
    # calibrate.py maps classes to integers (LEFT=0, RIGHT=1, REST=2)
    # We must format our gold data EXACTLY the same so train.py loads it smoothly.
    label_map = {
        'left_hand': 0,
        'right_hand': 1,
        'rest': 2
    }
    
    # Convert string labels to numerical labels
    y_mapped = np.array([label_map[label] for label in y])
    
    # IMPORTANT DATA SCALING:
    # Unicorn LSL natively outputs in Microvolts (μV). So your calib_data.npz stores 
    # large numbers (e.g., 15.4, 250.1). train.py expects this and explicitly multiples 
    # by 1e-6 to convert back to Volts for internal MNE math.
    # HOWEVER, MOABB provides data exactly in Volts (e.g., 0.0000154). 
    # To prevent train.py from shrinking the data twice and creating micro-scale zeros, 
    # we simulate the Unicorn hardware by converting MOABB Volts up to Microvolts first!
    X_microvolts = X * 1e6
    
    # We use 250Hz, which matches both the Unicorn and the Schirrmeister dataset
    fs = 250 
    
    output_file = 'calib_data.npz'
    np.savez(output_file, epochs=X_microvolts, labels=y_mapped, fs=fs)
    
    print("\n------------------------------")
    print(f"SUCCESS! Wrote {output_file}")
    print(f"- Total Epochs: {len(y_mapped)}")
    print(f"- Data Range (μV): {X_microvolts.min():.2f} to {X_microvolts.max():.2f}")
    print(f"- Sampling rate: {fs} Hz")
    print("------------------------------")
    print("\nYou can now run 'python train.py' and your pipeline will train on real BCI Motor Imagery data!")

if __name__ == '__main__':
    main()
