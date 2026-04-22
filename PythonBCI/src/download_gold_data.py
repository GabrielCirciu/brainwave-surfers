import numpy as np
import warnings
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import MotorImagery
import os

warnings.filterwarnings("ignore")

def main():
    print("Initializing MOABB Motor Imagery paradigm...")

    # Schirrmeister2017 (High-Gamma Dataset) contains 128 channels at 500Hz
    # It includes right_hand, left_hand, rest, and feet. We drop rest and feet.
    # We restrict to our specific 8 Unicorn channels (using standard 10-20 names FC3 and FC4 instead of CF3/CF4)
    # and resample down to 250Hz to match the Unicorn headset.
    dataset = Schirrmeister2017()
    paradigm = MotorImagery(
        n_classes=2, 
        events=['left_hand', 'right_hand'],
        channels=['FC3', 'C3', 'CP3', 'Cz', 'CPz', 'FC4', 'C4', 'CP4'],
        resample=250.0
    )
    
    base_output_dir = os.path.join("PythonBCI", "data", "raw", "gold-data")
    os.makedirs(base_output_dir, exist_ok=True)

    print("(This might take a minute depending on your internet connection)")

    # For testing, let's only download for the first 2 subjects
    dataset.subject_list = dataset.subject_list[:2]

    for subject_id in dataset.subject_list:
        print(f"\nDownloading {dataset.code} gold dataset for Subject {subject_id}...")
        
        # X will be shape (epochs, channels, time)
        # y will be string labels
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        
        # calibrate.py outputs (epochs, time, channels). We transpose MOABB data to match.
        X_transposed = np.transpose(X, (0, 2, 1))
        
        # calibrate.py maps classes to integers (LEFT=0, RIGHT=1)
        label_map = {'left_hand': 0, 'right_hand': 1}
        y_mapped = np.array([label_map[label] for label in y])
        X_microvolts = X_transposed * 1
        
        # Create dummy AUX data to match the (epochs, time, 10) shape from calibrate.py
        epochs, time_steps, channels = X_microvolts.shape
        aux_dummy = np.zeros((epochs, time_steps, 10))
        
        subject_dir = os.path.join(base_output_dir, f"subject_{subject_id}")
        os.makedirs(subject_dir, exist_ok=True)
        
        batch_count = 0
        for i in range(0, epochs, 10):
            end_idx = i + 10
            batch_eeg = X_microvolts[i:end_idx]
            batch_aux = aux_dummy[i:end_idx]
            batch_labels = y_mapped[i:end_idx]
            
            if len(batch_eeg) == 10:
                output_file = os.path.join(subject_dir, f'batch_{batch_count}.npz')
                batch_count += 1
            else:
                output_file = os.path.join(subject_dir, f'batch_{batch_count}_partial.npz')
                
            np.savez(output_file, eeg=batch_eeg, aux=batch_aux, labels=batch_labels)
            
        print(f"Subject {subject_id} complete. Saved {epochs} epochs across {(epochs // 10) + (1 if epochs % 10 != 0 else 0)} batches.")

    print(f"\nSUCCESS! All data downloaded and formatted into {base_output_dir}")

if __name__ == '__main__':
    main()
