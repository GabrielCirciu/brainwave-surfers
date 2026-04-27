import numpy as np
import warnings
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
import os

warnings.filterwarnings("ignore")

def main():
    print("Initializing MOABB Motor Imagery paradigm for BNCI2014_001...")

    # We will create two versions of the dataset:
    # 1. Full (22 EEG channels)
    # 2. Stripped (8 EEG channels matching our Unicorn/hiAmp motor-strip layout)
    
    dataset = BNCI2014_001()
    
    # Define our 8-channel motor-strip list
    motor_channels = ['FC3', 'C3', 'CP3', 'Cz', 'CPz', 'FC4', 'C4', 'CP4']
    
    # 1. Paradigm for full 22 channels
    paradigm_full = MotorImagery(
        n_classes=2, 
        events=['left_hand', 'right_hand'],
    )
    
    # 2. Paradigm for stripped 8 channels
    paradigm_stripped = MotorImagery(
        n_classes=2, 
        events=['left_hand', 'right_hand'],
        channels=motor_channels
    )
    
    base_dir = os.path.join("PythonBCI", "data", "raw", "gold-data-2")
    full_dir = os.path.join(base_dir, "full")
    stripped_dir = os.path.join(base_dir, "stripped")
    
    os.makedirs(full_dir, exist_ok=True)
    os.makedirs(stripped_dir, exist_ok=True)

    print("(This might take a few minutes depending on your internet connection)")

    for subject_id in dataset.subject_list:
        print(f"\nProcessing {dataset.code} Subject {subject_id}...")
        
        # --- Process Full ---
        X_f, y_f, meta_f = paradigm_full.get_data(dataset=dataset, subjects=[subject_id])
        X_f_micro = np.transpose(X_f, (0, 2, 1)) * 1e6
        
        # --- Process Stripped ---
        X_s, y_s, meta_s = paradigm_stripped.get_data(dataset=dataset, subjects=[subject_id])
        X_s_micro = np.transpose(X_s, (0, 2, 1)) * 1e6
        
        # Labels and AUX (common)
        label_map = {'left_hand': 0, 'right_hand': 1}
        y_mapped = np.array([label_map[label] for label in y_f])
        
        epochs, time_steps, _ = X_f_micro.shape
        aux_dummy = np.zeros((epochs, time_steps, 2))
        
        # Save Full
        subj_full_dir = os.path.join(full_dir, f"subject_{subject_id}")
        os.makedirs(subj_full_dir, exist_ok=True)
        for i in range(0, epochs, 10):
            end_idx = i + 10
            batch_eeg = X_f_micro[i:end_idx]
            batch_aux = aux_dummy[i:end_idx]
            batch_labels = y_mapped[i:end_idx]
            fname = f'batch_{i//10}.npz' if len(batch_eeg) == 10 else f'batch_{i//10}_partial.npz'
            np.savez(os.path.join(subj_full_dir, fname), eeg=batch_eeg, aux=batch_aux, labels=batch_labels)
            
        # Save Stripped
        subj_stripped_dir = os.path.join(stripped_dir, f"subject_{subject_id}")
        os.makedirs(subj_stripped_dir, exist_ok=True)
        for i in range(0, epochs, 10):
            end_idx = i + 10
            batch_eeg = X_s_micro[i:end_idx]
            batch_aux = aux_dummy[i:end_idx]
            batch_labels = y_mapped[i:end_idx]
            fname = f'batch_{i//10}.npz' if len(batch_eeg) == 10 else f'batch_{i//10}_partial.npz'
            np.savez(os.path.join(subj_stripped_dir, fname), eeg=batch_eeg, aux=batch_aux, labels=batch_labels)

        print(f"Subject {subject_id} complete. Saved Full ({X_f_micro.shape[2]} channels) and Stripped ({X_s_micro.shape[2]} channels).")

    print(f"\nSUCCESS! Two versions of the dataset saved in {base_dir}")

if __name__ == '__main__':
    main()
