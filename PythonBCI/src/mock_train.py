import numpy as np
import pickle
import mne
import os
import glob
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import sys

def main():
    print("Starting Mock Training Pipeline...")
    
    # Directory containing gold data batches
    data_dir = os.path.join("PythonBCI", "data", "raw", "gold-data", "subject_1")
    
    if not os.path.exists(data_dir):
        print(f"Error: Could not find directory {data_dir}")
        sys.exit(1)
        
    # Get all .npz batch files
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print(f"Error: No .npz files found in {data_dir}")
        sys.exit(1)
        
    all_eeg = []
    all_labels = []
    
    print(f"Found {len(npz_files)} batch files. Loading data...")
    for file in npz_files:
        data = np.load(file)
        # eeg shape: (epochs, time, channels)
        all_eeg.append(data['eeg'])
        all_labels.append(data['labels'])
        
    # Concatenate all batches
    epochs_data = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total concatenated data shape: {epochs_data.shape} (Trials, Samples, Channels)")
    
    # Transpose to MNE format: (epochs, channels, samples)
    epochs_data = np.transpose(epochs_data, (0, 2, 1))
    print(f"Transposed to MNE shape: {epochs_data.shape} (Trials, Channels, Samples)")
    
    fs = 250
    n_channels = epochs_data.shape[1]
    
    # We create an MNE info object to use their filter function 
    ch_names = [f'EEG {i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Convert to MNE format: (1, channels, samples) and Volts
    # Dynamically scale: if values are > 1e-3, they are likely microVolts, so we * 1e-6
    scale = 1e-6 if np.max(np.abs(epochs_data)) > 1e-3 else 1.0
    epochs_data = epochs_data * scale
    
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)

    print("Filtering data (8-30 Hz)...")
    epochs.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

    print("Cropping first 0.5 seconds to account for reaction time...")
    epochs.crop(tmin=0.5)

    X = epochs.get_data(copy=True)
    y = labels

    print("Training SVM model with CSP...")
    # Overfit the training data to guarantee ~100% accuracy for the mock test
    pipeline = Pipeline([
        ('CSP', CSP(n_components=4, reg=None, log=True, norm_trace=False)),
        ('clf', SVC(C=100.0, kernel='rbf', probability=True))
    ])
    
    # We suppress CSP warnings during fit to keep console clean
    mne.set_log_level('ERROR')
    pipeline.fit(X, y)
    mne.set_log_level('INFO')
    
    # Quick accuracy test on training data
    acc = pipeline.score(X, y)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    
    # Save the model
    models_dir = os.path.join("PythonBCI", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "model.pkl")
    print(f"Saving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("\nSUCCESS! You can now run mock_eeg_stream.py again.")

if __name__ == '__main__':
    main()
