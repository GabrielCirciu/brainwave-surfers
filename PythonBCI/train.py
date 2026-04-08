import numpy as np
import pickle
import mne
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import sys

def main():
    try:
        data = np.load('calib_data.npz')
        epochs_data = data['epochs']
        labels = data['labels']
        fs = data['fs']
    except FileNotFoundError:
        print("Could not find calib_data.npz. Please run calibrate.py first.")
        sys.exit(1)

    print(f"Loaded data shape: {epochs_data.shape} (Trials, Channels, Samples)")
    
    # Unicorn channels generic info
    n_channels = epochs_data.shape[1]
    ch_names = [f'EEG {i+1}' for i in range(n_channels)]
    ch_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    # Create MNE EpochsArray
    # Shape of epochs_data: (n_epochs, n_channels, n_times)
    # Important: Convert to Volts if it was stored in microvolts. MNE expects Volts. 
    # Usually Unicorn sends microvolts. So we multiply by 1e-6.
    epochs_data = epochs_data * 1e-6
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)

    # 1. Bandpass filter the data for Motor Imagery (Mu and Beta bands: 8-30 Hz)
    print("Filtering data (8-30 Hz)...")
    epochs.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

    # Extract the filtered data back into a numpy array
    X = epochs.get_data(copy=True)
    y = labels

    # 2. Build the Machine Learning Pipeline
    # CSP extracts spatial patterns that maximize variance for one class vs the other
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    # LDA classifies those variance features
    lda = LinearDiscriminantAnalysis()

    clf = Pipeline([('CSP', csp), ('LDA', lda)])

    # 3. Cross-Validation
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, n_jobs=1)
    
    print("\nEvaluation")
    print(f"Cross-Validation Accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")
    if scores.mean() < 0.6:
        print("Warning: Accuracy is low. You may need more trials or better focus during calibration.")

    # 4. Train final model on all data and save it
    print("\nTraining final model on all data...")
    clf.fit(X, y)
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    
    print("Model saved to model.pkl")
    print("You can now run realtime_predict.py!")

if __name__ == '__main__':
    main()
