import numpy as np
import pickle
import mne
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, GridSearchCV
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
    epochs_data = epochs_data * 1e-6
    epochs = mne.EpochsArray(epochs_data, info, verbose=False)

    # 1. Bandpass filter the data for Motor Imagery (Mu and Beta bands: 8-30 Hz)
    print("Filtering data (8-30 Hz)...")
    epochs.filter(8., 30., fir_design='firwin', skip_by_annotation='edge', verbose=False)

    # Drop the first 0.5 seconds to account for human reaction time to the visual cue!
    epochs.crop(tmin=0.5)

    # Extract the filtered data back into a numpy array
    X = epochs.get_data(copy=True)
    y = labels

    # 2. Build the AutoML parameter grid
    param_grid = [
        # 1. Support Vector Machine Combinations
        {
            'CSP__n_components': [2, 4],
            'CSP__reg': [None, 'ledoit_wolf'],
            'clf': [SVC(probability=True)],
            'clf__kernel': ['linear', 'rbf'],
            'clf__C': [0.1, 1.0, 10.0]
        },
        # 2. Linear Discriminant Analysis Combinations
        {
            'CSP__n_components': [2, 4],
            'CSP__reg': [None, 'ledoit_wolf'],
            'clf': [LinearDiscriminantAnalysis()],
            'clf__solver': ['lsqr', 'eigen'],
            'clf__shrinkage': ['auto']
        },
        # 3. Random Forest Combinations
        {
            'CSP__n_components': [2, 4],
            'CSP__reg': [None, 'ledoit_wolf'],
            'clf': [RandomForestClassifier(random_state=42)],
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [None, 3]
        }
    ]

    base_pipeline = Pipeline([
        ('CSP', CSP(log=True, norm_trace=False)),
        ('clf', SVC()) # Placeholder, will be swapped by GridSearchCV
    ])

    cv = ShuffleSplit(10, test_size=0.2, random_state=42)

    # 3. Run the Grid Search Tournament
    print("\nStarting automated model selection tournament (This will take a few seconds)...")
    grid_search = GridSearchCV(base_pipeline, param_grid, cv=cv, n_jobs=1, verbose=1)
    
    # We suppress CSP warnings during search to keep the console clean
    mne.set_log_level('ERROR')
    grid_search.fit(X, y)
    mne.set_log_level('INFO')

    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    
    print("\nTournament Results:")
    print(f"WINNING MODEL: {grid_search.best_params_['clf']}")
    print(f"WINNING HYPERPARAMS: {grid_search.best_params_}")
    print(f"Cross-Validation Accuracy: {best_score:.2f}")

    if best_score < 0.6:
        print("\nWarning: Accuracy is low. You may need more trials or better focus during calibration.")

    # 4. Save the absolute best model
    print("\nDeploying #1 Ranked Model to model.pkl...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    print("Model saved to model.pkl")
    print("You can now run realtime_predict.py!")

if __name__ == '__main__':
    main()
