import numpy as np
import os
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from PythonBCI.src.train import load_and_preprocess

class LogVarExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, picks=None):
        self.picks = picks
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        if self.picks is not None:
            X = X[:, self.picks, :]
        # var along time axis
        var = np.var(X, axis=2)
        # add small epsilon to avoid log(0)
        return np.log(var + 1e-9)

def run_test():
    session_dir = r"PythonBCI\data\raw\gaci-26-04-29-09-06"
    
    session_files = sorted(glob.glob(os.path.join(session_dir, "batch_*.npz")))
    all_sess_eeg, all_sess_labels = [], []
    for f in session_files:
        if "merged" in f: continue
        d = np.load(f)
        all_sess_eeg.append(d['eeg'])
        all_sess_labels.append(d['labels'])
    sess_eeg = np.concatenate(all_sess_eeg, axis=0)
    sess_labels = np.concatenate(all_sess_labels, axis=0)
    
    if sess_eeg.shape[1] == 1001: sess_eeg = sess_eeg[:, :1000, :]
    elif sess_eeg.shape[2] == 1001: sess_eeg = sess_eeg[:, :, :1000]

    for i in range(sess_eeg.shape[0]):
        sample_axis = 1 if sess_eeg.shape[1] == 1000 else 2
        centered = sess_eeg[i] - np.mean(sess_eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0: sess_eeg[i] = centered / trial_std

    np.savez("temp_fft.npz", eeg=sess_eeg, labels=sess_labels)

    # Preprocess
    epochs, fs = load_and_preprocess("temp_fft.npz", eeg_device="Unicorn", force_car=False)
    
    X_train = epochs.get_data(copy=False)
    y_train = epochs.events[:, -1]

    # Unicorn mapping: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
    # Indices: C3=1, Cz=2, C4=3
    
    pipelines = {
        "LogVar_All_LR": Pipeline([
            ("extract", LogVarExtractor()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0))
        ]),
        "LogVar_All_SVM": Pipeline([
            ("extract", LogVarExtractor()),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        "LogVar_Motor_C3_Cz_C4_LR": Pipeline([
            ("extract", LogVarExtractor(picks=[1, 2, 3])),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0))
        ]),
        "LogVar_Motor_C3_C4_LR": Pipeline([
            ("extract", LogVarExtractor(picks=[1, 3])),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0))
        ])
    }
    
    print(f"\n--- Training Log-Variance (Band Power) Pipelines on Gaci Data ({len(y_train)} trials) ---")
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5)
        print(f"{name} 5-Fold Accuracy: {np.mean(scores):.4f}")

if __name__ == '__main__':
    run_test()
