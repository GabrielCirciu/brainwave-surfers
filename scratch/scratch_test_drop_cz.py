import numpy as np
import os
import glob
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from PythonBCI.src.train import load_and_preprocess

def run_test():
    session_dir = r"PythonBCI\data\raw\vikt-26-04-29-14-07"
    
    # Load VIKT Session (since it had the best baseline)
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

    # Normalize
    for i in range(sess_eeg.shape[0]):
        sample_axis = 1 if sess_eeg.shape[1] == 1000 else 2
        centered = sess_eeg[i] - np.mean(sess_eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0: sess_eeg[i] = centered / trial_std

    np.savez("temp_cz_drop.npz", eeg=sess_eeg, labels=sess_labels)

    print("--- Testing Baseline (All 8 Channels) ---")
    epochs, fs = load_and_preprocess("temp_cz_drop.npz", eeg_device="Unicorn", force_car=False)
    X_all = epochs.get_data(copy=False)
    y_all = epochs.events[:, -1]
    
    pipe = Pipeline([
        ("cov", Covariances(estimator="oas")),
        ("ts", TangentSpace(metric="riemann")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=2000))
    ])
    
    scores_all = cross_val_score(pipe, X_all, y_all, cv=5)
    print(f"All 8 Channels Accuracy: {np.mean(scores_all):.4f}")

    print("\n--- Testing Without Cz and CPz (6 Channels) ---")
    epochs.drop_channels(["Cz", "CPz"])
    X_dropped = epochs.get_data(copy=False)
    
    scores_dropped = cross_val_score(pipe, X_dropped, y_all, cv=5)
    print(f"Dropped Cz/CPz Accuracy: {np.mean(scores_dropped):.4f}")

if __name__ == '__main__':
    run_test()
