import numpy as np
import os
import glob
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from PythonBCI.src.train import load_and_preprocess

def run_test():
    session_dir = r"PythonBCI\data\raw\gaci-26-04-29-09-06"
    
    # Load GACI Session
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

    # Normalize GACI
    for i in range(sess_eeg.shape[0]):
        sample_axis = 1 if sess_eeg.shape[1] == 1000 else 2
        centered = sess_eeg[i] - np.mean(sess_eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0: sess_eeg[i] = centered / trial_std

    np.savez("temp_brainstorm.npz", eeg=sess_eeg, labels=sess_labels)

    # Process through exact pipeline
    epochs, fs = load_and_preprocess("temp_brainstorm.npz", eeg_device="Unicorn", force_car=False)
    
    # ARTIFACT REJECTION
    # Drop trials where peak-to-peak amplitude exceeds a threshold
    # Since we normalized to std=1, max amp after CSD is ~3uV.
    # We can try dropping trials with amp > 2.0 uV.
    # Wait, the noise might be low frequency and removed by bandpass.
    # Let's just drop the top 20% noisiest trials!
    amps = np.max(np.abs(epochs.get_data(copy=False)), axis=(1,2))
    threshold = np.percentile(amps, 80) # Keep lowest 80%
    
    clean_indices = np.where(amps < threshold)[0]
    epochs_clean = epochs[clean_indices]
    
    print(f"Original trials: {len(epochs)}, Clean trials: {len(epochs_clean)}")
    
    X_train = epochs_clean.get_data(copy=False)
    y_train = epochs_clean.events[:, -1]

    pipelines = {
        "ts_lr": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=2000))
        ]),
        "ts_svm_strongL2": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=0.01, solver='lbfgs', max_iter=2000)) # Stronger L2
        ])
    }
    
    from sklearn.model_selection import cross_val_score
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5)
        print(f"{name} 5-Fold Acc: {np.mean(scores):.4f}")

if __name__ == '__main__':
    run_test()
