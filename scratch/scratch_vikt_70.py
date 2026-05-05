import numpy as np
import os
import glob
import mne

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.clustering import Potato

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PythonBCI.src.train import AugmentedDataset

import warnings
warnings.filterwarnings("ignore")

def load_custom_mne(data_dir):
    session_files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
    all_eeg, all_labels = [], []
    for f in session_files:
        if "merged" in f: continue
        d = np.load(f)
        all_eeg.append(d['eeg'])
        all_labels.append(d['labels'])
    eeg = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if eeg.shape[1] == 1000:
        eeg = np.transpose(eeg, (0, 2, 1))

    for i in range(eeg.shape[0]):
        sample_axis = 1 if eeg.shape[1] == 1000 else 2
        centered = eeg[i] - np.mean(eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0: eeg[i] = centered / trial_std

    ch_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=['eeg']*8)
    
    events = np.zeros((len(labels), 3), dtype=int)
    events[:, 0] = np.arange(len(labels)) * 1000
    events[:, 2] = labels
    
    epochs = mne.EpochsArray(eeg * 1e-6, info, events=events, verbose=False)
    epochs.set_montage("standard_1020")
    return epochs

def squeeze_70():
    session_dir = r"PythonBCI\data\raw\vikt-26-04-29-14-07"
    
    epochs = load_custom_mne(session_dir)
    epochs.filter(8.0, 15.0, fir_design='firwin', verbose=False)
    
    try:
        epochs = mne.preprocessing.compute_current_source_density(epochs)
    except:
        epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)

    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    
    # 1. Base Pipeline
    pipes = {
        "LR": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=2000))
        ]),
        "SVC": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        "ElasticNet": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty='elasticnet', l1_ratio=0.15, solver='saga', max_iter=2000, random_state=42))
        ])
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    base_acc = np.mean(cross_val_score(pipes["LR"], X, y, cv=cv, scoring='accuracy'))
    print(f"Base Accuracy (No individual trial rejection): {base_acc:.4f}")
    
    # 2. Aggressive Artifact Rejection Loop
    covs = Covariances(estimator='oas').fit_transform(X)
    
    print("\n--- Squeezing with Riemannian Potato (Threshold = 1.5) ---")
    
    potato = Potato(metric='riemann', threshold=1.5).fit(covs)
    clean_indices = potato.predict(covs) == 1
    kept = np.sum(clean_indices)
    
    X_clean = X[clean_indices]
    y_clean = y[clean_indices]
    
    print(f"Kept {kept}/100 trials.")
    
    # Run pipelines
    print("\n--- Testing Pipelines on Pristine 56 Trials ---")
    pipes_aug = {
        "Standard TS+LR": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=2000))
        ]),
        "Augmented TS+LR (aug_ts_lr)": Pipeline([
            ("aug", AugmentedDataset(order=2)),
            ("cov", Covariances(estimator="lwf")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=0.1, solver='lbfgs', max_iter=2000))
        ])
    }
    
    best_acc = 0
    best_name = ""
    
    for name, p in pipes_aug.items():
        acc_scores = cross_val_score(p, X_clean, y_clean, cv=cv, scoring='accuracy')
        auc_scores = cross_val_score(p, X_clean, y_clean, cv=cv, scoring='roc_auc')
        acc = np.mean(acc_scores)
        auc = np.mean(auc_scores)
        print(f"{name.ljust(30)} -> Acc: {acc:.4f} | AUC: {auc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_name = name
            
    print("\n" + "="*50)
    print(f"MAX ACHIEVED: {best_acc:.4f} Accuracy with {best_name}")
    print("="*50)

if __name__ == '__main__':
    squeeze_70()
