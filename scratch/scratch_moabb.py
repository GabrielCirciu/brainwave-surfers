import numpy as np
import os
import glob
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM
from pyriemann.spatialfilters import CSP

from PythonBCI.src.train import load_and_preprocess
import warnings
warnings.filterwarnings("ignore")

def run_test():
    session_dir = r"PythonBCI\data\raw\myta-26-04-30-10-54"
    
    # Load Myta Session
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

    np.savez("temp_moabb.npz", eeg=sess_eeg, labels=sess_labels)

    # Preprocess
    print("Preprocessing data...")
    epochs, fs = load_and_preprocess("temp_moabb.npz", eeg_device="Unicorn", force_car=True) # Force CAR because CSD often fails on 8 dry electrodes
    
    # Let's drop bad trials to give it the absolute best chance!
    epochs.drop_bad(reject=dict(eeg=200e-6), verbose=False)
    
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    
    print(f"Testing on {len(y)} clean trials...")

    # Define State-of-the-Art MOABB Pipelines
    pipelines = {
        "TS + LogisticRegression (L2)": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs'))
        ]),
        "TS + LogisticRegression (L1 Sparse)": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, penalty='l1', C=0.5, solver='saga'))
        ]),
        "Minimum Distance to Mean (MDM)": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("mdm", MDM(metric="riemann"))
        ]),
        "Geodesic Filtering MDM (FgMDM)": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("fgmdm", FgMDM(metric="riemann"))
        ]),
        "CSP (4 filters) + LDA": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("csp", CSP(nfilter=4, log=True)),
            ("clf", LDA())
        ]),
        "CSP (6 filters) + SVM": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("csp", CSP(nfilter=6, log=True)),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel='rbf', probability=True))
        ]),
        "TS + Random Forest": Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
        ])
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- MOABB State-of-the-Art Pipeline Evaluation ---")
    results = []
    for name, pipe in pipelines.items():
        try:
            acc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            results.append((name, np.mean(acc_scores), np.mean(auc_scores)))
        except Exception as e:
            print(f"Failed {name}: {e}")
            
    # Sort by AUC
    results.sort(key=lambda x: x[2], reverse=True)
    
    for name, acc, auc in results:
        print(f"{name.ljust(35)} -> Acc: {acc:.4f} | AUC: {auc:.4f}")

if __name__ == '__main__':
    run_test()
