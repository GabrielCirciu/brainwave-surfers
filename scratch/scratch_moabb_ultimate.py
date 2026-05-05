import numpy as np
import os
import glob
import mne

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score

# MOABB / PyRiemann State-of-the-Art imports
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM, FgMDM
from pyriemann.spatialfilters import CSP
from pyriemann.clustering import Potato

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

    # strict DC removal per trial per channel
    for i in range(eeg.shape[0]):
        sample_axis = 1 if eeg.shape[1] == 1000 else 2
        centered = eeg[i] - np.mean(eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0: eeg[i] = centered / trial_std

    # MNE Epochs creation
    ch_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=['eeg']*8)
    
    events = np.zeros((len(labels), 3), dtype=int)
    events[:, 0] = np.arange(len(labels)) * 1000
    events[:, 2] = labels
    
    epochs = mne.EpochsArray(eeg * 1e-6, info, events=events, verbose=False)
    epochs.set_montage("standard_1020")
    return epochs

def run_ultimate():
    session_dir = r"PythonBCI\data\raw\vikt-26-04-29-14-07"
    
    epochs = load_custom_mne(session_dir)
    print("Loaded Raw Data. Applying 8-15 Hz MOABB Standard Filter...")
    epochs.filter(8.0, 15.0, fir_design='firwin', verbose=False)
    
    # We will test BOTH CAR and CSD mathematically!
    print("Creating CAR and CSD branches...")
    epochs_car = epochs.copy().set_eeg_reference("average", ch_type="eeg", verbose=False)
    
    try:
        epochs_csd = mne.preprocessing.compute_current_source_density(epochs.copy())
        csd_success = True
    except:
        csd_success = False

    X_car = epochs_car.get_data(copy=False)
    y = epochs_car.events[:, -1]
    
    if csd_success:
        X_csd = epochs_csd.get_data(copy=False)
    
    # ---------------------------------------------------------
    # RIEMANNIAN POTATO (State-of-the-Art Artifact Rejection)
    # ---------------------------------------------------------
    print("\nApplying Riemannian Potato (Covariance Centroid Artifact Rejection)...")
    covs_car = Covariances(estimator='oas').fit_transform(X_car)
    potato = Potato(metric='riemann', threshold=3.0).fit(covs_car)
    clean_indices = potato.predict(covs_car) == 1
    
    X_car_clean = X_car[clean_indices]
    y_clean = y[clean_indices]
    if csd_success:
        X_csd_clean = X_csd[clean_indices]
        
    print(f"Potato preserved {len(y_clean)}/{len(y)} trials lying near the true Riemannian manifold.")
    
    # ---------------------------------------------------------
    # MOABB STATE-OF-THE-ART PIPELINES
    # ---------------------------------------------------------
    pipelines = {
        "TS + ElasticNet (MOABB Winner)": Pipeline([
            ("cov", Covariances(estimator="lwf")), # Ledoit-Wolf is often superior to OAS in MOABB
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty='elasticnet', l1_ratio=0.15, solver='saga', max_iter=2000, random_state=42))
        ]),
        "CSP + Ridge Classifier": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("csp", CSP(nfilter=6, log=True)),
            ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
        ]),
        "FgMDM (Geodesic filtering)": Pipeline([
            ("cov", Covariances(estimator="lwf")),
            ("fgmdm", FgMDM(metric="riemann"))
        ])
    }
    
    # Create an ENSEMBLE of the best models
    ensemble = VotingClassifier(estimators=[
        ('ts', pipelines["TS + ElasticNet (MOABB Winner)"]),
        ('csp', pipelines["CSP + Ridge Classifier"]),
        ('fgmdm', pipelines["FgMDM (Geodesic filtering)"])
    ], voting='hard')
    
    pipelines["Ultimate MOABB Ensemble"] = ensemble
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    def evaluate(X_data, label):
        print(f"\n--- Testing on {label} Data ---")
        best_auc, best_pipe = 0, ""
        for name, pipe in pipelines.items():
            acc_scores = cross_val_score(pipe, X_data, y_clean, cv=cv, scoring='accuracy', n_jobs=-1)
            # AUC requires predict_proba, which Voting(hard) and Ridge don't support directly, so we just log Accuracy
            acc = np.mean(acc_scores)
            print(f"{name.ljust(35)} -> Acc: {acc:.4f}")
            if acc > best_auc:
                best_auc = acc
                best_pipe = name
        return best_auc, best_pipe

    evaluate(X_car_clean, "Common Average Reference (CAR)")
    if csd_success:
        evaluate(X_csd_clean, "Current Source Density (CSD)")

if __name__ == '__main__':
    run_test = run_ultimate
    run_test()
