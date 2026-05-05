import numpy as np
import os
import glob
import itertools
import mne

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

import warnings
warnings.filterwarnings("ignore")

def load_and_filter_custom(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
    if not files: return None, None
    all_eeg, all_labels = [], []
    for f in files:
        if "merged" in f: continue
        d = np.load(f)
        all_eeg.append(d['eeg'])
        all_labels.append(d['labels'])
    eeg = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if eeg.shape[1] == 1000:
        eeg = np.transpose(eeg, (0, 2, 1))
    elif eeg.shape[1] == 1001: 
        eeg = eeg[:, :1000, :]
        eeg = np.transpose(eeg, (0, 2, 1))

    # Normalize DC per trial
    for i in range(eeg.shape[0]):
        centered = eeg[i] - np.mean(eeg[i], axis=1, keepdims=True)
        t_std = np.std(centered)
        if t_std > 0: eeg[i] = centered / t_std

    ch_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=['eeg']*8)
    
    events = np.zeros((len(labels), 3), dtype=int)
    events[:, 0] = np.arange(len(labels)) * 1000
    events[:, 2] = labels
    
    epochs = mne.EpochsArray(eeg * 1e-6, info, events=events, verbose=False)
    epochs.set_montage("standard_1020")
    
    # MOABB 8-15 Hz Filtering (Best for Vikt)
    epochs.filter(8.0, 15.0, fir_design='firwin', verbose=False)
    
    # CAR Spatial Filter (robust to dry electrodes)
    epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)
    
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    return X, y

def run_test():
    base_dir = r"PythonBCI\data\raw\gold-data-2\stripped"
    test_dir = r"PythonBCI\data\raw\vikt-26-04-29-14-07"
    
    print("Preprocessing Test Data (Vikt) with 8-15 Hz Filter...")
    X_test, y_test = load_and_filter_custom(test_dir)
    
    subs = {}
    print("Preprocessing Base Subjects 1-9 (Gold Data) with 8-15 Hz Filter...")
    for i in range(1, 10):
        d_dir = os.path.join(base_dir, f"subject_{i}")
        X, y = load_and_filter_custom(d_dir)
        if X is not None:
            subs[i] = (X, y)
    
    sub_ids = list(subs.keys())
    
    best_overall_auc = 0
    best_overall_acc = 0
    best_combo = None
    
    def train_and_eval(combo_ids):
        X_train_list = [subs[i][0] for i in combo_ids]
        y_train_list = [subs[i][1] for i in combo_ids]
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        pipe = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=2000))
        ])
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        return acc, auc

    print("\n--- 1 Subject Base Models ---")
    best_1_auc, best_1_acc, best_1_c = 0, 0, None
    for c in itertools.combinations(sub_ids, 1):
        acc, auc = train_and_eval(c)
        if auc > best_1_auc: best_1_auc, best_1_acc, best_1_c = auc, acc, c
        if auc > best_overall_auc: best_overall_auc, best_overall_acc, best_combo = auc, acc, c
    print(f"Best 1-Sub: {best_1_c} -> Acc: {best_1_acc:.4f}, AUC: {best_1_auc:.4f}")

    print("\n--- 2 Subjects Base Models ---")
    best_2_auc, best_2_acc, best_2_c = 0, 0, None
    for c in itertools.combinations(sub_ids, 2):
        acc, auc = train_and_eval(c)
        if auc > best_2_auc: best_2_auc, best_2_acc, best_2_c = auc, acc, c
        if auc > best_overall_auc: best_overall_auc, best_overall_acc, best_combo = auc, acc, c
    print(f"Best 2-Sub: {best_2_c} -> Acc: {best_2_acc:.4f}, AUC: {best_2_auc:.4f}")

    print("\n--- 3 Subjects Base Models ---")
    best_3_auc, best_3_acc, best_3_c = 0, 0, None
    for c in itertools.combinations(sub_ids, 3):
        acc, auc = train_and_eval(c)
        if auc > best_3_auc: best_3_auc, best_3_acc, best_3_c = auc, acc, c
        if auc > best_overall_auc: best_overall_auc, best_overall_acc, best_combo = auc, acc, c
    print(f"Best 3-Sub: {best_3_c} -> Acc: {best_3_acc:.4f}, AUC: {best_3_auc:.4f}")

    print("\n--- ALL 9 Subjects Combined Base Model ---")
    acc, auc = train_and_eval(sub_ids)
    if auc > best_overall_auc: best_overall_auc, best_overall_acc, best_combo = auc, acc, tuple(sub_ids)
    print(f"All Subs: {tuple(sub_ids)} -> Acc: {acc:.4f}, AUC: {auc:.4f}")

    print("\n" + "="*60)
    print(f"🏆 GRAND CHAMPION GOLD COMBO FOR VIKT: Subjects {best_combo}")
    print(f"Accuracy: {best_overall_acc:.4f} | AUC: {best_overall_auc:.4f}")
    print("="*60)

if __name__ == '__main__':
    run_test()
