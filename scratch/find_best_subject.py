import numpy as np
import os
import glob
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from PythonBCI.src.train import load_and_preprocess

def run_test():
    base_dir = r"PythonBCI\data\raw\gold-data-2\stripped"
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

    best_acc = 0
    best_sub = -1
    
    print(f"Testing Transfer Learning to Gaci Data ({len(sess_labels)} trials)")

    for sub_id in range(1, 10):
        sub_dir = os.path.join(base_dir, f"subject_{sub_id}")
        sub_files = sorted(glob.glob(os.path.join(sub_dir, "batch_*.npz")))
        if not sub_files: continue
        
        all_sub_eeg, all_sub_labels = [], []
        for f in sub_files:
            d = np.load(f)
            all_sub_eeg.append(d['eeg'])
            all_sub_labels.append(d['labels'])
        sub_eeg = np.concatenate(all_sub_eeg, axis=0)
        sub_labels = np.concatenate(all_sub_labels, axis=0)
        
        if sub_eeg.shape[1] == 1001: sub_eeg = sub_eeg[:, :1000, :]
        elif sub_eeg.shape[2] == 1001: sub_eeg = sub_eeg[:, :, :1000]

        # Normalize Subject
        for i in range(sub_eeg.shape[0]):
            sample_axis = 1 if sub_eeg.shape[1] == 1000 else 2
            centered = sub_eeg[i] - np.mean(sub_eeg[i], axis=sample_axis-1, keepdims=True)
            trial_std = np.std(centered)
            if trial_std > 0: sub_eeg[i] = centered / trial_std

        # Combine
        merged_eeg = np.concatenate([sub_eeg, sess_eeg], axis=0)
        merged_labels = np.concatenate([sub_labels, sess_labels], axis=0)
        np.savez("temp_transfer.npz", eeg=merged_eeg, labels=merged_labels)

        # Process through exact pipeline
        try:
            epochs, fs = load_and_preprocess("temp_transfer.npz", eeg_device="Unicorn", force_car=False)
            X_all = epochs.get_data(copy=False)
            y_all = epochs.events[:, -1]
        except Exception as e:
            print(f"Sub {sub_id} Failed preprocessing: {e}")
            continue

        X_train = X_all[:len(sub_labels)]
        y_train = y_all[:len(sub_labels)]
        X_test = X_all[len(sub_labels):]
        y_test = y_all[len(sub_labels):]

        # Train model strictly on Subject data
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, C=1.0, solver='lbfgs', max_iter=2000))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Test on Gaci data
        probs = pipeline.predict_proba(X_test)
        preds = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs[:, 1])
        
        # Also compute internal accuracy for reference
        int_preds = pipeline.predict(X_train)
        int_acc = accuracy_score(y_train, int_preds)
        
        print(f"Subject {sub_id:2d} -> Transfer Acc: {acc:.4f} (AUC: {auc:.4f}) | Internal Sub Acc: {int_acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_sub = sub_id

    print(f"\nWINNER: Subject {best_sub} was the most aligned with Gaci data! (Accuracy: {best_acc:.4f})")

if __name__ == '__main__':
    run_test()
