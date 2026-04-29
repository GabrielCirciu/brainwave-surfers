import numpy as np
import os
import glob
from PythonBCI.src.train import train_model

def run_test():
    sub1_dir = r"PythonBCI\data\raw\gold-data-2\stripped\subject_1"
    sub2_dir = r"PythonBCI\data\raw\gold-data-2\stripped\subject_2"
    session_dir = r"PythonBCI\data\raw\gaci-26-04-29-09-06"
    
    # Load Subject 1
    sub1_files = sorted(glob.glob(os.path.join(sub1_dir, "batch_*.npz")))
    all_sub1_eeg, all_sub1_labels = [], []
    for f in sub1_files:
        d = np.load(f)
        all_sub1_eeg.append(d['eeg'])
        all_sub1_labels.append(d['labels'])
    sub1_eeg = np.concatenate(all_sub1_eeg, axis=0)
    sub1_labels = np.concatenate(all_sub1_labels, axis=0)
    
    # Load Subject 2
    sub2_files = sorted(glob.glob(os.path.join(sub2_dir, "batch_*.npz")))
    all_sub2_eeg, all_sub2_labels = [], []
    for f in sub2_files:
        d = np.load(f)
        all_sub2_eeg.append(d['eeg'])
        all_sub2_labels.append(d['labels'])
    sub2_eeg = np.concatenate(all_sub2_eeg, axis=0)
    sub2_labels = np.concatenate(all_sub2_labels, axis=0)
    
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
    
    # Combine Base Data
    base_eeg = np.concatenate([sub1_eeg, sub2_eeg], axis=0)
    base_labels = np.concatenate([sub1_labels, sub2_labels], axis=0)

    # Fix sizes before merging
    if base_eeg.shape[1] == 1001:
        base_eeg = base_eeg[:, :1000, :]
    elif base_eeg.shape[2] == 1001:
        base_eeg = base_eeg[:, :, :1000]
        
    if sess_eeg.shape[1] == 1001:
        sess_eeg = sess_eeg[:, :1000, :]
    elif sess_eeg.shape[2] == 1001:
        sess_eeg = sess_eeg[:, :, :1000]

    # Normalize BOTH to std=1 per trial, removing DC offset first
    for i in range(base_eeg.shape[0]):
        sample_axis = 1 if base_eeg.shape[1] == 1000 else 2
        centered = base_eeg[i] - np.mean(base_eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0:
            base_eeg[i] = centered / trial_std
            
    for i in range(sess_eeg.shape[0]):
        sample_axis = 1 if sess_eeg.shape[1] == 1000 else 2
        centered = sess_eeg[i] - np.mean(sess_eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0:
            sess_eeg[i] = centered / trial_std

    print(f"Normalized and merging: Base trials: {len(base_labels)}, Session trials: {len(sess_labels)}")
    
    merged_eeg = np.concatenate([base_eeg, sess_eeg], axis=0)
    merged_labels = np.concatenate([base_labels, sess_labels], axis=0)
    
    np.savez("temp_merged.npz", eeg=merged_eeg, labels=merged_labels)
    
    pipelines = ["aug_ts_lr", "aug_ts_svm", "aug_ts_mlp", "csp_lda"]
    for pipe in pipelines:
        print(f"\n--- Training {pipe} ---")
        try:
            model, report = train_model("temp_merged.npz", pipeline_name=pipe, no_save=True, eeg_device="Unicorn")
            print(f"Result for {pipe} -> AUC: {report['auc']:.4f}, Accuracy: {report['accuracy']:.4f}")
        except Exception as e:
            print(f"Failed {pipe}: {e}")

if __name__ == '__main__':
    run_test()
