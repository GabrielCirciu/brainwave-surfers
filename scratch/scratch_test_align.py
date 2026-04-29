import numpy as np
import os
import glob
from PythonBCI.src.train import train_model

def run_test():
    base_dir = r"PythonBCI\data\raw\gold-data-2\stripped\subject_1"
    session_dir = r"PythonBCI\data\raw\gaci-26-04-29-09-06"
    
    # Load base data
    base_files = sorted(glob.glob(os.path.join(base_dir, "batch_*.npz")))
    all_base_eeg, all_base_labels = [], []
    for f in base_files:
        d = np.load(f)
        all_base_eeg.append(d['eeg'])
        all_base_labels.append(d['labels'])
    base_eeg = np.concatenate(all_base_eeg, axis=0)
    base_labels = np.concatenate(all_base_labels, axis=0)
    
    # Load session data
    session_files = sorted(glob.glob(os.path.join(session_dir, "batch_*.npz")))
    all_sess_eeg, all_sess_labels = [], []
    for f in session_files:
        if "merged" in f: continue
        d = np.load(f)
        all_sess_eeg.append(d['eeg'])
        all_sess_labels.append(d['labels'])
    sess_eeg = np.concatenate(all_sess_eeg, axis=0)
    sess_labels = np.concatenate(all_sess_labels, axis=0)
    
    # Extract only C3, Cz, C4
    # Base: 1=C3, 3=Cz, 6=C4
    base_eeg_aligned = base_eeg[:, :, [1, 3, 6]]
    
    # Sess: 1=C3, 2=Cz, 3=C4
    sess_eeg_aligned = sess_eeg[:, :, [1, 2, 3]]
    
    # Truncate
    if base_eeg_aligned.shape[1] == 1001:
        base_eeg_aligned = base_eeg_aligned[:, :1000, :]
    if sess_eeg_aligned.shape[1] == 1001:
        sess_eeg_aligned = sess_eeg_aligned[:, :1000, :]

    # Normalize BOTH properly (center then std)
    for i in range(base_eeg_aligned.shape[0]):
        centered = base_eeg_aligned[i] - np.mean(base_eeg_aligned[i], axis=0, keepdims=True)
        t_std = np.std(centered)
        if t_std > 0: base_eeg_aligned[i] = centered / t_std
            
    for i in range(sess_eeg_aligned.shape[0]):
        centered = sess_eeg_aligned[i] - np.mean(sess_eeg_aligned[i], axis=0, keepdims=True)
        t_std = np.std(centered)
        if t_std > 0: sess_eeg_aligned[i] = centered / t_std

    merged_eeg = np.concatenate([base_eeg_aligned, sess_eeg_aligned], axis=0)
    merged_labels = np.concatenate([base_labels, sess_labels], axis=0)
    
    np.savez("temp_aligned.npz", eeg=merged_eeg, labels=merged_labels)
    
    print(f"\n--- Training Aligned Data (3 channels) ---")
    try:
        # Force CAR because CSD requires 10-20 standard which we just broke by using only 3 channels
        model, report = train_model("temp_aligned.npz", pipeline_name="aug_ts_lr", no_save=True, eeg_device="Unicorn", force_car=True)
        print(f"Result for aug_ts_lr -> AUC: {report['auc']:.4f}, Accuracy: {report['accuracy']:.4f}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    run_test()
