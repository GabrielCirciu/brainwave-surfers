import numpy as np
import os
import glob
from PythonBCI.src.train import train_model

def run_test():
    session_dir = r"PythonBCI\data\raw\gaci-26-04-29-09-06"
    
    # Load raw session data (excluding 'merged')
    session_files = sorted(glob.glob(os.path.join(session_dir, "batch_*.npz")))
    all_sess_eeg, all_sess_labels = [], []
    for f in session_files:
        if "merged" in f: continue
        d = np.load(f)
        all_sess_eeg.append(d['eeg'])
        all_sess_labels.append(d['labels'])
    
    sess_eeg = np.concatenate(all_sess_eeg, axis=0)
    sess_labels = np.concatenate(all_sess_labels, axis=0)
    
    # Truncate to 1000
    if sess_eeg.shape[1] == 1001:
        sess_eeg = sess_eeg[:, :1000, :]
    elif sess_eeg.shape[2] == 1001:
        sess_eeg = sess_eeg[:, :, :1000]

    # PROPER NORMALIZATION: Remove DC offset before calculating std!
    print("Variance before centering:", np.var(sess_eeg[0]))
    for i in range(sess_eeg.shape[0]):
        # Depending on shape, channel axis is 1 or 2. 
        # If shape is (Trials, Samples, Channels) like online_refine... wait, 
        # online_refine saves (Trials, Samples, Channels) or (Trials, Channels, Samples)?
        # Let's just center along the sample axis
        
        # Determine sample axis: if shape[1] == 1000, sample axis is 1.
        sample_axis = 1 if sess_eeg.shape[1] == 1000 else 2
        
        # Center each channel
        means = np.mean(sess_eeg[i], axis=sample_axis - 1, keepdims=True)
        centered = sess_eeg[i] - means
        
        trial_std = np.std(centered)
        if trial_std > 0:
            sess_eeg[i] = centered / trial_std

    print("Variance after centering & normalizing:", np.var(sess_eeg[0]))

    np.savez("temp_sess_proper.npz", eeg=sess_eeg, labels=sess_labels)
    
    print(f"\n--- Training Only Session Data ({len(sess_labels)} trials) PROPER NORMALIZATION ---")
    try:
        model, report = train_model("temp_sess_proper.npz", pipeline_name="aug_ts_lr", no_save=True, eeg_device="Unicorn")
        print(f"Result for aug_ts_lr -> AUC: {report['auc']:.4f}, Accuracy: {report['accuracy']:.4f}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    run_test()
