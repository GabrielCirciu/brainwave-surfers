import numpy as np
from pylsl import StreamInlet, resolve_byprop, resolve_streams
import time
from datetime import datetime
import os
import pickle
import glob
import csv
import argparse
from train import train_model

BUFFER_DUR = 4.0

def main():
    parser = argparse.ArgumentParser(description="Online model refinement: Combine base data with live trials.")
    parser.add_argument("--base_dir", required=True, help="Directory containing base training data (e.g., gold-data-2/full/subject_1)")
    parser.add_argument("--session_name", required=True,default="refine_session", help="Name for this new refinement session")
    parser.add_argument("--device", default="Unicorn", help="EEG Device: Unicorn, hiAmp")
    parser.add_argument("--use_aux", default=False, action="store_true", help="Use auxiliary channels along with EEG")
    args = parser.parse_args()

    # 1. Setup New Session
    session_name = args.session_name
    
    folder_time = datetime.now().strftime("%y-%m-%d-%H-%M")
    full_session_id = f"{session_name}-{folder_time}"
    
    output_dir = os.path.join("PythonBCI", "data", "raw", full_session_id)
    os.makedirs(output_dir, exist_ok=True)
    model_save_dir = os.path.join("PythonBCI", "models", full_session_id)
    os.makedirs(model_save_dir, exist_ok=True)
    
    print(f"\nRecording data to: {output_dir}")
    print(f"Models will be saved to: {model_save_dir}\n")

    # 2. Load Base Data
    print(f"Loading base data from {args.base_dir}...")
    base_files = sorted(glob.glob(os.path.join(args.base_dir, "batch_*.npz")))
    if not base_files:
        print(f"Error: No batch_*.npz files found in {args.base_dir}")
        return
        
    all_base_eeg = []
    all_base_aux = []
    all_base_labels = []
    for f in base_files:
        data = np.load(f)
        all_base_eeg.append(data['eeg'])
        if 'aux' in data:
            all_base_aux.append(data['aux'])
        else:
            # Create dummy aux if missing from base data
            dummy_aux = np.zeros((data['eeg'].shape[0], data['eeg'].shape[1], 10))
            all_base_aux.append(dummy_aux)
        all_base_labels.append(data['labels'])
    
    base_eeg = np.concatenate(all_base_eeg, axis=0)
    base_aux = np.concatenate(all_base_aux, axis=0)
    base_labels = np.concatenate(all_base_labels, axis=0)
    print(f"Loaded {len(base_labels)} trials from base dataset.")

    # 3. LSL Setup
    print("Looking for UnityMarkers stream...")
    marker_streams = resolve_byprop('name', 'UnityMarkers', 1, 3.0)
    if not marker_streams:
        print("Error: UnityMarkers stream not found. Is Unity Playing?")
        return
    marker_inlet = StreamInlet(marker_streams[0])
    
    print("Looking for EEG stream...")
    eeg_streams = [s for s in resolve_streams() if s.type() == 'Data']
    if not eeg_streams:
        print("Error: No EEG streams found.")
        return
    eeg_inlet = StreamInlet(eeg_streams[0])
    
    sampling_frequency = int(eeg_inlet.info().nominal_srate())
    stream_channels = eeg_inlet.info().channel_count()
    BUFFER_SAMPLES = int(sampling_frequency * BUFFER_DUR)
    
    # Determine expected AUX channels based on device/stream
    expected_aux_channels = 10 if stream_channels < 20 else 1

    # Sync base data sample length and AUX channels with live expectations to avoid concatenation errors
    print(f"Syncing base data to match live session (Length: {BUFFER_SAMPLES}, AUX: {expected_aux_channels})...")
    
    # Sync Length (Dimension 1)
    if base_eeg.shape[1] != BUFFER_SAMPLES:
        if base_eeg.shape[1] > BUFFER_SAMPLES:
            base_eeg = base_eeg[:, :BUFFER_SAMPLES, :]
            base_aux = base_aux[:, :BUFFER_SAMPLES, :]
        else:
            pad_w = BUFFER_SAMPLES - base_eeg.shape[1]
            base_eeg = np.pad(base_eeg, ((0,0), (0, pad_w), (0,0)), mode='edge')
            base_aux = np.pad(base_aux, ((0,0), (0, pad_w), (0,0)), mode='edge')
    
    # Sync AUX Channels (Dimension 2)
    if base_aux.shape[2] != expected_aux_channels:
        if base_aux.shape[2] > expected_aux_channels:
            base_aux = base_aux[:, :, :expected_aux_channels]
        else:
            pad_c = expected_aux_channels - base_aux.shape[2]
            base_aux = np.pad(base_aux, ((0,0), (0,0), (0, pad_c)), mode='constant')

    # 4. Recording State
    new_epochs_eeg = []
    new_epochs_aux = []
    new_labels = []
    
    batch_count = 0
    is_recording = False
    trial_chunks = []
    trial_timestamps = []
    metrics_log = []

    # 5. Baseline Evaluation (Before any new trials)
    print("\n--- Evaluating Baseline Performance (Gold Data Only) ---")
    temp_base_path = os.path.join(output_dir, "base_only.npz")
    
    base_save_dict = {"eeg": base_eeg, "labels": base_labels}
    if args.use_aux:
        base_save_dict["aux"] = base_aux
        
    np.savez(temp_base_path, **base_save_dict)
    
    pipelines = ["aug_ts_lr", "aug_ts_svm", "aug_ts_mlp"]
    for pipe in pipelines:
        try:
            model, report = train_model(
                data_path=temp_base_path, 
                pipeline_name=pipe,
                save_dir=model_save_dir,
                use_grid=False,
                use_aux=args.use_aux,
                eeg_device=args.device,
                no_save=True
            )
            score = report.get("accuracy", 0)
            print(f"  - {pipe}: {score:.4f}")
            metrics_log.append({
                "Batch": -1, # Baseline
                "Pipeline": pipe,
                "Accuracy": score,
                "AUC": report.get("auc", 0),
                "Status": "Baseline (Gold Only)",
                "Best Params": str(report.get("best_params", {}))
            })
        except Exception as e:
            print(f"  - {pipe} baseline failed: {e}")
            
    # Initial CSV save
    csv_path = os.path.join(model_save_dir, "metrics.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Batch", "Pipeline", "Accuracy", "AUC", "Status", "Best Params"])
        writer.writeheader()
        writer.writerows(metrics_log)

    print("\nREADY. Waiting for trial markers from Unity...")

    while True:
        chunk, timestamps = eeg_inlet.pull_chunk(timeout=0.1)
        if chunk and is_recording:
            trial_chunks.append(np.array(chunk)[:, :stream_channels])
            trial_timestamps.append(np.array(timestamps))

        marker, _ = marker_inlet.pull_sample(timeout=0.1)
        if marker:
            cmd = marker[0]
            if cmd in ("LEFT_START", "RIGHT_START"):
                is_recording = True
                current_trial_class = 0 if cmd == "LEFT_START" else 1
                trial_chunks, trial_timestamps = [], []
            
            elif cmd == "DISCARD":
                print("Trial discarded.")
                is_recording = False
            
            elif cmd in ("LEFT_END", "RIGHT_END") and is_recording:
                is_recording = False
                if trial_chunks:
                    trial_data = np.concatenate(trial_chunks, axis=0)
                    trial_ts = np.concatenate(trial_timestamps, axis=0)
                    
                    # Uniform Length
                    if trial_data.shape[0] >= BUFFER_SAMPLES:
                        trial_data = trial_data[:BUFFER_SAMPLES, :]
                    else:
                        trial_data = np.pad(trial_data, ((0, BUFFER_SAMPLES - trial_data.shape[0]), (0, 0)), mode='edge')
                    
                    # Split EEG/AUX
                    if stream_channels < 20:
                        eeg_data = trial_data[:, :8]
                        aux_data = trial_data[:, 8:17]
                        # Append timestamp to the last column of aux_data
                        ts_col = trial_ts[:BUFFER_SAMPLES].reshape(-1, 1)
                        if ts_col.shape[0] < BUFFER_SAMPLES:
                            ts_col = np.pad(ts_col, ((0, BUFFER_SAMPLES - ts_col.shape[0]), (0, 0)), mode='edge')
                        aux_data_full = np.hstack((aux_data, ts_col))
                    else:
                        eeg_data = trial_data
                        # For high-res devices, we just provide the timestamp as the only AUX channel
                        ts_col = trial_ts[:BUFFER_SAMPLES].reshape(-1, 1)
                        if ts_col.shape[0] < BUFFER_SAMPLES:
                            ts_col = np.pad(ts_col, ((0, BUFFER_SAMPLES - ts_col.shape[0]), (0, 0)), mode='edge')
                        aux_data_full = ts_col
                    
                    new_epochs_eeg.append(eeg_data)
                    new_epochs_aux.append(aux_data_full)
                    new_labels.append(current_trial_class)
                    
                    print(f"Trial saved! ({len(new_labels)}/10 in current batch)")

                    if len(new_labels) == 10:
                        print(f"\n--- Batch {batch_count} Complete ---")
                        print("Saving new data and retraining refined model...")
                        
                        # Save the new batch
                        new_eeg_arr = np.array(new_epochs_eeg)
                        new_aux_arr = np.array(new_epochs_aux)
                        new_labels_arr = np.array(new_labels)
                        batch_path = os.path.join(output_dir, f"batch_{batch_count}.npz")
                        np.savez(batch_path, eeg=new_eeg_arr, aux=new_aux_arr, labels=new_labels_arr)
                        
                        # Merge Base + ALL recorded batches in THIS session
                        session_batches = sorted(glob.glob(os.path.join(output_dir, "batch_*.npz")))
                        all_session_eeg = []
                        all_session_aux = []
                        all_session_labels = []
                        for sb in session_batches:
                            sdata = np.load(sb)
                            all_session_eeg.append(sdata['eeg'])
                            all_session_aux.append(sdata['aux'])
                            all_session_labels.append(sdata['labels'])
                        
                        # Final Merge: [Gold Data] + [Session Batch 0] + [Session Batch 1] ...
                        merged_eeg = np.concatenate([base_eeg] + all_session_eeg, axis=0)
                        merged_labels = np.concatenate([base_labels] + all_session_labels, axis=0)
                        
                        save_dict = {"eeg": merged_eeg, "labels": merged_labels}
                        if args.use_aux:
                            merged_aux = np.concatenate([base_aux] + all_session_aux, axis=0)
                            save_dict["aux"] = merged_aux
                        
                        temp_merged_path = os.path.join(output_dir, "merged_for_training.npz")
                        np.savez(temp_merged_path, **save_dict)
                        
                        print(f"Training on {len(merged_labels)} total trials ({len(base_labels)} base + {len(merged_labels)-len(base_labels)} new)...")
                        
                        pipelines = ["aug_ts_lr", "aug_ts_svm", "aug_ts_mlp"]
                        best_score, best_model, best_name = -1, None, ""
                        
                        for pipe in pipelines:
                            try:
                                model, report = train_model(
                                    data_path=temp_merged_path, 
                                    pipeline_name=pipe,
                                    save_dir=model_save_dir,
                                    use_grid=False,
                                    use_aux=args.use_aux,
                                    eeg_device=args.device,
                                    no_save=True
                                )
                                score = report.get("accuracy", 0)
                                print(f"  - {pipe}: {score:.4f}")
                                
                                metrics_log.append({
                                    "Batch": batch_count,
                                    "Pipeline": pipe,
                                    "Accuracy": score,
                                    "AUC": report.get("auc", 0),
                                    "Status": "Refined",
                                    "Best Params": str(report.get("best_params", {}))
                                })

                                if score > best_score:
                                    best_score, best_model, best_name = score, model, pipe
                            except Exception as e:
                                print(f"  - {pipe} failed: {e}")
                        
                        if best_model:
                            # Save refined model to session directory
                            model_path = os.path.join(model_save_dir, "model.pkl")
                            with open(model_path, 'wb') as f:
                                pickle.dump(best_model, f)
                            
                            # Deploy to root models folder for immediate use
                            root_model_path = os.path.join("PythonBCI", "models", "model.pkl")
                            with open(root_model_path, 'wb') as f:
                                pickle.dump(best_model, f)
                            print(f"Refined model ({best_name}) deployed to {root_model_path}\n")

                        # Save metrics after every batch
                        with open(csv_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=["Batch", "Pipeline", "Accuracy", "AUC", "Status", "Best Params"])
                            writer.writeheader()
                            writer.writerows(metrics_log)
                        
                        batch_count += 1
                        new_epochs_eeg, new_epochs_aux, new_labels = [], [], []

            elif cmd == "CALIBRATION_END":
                break

    print("\nRefinement session complete.")

if __name__ == '__main__':
    main()
