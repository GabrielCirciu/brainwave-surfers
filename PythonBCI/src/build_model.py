import argparse
import os
import glob
import numpy as np
import pickle
import csv
from pathlib import Path
from train import train_model, AVAILABLE_PIPELINES

def main():
    parser = argparse.ArgumentParser(description="Merge calibration batches and train a BCI model.")
    parser.add_argument("--data_dir", required=True, help="Directory containing batch_*.npz files (e.g., PythonBCI/data/raw/gaci-...)")
    parser.add_argument("--pipeline", default="all", help=f"Pipeline name: all, {', '.join(AVAILABLE_PIPELINES)}")
    parser.add_argument("--use_grid", action="store_true", help="Use GridSearchCV for hyperparameters")
    parser.add_argument("--use_aux", action="store_true", help="Use auxiliary channels along with EEG")
    parser.add_argument("--save_dir", help="Directory to save the models (defaults to models/session_name)")
    parser.add_argument("--eeg_device", default="Unicorn", help="Device used for EEG: Unicorn, hiAmp, gold, BNCI2014_001")
    parser.add_argument("--resample_fs", action="store_true", help="Resample data to 250Hz")
    parser.add_argument("--visualize", action="store_true", help="Generate preprocessing verification plots")
    parser.add_argument("--force_car", action="store_true", help="Force Common Average Reference instead of CSD")
    parser.add_argument("--no_save", action="store_true", help="Don't save individual model files (only final model.pkl)")
    parser.add_argument("--combined", action="store_true", help="Train on all batches combined, ignoring individual batch performance")
    parser.add_argument("--extra_models", action="store_true", help="Include non-core models (CSP, standard Covariance) in evaluation")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    batch_files = sorted(glob.glob(str(data_dir / "batch_*.npz")))
    
    if not batch_files:
        print(f"Error: No batch_*.npz files found in {args.data_dir}")
        return
        
    if not args.save_dir:
        session_name = data_dir.name
        save_dir = Path(__file__).parents[1] / "models" / session_name
    else:
        save_dir = Path(args.save_dir)
    
    min_acc = 0.55
    metrics_log = []
    
    core_pipelines = ["aug_ts_lr", "aug_ts_svm", "aug_ts_mlp"]
    
    if args.pipeline == "all":
        if args.extra_models:
            pipelines_to_try = AVAILABLE_PIPELINES
        else:
            pipelines_to_try = core_pipelines
    else:
        pipelines_to_try = [args.pipeline]

    print(f"\nPerforming Pipeline-Specific Filtering (Threshold: {min_acc})...")
    
    best_overall_model = None
    best_overall_score = -1
    best_pipeline_name = ""
    
    for pipe_name in pipelines_to_try:
        print(f"\n" + "="*60)
        print(f"PIPELINE: {pipe_name}")
        print("="*60)
        
        def evaluate_all_batches(car_flag):
            batches_info = []
            accepted_count = 0
            for i, batch_file in enumerate(batch_files):
                try:
                    _, report = train_model(
                        data_path=batch_file, 
                        pipeline_name=pipe_name,
                        save_dir=save_dir,
                        use_grid=args.use_grid,
                        use_aux=args.use_aux,
                        resample_fs=args.resample_fs,
                        eeg_device=args.eeg_device,
                        visualize=False,
                        no_save=True,
                        force_car=car_flag
                    )
                    acc = report.get("accuracy", 0)
                    auc = report.get("auc", 0)
                    params = report.get("best_params", {})
                    is_accepted = acc >= min_acc
                    if is_accepted: accepted_count += 1
                    batches_info.append({
                        "Batch": i,
                        "Accuracy": acc,
                        "AUC": auc,
                        "Params": params,
                        "Accepted": is_accepted
                    })
                except:
                    batches_info.append({"Batch": i, "Accuracy": -1, "AUC": -1, "Params": {}, "Accepted": False})
            return accepted_count, batches_info

        if args.combined:
            print(f"Combined mode: Including all {len(batch_files)} batches regardless of performance.")
            valid_batches_for_pipe = []
            for i, batch_file in enumerate(batch_files):
                data = np.load(batch_file)
                eeg = data['eeg']
                labels = data['labels']
                aux = data['aux'] if 'aux' in data else None
                valid_batches_for_pipe.append((eeg, labels, aux))
            use_car_for_pipe = args.force_car
            prep_str = "CAR" if use_car_for_pipe else "CSD (Auto)"
        else:
            print("Testing all batches with CSD...")
            csd_count, csd_info = evaluate_all_batches(car_flag=False)
            print("Testing all batches with CAR...")
            car_count, car_info = evaluate_all_batches(car_flag=True)
            
            use_car_for_pipe = False
            if car_count > csd_count:
                use_car_for_pipe = True
            elif car_count == csd_count and car_count > 0:
                avg_csd = sum(b["Accuracy"] for b in csd_info if b["Accepted"]) / csd_count
                avg_car = sum(b["Accuracy"] for b in car_info if b["Accepted"]) / car_count
                if avg_car > avg_csd:
                    use_car_for_pipe = True
            
            chosen_info = car_info if use_car_for_pipe else csd_info
            prep_str = "CAR" if use_car_for_pipe else "CSD"
            print(f"\nChosen Preprocessing: {prep_str} (Accepted {len([b for b in chosen_info if b['Accepted']])}/{len(batch_files)} batches)")
            
            valid_batches_for_pipe = []
            for b in chosen_info:
                data = np.load(batch_files[b["Batch"]])
                eeg = data['eeg']
                labels = data['labels']
                aux = data['aux'] if 'aux' in data else None
                
                status = f"Accepted ({prep_str})" if b["Accepted"] else "Dropped"
                metrics_log.append({
                    "Batch": b["Batch"],
                    "Pipeline": pipe_name,
                    "Accuracy": b["Accuracy"],
                    "AUC": b["AUC"],
                    "Status": status,
                    "Best Params": str(b["Params"])
                })
                
                if b["Accepted"]:
                    valid_batches_for_pipe.append((eeg, labels, aux))
                    print(f"  Batch {b['Batch']}: {status} ({b['Accuracy']:.4f})")
                else:
                    print(f"  Batch {b['Batch']}: Dropped ({b['Accuracy']:.4f})")

        if not valid_batches_for_pipe:
            print(f"Warning: No batches passed threshold for {pipe_name}. Skipping.")
            continue
            
        print(f"\nMerging {len(valid_batches_for_pipe)} batches for {pipe_name}...")
        all_eeg = [b[0] for b in valid_batches_for_pipe]
        all_labels = [b[1] for b in valid_batches_for_pipe]
        all_aux = [b[2] for b in valid_batches_for_pipe if b[2] is not None]
        
        eeg_merged = np.concatenate(all_eeg, axis=0)
        labels_merged = np.concatenate(all_labels, axis=0)
        
        save_data = {"eeg": eeg_merged, "labels": labels_merged}
        if all_aux:
            save_data["aux"] = np.concatenate(all_aux, axis=0)
            
        pipe_merged_path = data_dir / f"merged_cleaned_{pipe_name}.npz"
        np.savez(pipe_merged_path, **save_data)
        
        print(f"Training final {pipe_name} on {len(labels_merged)} trials...")
        try:
            model, report = train_model(
                data_path=pipe_merged_path, 
                pipeline_name=pipe_name,
                save_dir=save_dir,
                use_grid=args.use_grid,
                use_aux=args.use_aux,
                resample_fs=args.resample_fs,
                eeg_device=args.eeg_device,
                visualize=args.visualize,
                no_save=args.no_save,
                force_car=use_car_for_pipe
            )
            
            final_score = report.get("auc", 0)
            if np.isnan(final_score) or final_score == 0:
                final_score = report.get("accuracy", 0)
                
            metrics_log.append({
                "Batch": "Final",
                "Pipeline": pipe_name,
                "Accuracy": report.get("accuracy", 0),
                "AUC": report.get("auc", 0),
                "Status": f"Merged ({len(valid_batches_for_pipe)} batches)",
                "Best Params": str(report.get("best_params", {}))
            })
            
            if final_score > best_overall_score:
                best_overall_score = final_score
                best_overall_model = model
                best_pipeline_name = pipe_name
                
        except Exception as e:
            print(f"Final training failed for {pipe_name}: {e}")

    if best_overall_model is None:
        print("\nError: All pipelines failed or no batches were accepted.")
        return
        
    print("\n" + "="*60)
    print(f"DONE! Best overall pipeline is {best_pipeline_name} with score {best_overall_score:.4f}")
    
    metrics_path = save_dir / "metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Batch", "Pipeline", "Accuracy", "AUC", "Status", "Best Params"])
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"Metrics log saved to: {metrics_path}")

    model_pkl_path = save_dir / "model.pkl"
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(best_overall_model, f)
        
    print(f"SUCCESS! Best model copied to {model_pkl_path} for real-time inference.")
    print("="*60)

if __name__ == "__main__":
    main()
