import argparse
import os
import glob
import numpy as np
import pickle
from train import train_model

def main():
    parser = argparse.ArgumentParser(description="Merge calibration batches and train a BCI model.")
    parser.add_argument("--data_dir", required=True, help="Directory containing batch_*.npz files (e.g., PythonBCI/data/raw/gaci-...)")
    parser.add_argument("--pipeline", default="all", help="Pipeline name: all, cov_ts_lr, csp_svm, csp_lda, csp_rf")
    parser.add_argument("--use_grid", action="store_true", help="Use GridSearchCV for hyperparameters")
    args = parser.parse_args()

    # 1. Merge the batch files
    batch_files = glob.glob(os.path.join(args.data_dir, "batch_*.npz"))
    
    if not batch_files:
        print(f"Error: No batch_*.npz files found in {args.data_dir}")
        return
        
    print(f"Found {len(batch_files)} batch files. Merging...")
    all_eeg = []
    all_labels = []
    
    for f in batch_files:
        data = np.load(f)
        all_eeg.append(data['eeg'])
        all_labels.append(data['labels'])
        
    eeg = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    merged_path = os.path.join(args.data_dir, "merged.npz")
    np.savez(merged_path, eeg=eeg, labels=labels)
    
    print(f"Merged data saved to {merged_path}. Total shape: {eeg.shape}")
    
    # 2. Call train.py's internal function
    save_dir = os.path.join("PythonBCI", "models")
    
    pipelines_to_try = ["cov_ts_lr", "csp_svm", "csp_lda", "csp_rf"] if args.pipeline == "all" else [args.pipeline]
    
    best_overall_model = None
    best_overall_score = -1
    best_pipeline_name = ""
    
    for pipe_name in pipelines_to_try:
        print(f"\n--- Training pipeline: {pipe_name} ---")
        try:
            model, report = train_model(
                data_path=merged_path, 
                pipeline_name=pipe_name,
                save_dir=save_dir,
                use_grid=args.use_grid
            )
            score = report.get("accuracy", 0)
            if np.isnan(score):
                score = report.get("auc", 0)
                
            print(f"Pipeline {pipe_name} achieved score: {score:.4f}")
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_model = model
                best_pipeline_name = pipe_name
        except Exception as e:
            print(f"Pipeline {pipe_name} failed: {e}")
            
    if best_overall_model is None:
        print("\nError: All pipelines failed to train.")
        return
        
    print(f"\n==================================================")
    print(f"WINNER: {best_pipeline_name} with score {best_overall_score:.4f}")
    print(f"==================================================")
    
    # 3. Explicitly save the output as "model.pkl" so realtime_predict.py can find it
    model_pkl_path = os.path.join(save_dir, "model.pkl")
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(best_overall_model, f)
        
    print(f"SUCCESS! Best model copied to {model_pkl_path} for real-time inference.")

if __name__ == "__main__":
    main()
