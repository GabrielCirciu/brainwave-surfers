import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import train_model

DATA_DIRS = [
    #r".\PythonBCI\data\raw\1\raw",
    #r".\PythonBCI\data\raw\2",
]

def main():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    pipelines = ["aug_ts_lr", "aug_ts_svm", "aug_ts_mlp"]
    min_acc = 0.55

    final_results = []
    dataset_names = []

    for d_idx, data_dir in enumerate(DATA_DIRS):
        path = Path(data_dir)
        dataset_names.append(path.name)
        
        batch_files = sorted(glob.glob(str(path / "batch_*.npz")))
        if not batch_files:
            print(f"Warning: No batch_*.npz files found in {data_dir}")
            final_results.append({})
            continue
            
        print(f"\nProcessing Dataset {d_idx + 1}/{len(DATA_DIRS)}: {path.name} with {len(batch_files)} batches")
        
        dataset_res = {pipe: {} for pipe in pipelines}
        for b_idx, batch_file in enumerate(batch_files):
            print(f"  Batch {b_idx + 1}/{len(batch_files)}: {Path(batch_file).name}")
            for pipe in pipelines:
                print(f"    Evaluating {pipe}...")
                try:
                    _, report = train_model(
                        data_path=batch_file,
                        pipeline_name=pipe,
                        save_dir="temp",
                        use_grid=False,
                        eeg_device="Unicorn",
                        force_car=True,
                        no_save=True
                    )
                    acc = report.get("accuracy", 0)
                    if np.isnan(acc): acc = 0.0
                    
                    if acc >= min_acc:
                        dataset_res[pipe][b_idx] = acc
                        print(f"      Accuracy: {acc:.4f} (Accepted)")
                    else:
                        print(f"      Accuracy: {acc:.4f} (Dropped - below {min_acc})")
                        dataset_res[pipe][b_idx] = acc
                except Exception as e:
                    print(f"      Error evaluating {pipe}: {e}")
                    dataset_res[pipe][b_idx] = 0.0
                    
        final_dataset_res = {}
        for pipe in pipelines:
            sorted_batches = sorted(dataset_res[pipe].items(), key=lambda x: x[1], reverse=True)
            accepted_batches = [b for i, (b, acc) in enumerate(sorted_batches) if acc >= min_acc or i < 5]
            accepted_batches.sort()
            
            if not accepted_batches:
                print(f"    No batches passed threshold for {pipe}. Skipping final evaluation.")
                continue
                
            print(f"    Merging {len(accepted_batches)} accepted batches for final {pipe} evaluation...")
            all_eeg = []
            all_labels = []
            all_aux = []
            has_aux = False
            
            for b_idx in accepted_batches:
                data = np.load(batch_files[b_idx])
                all_eeg.append(data['eeg'])
                all_labels.append(data['labels'])
                if 'aux' in data:
                    all_aux.append(data['aux'])
                    has_aux = True
                    
            eeg_merged = np.concatenate(all_eeg, axis=0)
            labels_merged = np.concatenate(all_labels, axis=0)
            
            save_data = {"eeg": eeg_merged, "labels": labels_merged}
            if has_aux and len(all_aux) == len(accepted_batches):
                save_data["aux"] = np.concatenate(all_aux, axis=0)
                
            pipe_merged_path = path / f"temp_merged_{pipe}.npz"
            np.savez(pipe_merged_path, **save_data)
            
            print(f"    Training final {pipe} model on merged data...")
            try:
                _, final_report = train_model(
                    data_path=pipe_merged_path,
                    pipeline_name=pipe,
                    save_dir="temp",
                    use_grid=False,
                    eeg_device="Unicorn",
                    force_car=True,
                    no_save=True
                )
                final_acc = final_report.get("accuracy", 0)
                final_auc = final_report.get("auc", 0)
                if np.isnan(final_acc): final_acc = 0.0
                if np.isnan(final_auc): final_auc = 0.0
                final_dataset_res[pipe] = {"accuracy": final_acc, "auc": final_auc}
                print(f"      Final Accuracy: {final_acc:.4f} | AUC: {final_auc:.4f}")
            except Exception as e:
                print(f"      Error evaluating final {pipe}: {e}")
                
            try:
                os.remove(pipe_merged_path)
            except:
                pass

        final_results.append(final_dataset_res)

    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(10, 5))
    
    cmap = plt.get_cmap('tab10')
    dataset_colors = [cmap(i % 10) for i in range(len(DATA_DIRS))]
    
    for metric_key, ax, title in [("accuracy", ax_acc, "Accuracy"), ("auc", ax_auc, "ROC AUC")]:
        box_data = []
        box_labels = ["LR", "SVM", "MLP"]
        
        for pipe in pipelines:
            pipe_data = []
            for d_idx in range(len(DATA_DIRS)):
                res = final_results[d_idx]
                if res and pipe in res:
                    pipe_data.append(res[pipe][metric_key])
            if pipe_data:
                box_data.append(pipe_data)
                
        if box_data:
            bplot = ax.boxplot(box_data, patch_artist=True, tick_labels=box_labels)
            
            box_colors = ['darkorange', 'dodgerblue', 'forestgreen']
            for patch, color in zip(bplot['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
                
            for median in bplot['medians']:
                median.set(color='black', linewidth=1.5)
                
            for p_idx, pipe in enumerate(pipelines):
                box_idx = p_idx + 1
                for d_idx, ds_name in enumerate(dataset_names):
                    res = final_results[d_idx]
                    if res and pipe in res:
                        val = res[pipe][metric_key]
                        ax.scatter(box_idx, val, color='black', alpha=0.5,
                                   marker='o', zorder=3, s=30)
            
            if metric_key == "accuracy":
                ax.axhline(y=min_acc, color='gray', linestyle='-.', linewidth=2, label=f'Threshold ({min_acc})')
                ax.axhline(y=0.5, color='red', linestyle='-.', linewidth=2, label='Random (0.50)')
            else:
                ax.axhline(y=0.5, color='red', linestyle='-.', linewidth=2, label='Random (0.50)')
                
            ax.set_ylabel(title)
            ax.set_title(f"Final Model {title}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1.05)
            ax.legend(loc='lower right')

    plt.tight_layout()
    
    save_path = "directory_comparison_both.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")

if __name__ == "__main__":
    main()
