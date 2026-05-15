import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import train_model

def evaluate_dataset(data_dir, pipelines, min_acc=0.55):
    path = Path(data_dir)
    batch_files = sorted(glob.glob(str(path / "batch_*.npz")))
    if not batch_files:
        print(f"Warning: No batch_*.npz files found in {data_dir}")
        return {}
        
    print(f"\nProcessing Dataset: {path.name} with {len(batch_files)} batches")
    
    dataset_res = {pipe: {} for pipe in pipelines}
    for b_idx, batch_file in enumerate(batch_files):
        print(f"  Batch {b_idx + 1}/{len(batch_files)}: {Path(batch_file).name}")
        for pipe in pipelines:
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
                dataset_res[pipe][b_idx] = acc
            except Exception as e:
                dataset_res[pipe][b_idx] = 0.0
                
    final_dataset_res = {}
    for pipe in pipelines:
        sorted_batches = sorted(dataset_res[pipe].items(), key=lambda x: x[1], reverse=True)
        accepted_batches = [b for i, (b, acc) in enumerate(sorted_batches) if acc >= min_acc or i < 5]
        accepted_batches.sort()
        
        if not accepted_batches:
            final_dataset_res[pipe] = {"accuracy": 0.0, "auc": 0.0}
            continue
            
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
            print(f"    Final {pipe} Accuracy: {final_acc:.4f} | AUC: {final_auc:.4f}")
        except Exception as e:
            final_dataset_res[pipe] = {"accuracy": 0.0, "auc": 0.0}
            
        try:
            os.remove(pipe_merged_path)
        except:
            pass

    return final_dataset_res

def main():
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
    })

    pipelines = ["cov_ts_lr", "cov_ts_svm", "cov_ts_mlp", "cov_mdm", "csp_svm", "csp_lda", "csp_rf", "csp_mlp", "aug_ts_lr", "aug_ts_svm", "aug_ts_mlp"]
    min_acc = 0.55
    data_dir = r".\PythonBCI\data\raw\gold-data-2\stripped\subject_1"
    
    print(f"Evaluating {data_dir}...")
    res = evaluate_dataset(data_dir, pipelines, min_acc)

    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(12, 5))
    
    # fig.suptitle('Performance Across All Pipelines (subject_3)', fontweight='bold')
    
    metrics = [("accuracy", ax_acc, "Accuracy"), ("auc", ax_auc, "ROC AUC")]
    colors = plt.cm.tab10(np.linspace(0, 1, len(pipelines)))
    
    for metric_key, ax, title in metrics:
        data = [res.get(pipe, {}).get(metric_key, 0) for pipe in pipelines]
        
        bars = ax.bar(pipelines, data, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
        
        ax.set_xticks(range(len(pipelines)))
        ax.set_xticklabels(pipelines, rotation=45, ha="right")
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title, fontweight='bold')
        
        if metric_key == "accuracy":
            ax.axhline(0.5, ls='--', color='red', lw=1.5, zorder=5, label='Random')
            # ax.axhline(min_acc, ls='-.', color='gray', lw=1.5, zorder=5, label=f'Threshold ({min_acc})')
            ax.legend(loc='lower right', frameon=True)
        else:
            ax.axhline(0.5, ls='--', color='red', lw=1.5, zorder=5, label='Random')
            ax.legend(loc='lower right', frameon=True)
            
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    save_path = "subject_3_pipelines.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")

if __name__ == "__main__":
    main()
