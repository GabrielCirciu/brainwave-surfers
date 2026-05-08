import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import train_model

DATA_DIRS_22CH = [rf".\PythonBCI\data\raw\gold-data-2\full\subject_{i}" for i in range(1, 10)]
DATA_DIRS_8CH = [rf".\PythonBCI\data\raw\gold-data-2\stripped\subject_{i}" for i in range(1, 10)]

def evaluate_datasets(data_dirs, pipelines, min_acc=0.55):
    final_results = []
    
    for d_idx, data_dir in enumerate(data_dirs):
        path = Path(data_dir)
        
        batch_files = sorted(glob.glob(str(path / "batch_*.npz")))
        if not batch_files:
            print(f"Warning: No batch_*.npz files found in {data_dir}")
            final_results.append({})
            continue
            
        print(f"\nProcessing Dataset {d_idx + 1}/{len(data_dirs)}: {path.name} with {len(batch_files)} batches")
        
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
                pass
                
            try:
                os.remove(pipe_merged_path)
            except:
                pass

        final_results.append(final_dataset_res)
    return final_results

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
    
    print("Evaluating 22 Channel (Full) Data...")
    res_22 = evaluate_datasets(DATA_DIRS_22CH, pipelines, min_acc)
    
    print("\nEvaluating 8 Channel (Stripped) Data...")
    res_8 = evaluate_datasets(DATA_DIRS_8CH, pipelines, min_acc)

    # Plotting
    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(12, 5))
    
    metrics = [("accuracy", ax_acc, "Accuracy"), ("auc", ax_auc, "ROC AUC")]
    box_colors = {'22ch': 'dodgerblue', '8ch': 'darkorange'}
    
    for metric_key, ax, title in metrics:
        positions_22 = [p - 0.2 for p in range(1, len(pipelines) + 1)]
        positions_8 = [p + 0.2 for p in range(1, len(pipelines) + 1)]
        
        data_22 = [[r.get(pipe, {}).get(metric_key, 0) for r in res_22 if r and pipe in r] for pipe in pipelines]
        data_8 = [[r.get(pipe, {}).get(metric_key, 0) for r in res_8 if r and pipe in r] for pipe in pipelines]
        
        # 22ch boxes
        bp_22 = ax.boxplot(data_22, positions=positions_22, widths=0.3, patch_artist=True)
        for patch in bp_22['boxes']:
            patch.set_facecolor(box_colors['22ch'])
            patch.set_alpha(0.6)
        for median in bp_22['medians']:
            median.set(color='black', linewidth=1.5)
            
        # 8ch boxes
        bp_8 = ax.boxplot(data_8, positions=positions_8, widths=0.3, patch_artist=True)
        for patch in bp_8['boxes']:
            patch.set_facecolor(box_colors['8ch'])
            patch.set_alpha(0.6)
        for median in bp_8['medians']:
            median.set(color='black', linewidth=1.5)
            
        # Scatter dots
        for idx, (p22, p8) in enumerate(zip(positions_22, positions_8)):
            if data_22[idx]:
                ax.scatter([p22]*len(data_22[idx]), data_22[idx], color='black', alpha=0.5, s=30, zorder=3)
            if data_8[idx]:
                ax.scatter([p8]*len(data_8[idx]), data_8[idx], color='black', alpha=0.5, s=30, zorder=3)
        
        ax.set_xticks(range(1, len(pipelines) + 1))
        ax.set_xticklabels(["LR", "SVM", "MLP"])
        # ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title)
        
        if metric_key == "accuracy":
            ax.axhline(0.5, ls='--', color='red', lw=1.5, zorder=5, label='Random')
        else:
            ax.axhline(0.5, ls='--', color='red', lw=1.5, zorder=5, label='Random')
            
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Patch(facecolor=box_colors['22ch'], alpha=0.6, label='22 Channels'),
            Patch(facecolor=box_colors['8ch'], alpha=0.6, label='8 Channels'),
            Line2D([0], [0], color='red', ls='--', lw=1.5, label='Random')
        ]
            
        ax.legend(handles=legend_elements, loc='lower right', frameon=True)

    plt.tight_layout()
    save_path = "compare_channels.pdf"
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nVisualization saved to {save_path}")

if __name__ == "__main__":
    main()
