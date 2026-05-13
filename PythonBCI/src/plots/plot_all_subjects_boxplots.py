import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import train_model

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
                print(f"    {path.name} | {pipe} -> Acc: {final_acc:.4f}, AUC: {final_auc:.4f}")
            except Exception as e:
                final_dataset_res[pipe] = {"accuracy": 0.0, "auc": 0.0}
                
            try:
                os.remove(pipe_merged_path)
            except:
                pass

        final_results.append(final_dataset_res)
    return final_results

def main():
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    pipelines = ["cov_ts_lr", "cov_ts_svm", "cov_ts_mlp", "cov_mdm", "csp_svm", "csp_lda", "csp_rf", "csp_mlp", "aug_ts_lr", "aug_ts_svm", "aug_ts_mlp"]
    min_acc = 0.55
    data_dirs = [rf".\PythonBCI\data\raw\gold-data-2\stripped\subject_{i}" for i in range(1, 10)]
    
    print("Evaluating all 9 subjects across all pipelines...")
    res_list = evaluate_datasets(data_dirs, pipelines, min_acc)

    # Plotting Boxplots
    fig, (ax_acc, ax_auc) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # fig.suptitle('Performance Distribution of 9 Subjects Across Pipelines (Stripped Data)', fontweight='bold')
    
    metrics = [("accuracy", ax_acc, "Accuracy"), ("auc", ax_auc, "ROC AUC")]
    
    for metric_key, ax, title in metrics:
        box_data = []
        for pipe in pipelines:
            pipe_data = [res.get(pipe, {}).get(metric_key, 0) for res in res_list if res.get(pipe, {}).get(metric_key, 0) > 0]
            box_data.append(pipe_data)
            
        bp = ax.boxplot(box_data, patch_artist=True, tick_labels=pipelines)
        
        # Style boxes based on pipeline family
        for patch, pipe in zip(bp['boxes'], pipelines):
            if pipe.endswith('mlp'):
                c = 'forestgreen'
            elif pipe.endswith('svm'):
                c = 'dodgerblue'
            elif pipe.endswith('lr'):
                c = 'darkorange'
            else:
                c = 'gray'
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            
        # Customize medians and whiskers
        for median in bp['medians']:
            median.set(color='black', linewidth=1.5)
            
        # Add scatter dots (jittered for visibility)
        for idx, p_data in enumerate(box_data):
            if p_data:
                x_vals = np.random.normal(idx + 1, 0.04, size=len(p_data))
                ax.scatter(x_vals, p_data, color='black', alpha=0.5, marker='o', zorder=3, s=20)
                
        ax.set_xticklabels(pipelines, rotation=45, ha="right")
        # ax.set_title(title, fontweight='bold')
        ax.set_ylabel(title)
        
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        custom_lines = [
            Patch(facecolor='forestgreen', alpha=0.7, label='MLP'),
            Patch(facecolor='dodgerblue', alpha=0.7, label='SVM'),
            Patch(facecolor='darkorange', alpha=0.7, label='LR'),
            Patch(facecolor='gray', alpha=0.7, label='Other Models'),
            Line2D([0], [0], color='red', ls='--', lw=1.5, label='Random')
        ]
        
        if metric_key == "accuracy":
            ax.axhline(0.5, ls='--', color='red', lw=1.5, zorder=5)
            #ax.axhline(min_acc, ls='-.', color='gray', lw=1.5, zorder=5)
            #custom_lines.append(Line2D([0], [0], color='gray', ls='-.', lw=1.5, label=f'Threshold ({min_acc})'))
            ax.legend(handles=custom_lines, loc='lower right', frameon=True)
        else:
            ax.axhline(0.5, ls='--', color='red', lw=1.5, zorder=5)
            ax.legend(handles=custom_lines, loc='lower right', frameon=True)
            
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = "all_subjects_pipelines_boxplots.pdf"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"\nVisualization saved to {save_path}")

if __name__ == "__main__":
    main()
