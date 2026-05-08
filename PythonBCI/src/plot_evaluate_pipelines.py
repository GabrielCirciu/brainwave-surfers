import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from train import train_model

def main():
    """
    Evaluates multiple pipelines across batches in multiple datasets.
    Plots line plot across batches
    Plots boxplot across environments
    """

    parser = argparse.ArgumentParser(description="Evaluate multiple pipelines across batches in multiple datasets.")
    parser.add_argument("data_dirs", nargs='+', help="Paths to dataset folders containing batch_*.npz files.")
    parser.add_argument("--save_plot", default="Environments.pdf", help="Path to save the generated plot.")
    parser.add_argument("--metric", default="accuracy", choices=["accuracy", "auc"], help="Metric to evaluate and plot.")
    args = parser.parse_args()

    # LaTeX/Overleaf friendly font sizes
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

    # Store results: results[dataset_idx][pipeline_name][batch_idx] = accuracy
    # Using dictionary since some batches might be dropped or missing
    all_results = []
    final_results = []
    dataset_names = []

    for d_idx, data_dir in enumerate(args.data_dirs):
        path = Path(data_dir)
        dataset_names.append(path.name)
        
        batch_files = sorted(glob.glob(str(path / "batch_*.npz")))
        if not batch_files:
            print(f"Warning: No batch_*.npz files found in {data_dir}")
            all_results.append({})
            continue
            
        print(f"\nProcessing Dataset {d_idx + 1}/{len(args.data_dirs)}: {path.name} with {len(batch_files)} batches")
        
        dataset_res = {pipe: {} for pipe in pipelines}
        for b_idx, batch_file in enumerate(batch_files):
            print(f"  Batch {b_idx + 1}/{len(batch_files)}: {Path(batch_file).name}")
            for pipe in pipelines:
                print(f"    Evaluating {pipe}...")
                try:
                    # Unicorn headset, force_car=True gives the requested preprocessing
                    # We pass no_save=True to avoid cluttering models/
                    _, report = train_model(
                        data_path=batch_file,
                        pipeline_name=pipe,
                        save_dir="temp", # unused when no_save=True
                        use_grid=False,
                        eeg_device="Unicorn",
                        force_car=True,
                        no_save=True
                    )
                    acc = report.get(args.metric, 0)
                    if np.isnan(acc): acc = 0.0
                    
                    if acc >= min_acc:
                        dataset_res[pipe][b_idx] = acc
                        print(f"      {args.metric}: {acc:.4f} (Accepted)")
                    else:
                        print(f"      {args.metric}: {acc:.4f} (Dropped - below {min_acc})")
                        dataset_res[pipe][b_idx] = acc  # Keep the value so lines don't disappear
                except Exception as e:
                    print(f"      Error evaluating {pipe}: {e}")
                    dataset_res[pipe][b_idx] = 0.0  # Assign 0 or previous to keep line continuous
                    
        final_dataset_res = {}
        for pipe in pipelines:
            sorted_batches = sorted(dataset_res[pipe].items(), key=lambda x: x[1], reverse=True)
            accepted_batches = [b for i, (b, acc) in enumerate(sorted_batches) if acc >= min_acc or i < 5]
            accepted_batches.sort()
            
            if not accepted_batches:
                print(f"    No batches passed threshold for {pipe}. Skipping final evaluation.")
                final_dataset_res[pipe] = 0.0
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
                final_acc = final_report.get(args.metric, 0)
                if np.isnan(final_acc): final_acc = 0.0
                final_dataset_res[pipe] = final_acc
                print(f"      Final {args.metric}: {final_acc:.4f}")
            except Exception as e:
                print(f"      Error evaluating final {pipe}: {e}")
                final_dataset_res[pipe] = 0.0
                
            # Cleanup temp file
            try:
                os.remove(pipe_merged_path)
            except:
                pass

        final_results.append(final_dataset_res)
        all_results.append(dataset_res)

    # Plotting
    fig, (ax, ax_box) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [2.2, 1]})
    
    dataset_colors = ['darkorange', 'dodgerblue', 'forestgreen']
    
    box_data = []
    box_labels = []
        
    max_batch = 0
    for d_idx, dataset_res in enumerate(all_results):
        color = dataset_colors[d_idx % len(dataset_colors)]
        custom_labels = ["Controlled Env.", "Office Env.", "Home Env."]
        ds_name = custom_labels[d_idx % len(custom_labels)]
        
        all_batches = set()
        for pipe in pipelines:
            if dataset_res[pipe]:
                all_batches.update(dataset_res[pipe].keys())
                
        if not all_batches:
            continue
            
        batches = sorted(list(all_batches))
        max_batch = max(max_batch, max(batches))
        
        valid_x = []
        valid_y = []
        env_true_accuracies = []
        
        # Add a slight vertical offset so overlapping lines remain visible
        offset = d_idx * 0.005 
        
        for b in batches:
            accs = [dataset_res[pipe][b] for pipe in pipelines if b in dataset_res[pipe]]
            if accs:
                best_acc = max(accs)
                valid_x.append(b)
                valid_y.append(best_acc + offset)
                
        env_final_accuracies = [final_results[d_idx][pipe] for pipe in pipelines if final_results[d_idx].get(pipe, 0) > 0]
        if env_final_accuracies:
            box_data.append(env_final_accuracies)
            box_labels.append(ds_name)
                
        # Plot best model line for dataset
        ax.plot(valid_x, valid_y, color=color, linestyle='-', linewidth=2,
                label=f"{ds_name}")

    ax.axhline(y=min_acc, color='gray', linestyle='-.', label=f'Threshold ({min_acc})', linewidth=2)
    ax.axhline(y=0.5, color='red', linestyle='-.', label=f'Random (0.50)', linewidth=2)
    ax.set_xticks(range(max_batch + 1))
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Batch")
    ax.set_ylabel(args.metric.replace("_", " ").title())
    ax.set_title("Pipeline Performance Across Batches and Environments")
    
    # Legend in the line chart on the bottom left
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    # Plot Boxplot
    if box_data:
        bplot = ax_box.boxplot(box_data, patch_artist=True, tick_labels=box_labels)
        for patch, color in zip(bplot['boxes'], dataset_colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            
        ax_box.tick_params(axis='x', labelrotation=15)
            
        # Customize medians and whiskers
        for median in bplot['medians']:
            median.set(color='black', linewidth=1.5)
            
        # Add scatter dots to show individual model performance
        model_colors = {'aug_ts_lr': 'red', 'aug_ts_svm': 'purple', 'aug_ts_mlp': 'black'}
        model_markers = {'aug_ts_lr': 'o', 'aug_ts_svm': 's', 'aug_ts_mlp': '^'}
        model_labels = {'aug_ts_lr': 'LR', 'aug_ts_svm': 'SVM', 'aug_ts_mlp': 'MLP'}
        
        box_idx = 1
        for d_idx, dataset_res in enumerate(all_results):
            env_final_accuracies = [final_results[d_idx][pipe] for pipe in pipelines if final_results[d_idx].get(pipe, 0) > 0]
            if not env_final_accuracies:
                continue
                
            for pipe in pipelines:
                acc = final_results[d_idx].get(pipe, 0)
                if acc > 0:
                    ax_box.scatter(box_idx, acc, color=model_colors[pipe], marker=model_markers[pipe], 
                                 zorder=3, s=80, edgecolors='white', 
                                 label=model_labels[pipe] if box_idx == 1 else "")
            box_idx += 1
            
        ax_box.axhline(y=min_acc, color='gray', linestyle='-.', linewidth=2)
        ax_box.axhline(y=0.5, color='red', linestyle='-.', linewidth=2)
        ax_box.set_ylabel(args.metric.replace("_", " ").title())
        ax_box.set_title("Final Model Performance")
        ax_box.grid(True, alpha=0.3)
        ax_box.legend(loc='lower left', title="Models")
        # Ensure y-axis scales match
        ax_box.set_ylim(ax.get_ylim())

    plt.tight_layout()
    save_path = args.save_plot
    if args.metric != "accuracy" and save_path == "Environments.pdf":
        save_path = f"Environments_{args.metric.upper()}.pdf"
    plt.savefig(save_path)
    print(f"\nVisualization saved to {save_path}")

if __name__ == "__main__":
    main()
