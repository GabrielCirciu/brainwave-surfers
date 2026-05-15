import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import glob
import argparse
from pathlib import Path
from mne.time_frequency import tfr_morlet

def load_and_preprocess(data_dir, eeg_device="Unicorn"):
    """
    Loads data from a directory and applies the preprocessing pipeline 
    used in train.py for visualization purposes.
    """
    npz_files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not npz_files:
        print(f"Error: No .npz files found in {data_dir}")
        return None, None

    all_eeg = []
    all_labels = []
    
    print(f"Found {len(npz_files)} batch files. Loading data...")
    for file in npz_files:
        data = np.load(file)
        all_eeg.append(data['eeg'])
        all_labels.append(data['labels'])
        
    epochs_data = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    epochs_data = np.transpose(epochs_data, (0, 2, 1))

    eeg_ch_count = 8
    base_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    
    if eeg_device == "Unicorn":
        if np.max(np.abs(epochs_data[:, :eeg_ch_count, :])) > 100.0:
            print("Scaling Unicorn EEG from raw units to µV...")
            epochs_data[:, :eeg_ch_count, :] /= 1000.0
        ch_names = base_names + [f"AUX {i-7}" for i in range(eeg_ch_count, epochs_data.shape[1])]
        ch_types = ["eeg"] * eeg_ch_count + ["misc"] * (epochs_data.shape[1] - eeg_ch_count)
    elif eeg_device == "hiAmp":
        epochs_data = epochs_data[:, [18, 27, 36, 29, 38, 22, 31, 40], :]
        ch_names = base_names
        ch_types = ["eeg"] * 8
    else:
        ch_names = [f"EEG {i+1}" for i in range(epochs_data.shape[1])]
        ch_types = ["eeg"] * epochs_data.shape[1]
    
    fs = epochs_data.shape[2] / 4
    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)
    
    for freq in [50]:
        epochs_data = mne.filter.notch_filter(
            epochs_data.astype(np.float64), 
            fs, 
            freq, 
            method='iir',
            verbose=False
        )
        
    events = np.column_stack((
        np.arange(len(labels)), 
        np.zeros(len(labels), dtype=int), 
        labels.astype(int)
    ))
    
    epochs = mne.EpochsArray(epochs_data * 1e-6, info, events=events, verbose=False)
    
    epochs.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False) 
    
    print("Applying Artifact Rejection (150µV threshold)...")
    epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
    print(f"Remaining trials: {len(epochs)}")
    
    if len(epochs) == 0:
        print("Error: All epochs dropped during artifact rejection.")
        return None, None, None
    
    try:
        epochs.set_montage("standard_1020")
    except Exception as e:
        print(f"Warning: Could not set montage: {e}")

    epochs_raw = epochs.copy()
    
    try:
        epochs = mne.preprocessing.compute_current_source_density(epochs)
    except Exception as e:
        print(f"Surface Laplacian failed ({e}), falling back to CAR...")
        epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)
        
    return epochs, labels, epochs_raw

def plot_psd_erd(epochs, labels, epochs_raw, save_path=None):
    """
    Generates a diagnostic plot that matches train.py's PSD exactly 
    and shows all channels in the time-series.
    """
    if epochs is None:
        return

    try:
        # LaTeX-friendly styling
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        })

        epochs_viz = epochs.copy().crop(tmin=0.1, tmax=3.9)
        epochs_psd_viz = epochs_raw.copy()
        
        base_name = save_path.replace('.pdf', '').replace('.png', '') if save_path else "erd_visualization"

        fig_psd = plt.figure(figsize=(10, 4))
        ax_psd = fig_psd.add_subplot(1, 1, 1)
        epochs_raw.compute_psd(fmin=1, fmax=45).plot(axes=ax_psd, show=False)
        
        channel_colors = [
            '#1F93FF', '#2385E5', '#2577CB', 
            '#505050', '#808080',            
            '#F54927', '#DD4424', '#C53E22'  
        ]
        
        channel_lines = [line for line in ax_psd.lines if len(line.get_xdata()) > 10]
        
        end_points = []
        line_idx = 0
        for line in channel_lines:
            line.set_linewidth(1.0)
            if line_idx < len(channel_colors) and line_idx < len(epochs_raw.ch_names):
                line.set_color(channel_colors[line_idx])
                
                x_data = line.get_xdata()
                y_data = line.get_ydata()
                if len(x_data) > 0 and len(y_data) > 0:
                    end_points.append({
                        'x': x_data[-1],
                        'y': y_data[-1],
                        'ch_name': epochs_raw.ch_names[line_idx],
                        'color': channel_colors[line_idx]
                    })
            line_idx += 1
            
        end_points.sort(key=lambda pt: pt['y'])
        min_dist = 1.5
        for _ in range(10):
            for i in range(len(end_points) - 1):
                if end_points[i+1]['y'] - end_points[i]['y'] < min_dist:
                    diff = min_dist - (end_points[i+1]['y'] - end_points[i]['y'])
                    end_points[i+1]['y'] += diff / 2
                    end_points[i]['y'] -= diff / 2
                    
        for pt in end_points:
            ax_psd.text(pt['x'] + 0.5, pt['y'], f" {pt['ch_name']}", color=pt['color'],
                        verticalalignment='center', fontsize=10, fontweight='bold')
                        
        lines_to_remove = [line for line in ax_psd.lines if line not in channel_lines]
        for line in lines_to_remove:
            line.remove()

        ax_psd.axvline(x=8, color='darkslategrey', linestyle=':', alpha=0.9, linewidth=1.5)
        ax_psd.axvline(x=30, color='darkslategrey', linestyle=':', alpha=0.9, linewidth=1.5)
        
        y_min, y_max = ax_psd.get_ylim()
        ax_psd.text(10, y_max - (y_max - y_min) * 0.02, "8 Hz", color='darkslategrey', fontsize=10, 
                    rotation=0, verticalalignment='top', horizontalalignment='center', fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        ax_psd.text(32, y_max - (y_max - y_min) * 0.02, "30 Hz", color='darkslategrey', fontsize=10, 
                    rotation=0, verticalalignment='top', horizontalalignment='center', fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
            
        for ax in fig_psd.axes:
            if ax != ax_psd:
                for collection in ax.collections:
                    collection.set_facecolor(channel_colors)
                    collection.set_edgecolor('none')
        ax_psd.set_title("Power Spectral Density")
        ax_psd.set_ylabel("Power (µV²/Hz [dB])")
        ax_psd.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        psd_path = f"{base_name}_psd.pdf"
        fig_psd.savefig(psd_path, bbox_inches='tight', dpi=300)
        print(f"PSD plot saved to: {psd_path}")
        plt.close(fig_psd)

        fig_time = plt.figure(figsize=(10, 4))
        ax_time = fig_time.add_subplot(1, 1, 1)
        
        picks = [ch for ch in ["C3", "C4"] if ch in epochs_viz.ch_names]
        if not picks: picks = [epochs_viz.ch_names[1]]
        
        sample_idx = 0
        data = epochs_viz.get_data(picks=picks) 
        times = epochs_viz.times
        
        for i, ch_name in enumerate(picks):
            display_data = data[sample_idx, i, :] * 1e6
            color = None
            if ch_name in epochs_raw.ch_names:
                ch_idx = epochs_raw.ch_names.index(ch_name)
                if ch_idx < len(channel_colors):
                    color = channel_colors[ch_idx]
            ax_time.plot(times, display_data, label=ch_name, linewidth=1.5, color=color)
        
        label_text = "LEFT" if labels[sample_idx] == 0 else "RIGHT" if labels[sample_idx] == 1 else ""
        ax_time.set_title(f"Time Series Contrast: C3 and C4 ({label_text} trial)")
        ax_time.legend(loc='upper right', frameon=True)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude (µV)")
        ax_time.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        time_path = f"{base_name}_time.pdf"
        fig_time.savefig(time_path, bbox_inches='tight', dpi=300)
        print(f"Time-series plot saved to: {time_path}")
        plt.close(fig_time)

    except Exception as e:
        print(f"Could not generate plot: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate publication-quality EEG plots (PDF for LaTeX).")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .npz data batches.")
    parser.add_argument("--save", type=str, default="erd_visualization.pdf", help="Path to save the generated plot (e.g. .pdf).")
    parser.add_argument("--device", type=str, default="Unicorn", help="EEG Device: Unicorn, hiAmp")
    args = parser.parse_args()
    
    data_dir = args.dir
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist.")
    else:
        epochs, labels, epochs_raw = load_and_preprocess(data_dir, eeg_device=args.device)
        if epochs:
            plot_psd_erd(epochs, labels, epochs_raw, save_path=args.save)
