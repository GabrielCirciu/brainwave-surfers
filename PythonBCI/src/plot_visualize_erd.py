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
        
    # Concatenate all batches
    epochs_data = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Transpose to MNE format: (epochs, channels, samples)
    epochs_data = np.transpose(epochs_data, (0, 2, 1))
    
    # 1. Device-specific Channel Mapping and Scaling
    eeg_ch_count = 8
    base_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    
    if eeg_device == "Unicorn":
        # Scale only the EEG channels
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
    
    # Notch Filter
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
    
    # Bandpass Filter
    epochs.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False) 
    
    # 2. Artifact Rejection (Cleans up the PSD significantly)
    print("Applying Artifact Rejection (150µV threshold)...")
    epochs.drop_bad(reject=dict(eeg=150e-6), verbose=False)
    print(f"Remaining trials: {len(epochs)}")
    
    if len(epochs) == 0:
        print("Error: All epochs dropped during artifact rejection.")
        return None, None, None
    
    # Set montage for both to ensure spatial colors and head legends work in PSD
    try:
        epochs.set_montage("standard_1020")
    except Exception as e:
        print(f"Warning: Could not set montage: {e}")

    # Save a copy before CSD for PSD visualization (to avoid unit errors)
    epochs_raw = epochs.copy()
    
    # Spatial Filter: CSD
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
            "font.family": "serif",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        })

        # 1. Prepare data for visualization
        epochs_viz = epochs.copy().crop(tmin=0.1, tmax=3.9)
        epochs_psd_viz = epochs_raw.copy()
        
        # Base save name
        base_name = save_path.replace('.pdf', '').replace('.png', '') if save_path else "erd_visualization"

        # --- FIGURE 1: Power Spectral Density ---
        # 1. Plot PSD exactly like train.py (with spatial colors and head legend)
        # MNE automatically scales the plot to µV²/Hz, so we do NOT scale manually here.
        fig_psd = plt.figure(figsize=(10, 6))
        ax_psd = fig_psd.add_subplot(1, 1, 1)
        epochs_raw.compute_psd(fmin=1, fmax=45).plot(axes=ax_psd, show=False)
        ax_psd.set_title("Power Spectral Density")
        ax_psd.set_ylabel("Power (µV²/Hz [dB])")
        ax_psd.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        psd_path = f"{base_name}_psd.pdf"
        fig_psd.savefig(psd_path, bbox_inches='tight')
        print(f"PSD plot saved to: {psd_path}")
        plt.close(fig_psd)

        # --- FIGURE 2: Time Series Contrast ---
        fig_time = plt.figure(figsize=(10, 6))
        ax_time = fig_time.add_subplot(1, 1, 1)
        
        picks = [ch for ch in ["C3", "C4"] if ch in epochs_viz.ch_names]
        if not picks: picks = [epochs_viz.ch_names[1]]
        
        sample_idx = 0
        data = epochs_viz.get_data(picks=picks) 
        times = epochs_viz.times
        
        for i, ch_name in enumerate(picks):
            display_data = data[sample_idx, i, :] * 1e6
            ax_time.plot(times, display_data, label=ch_name, linewidth=1.5)
        
        label_text = "LEFT" if labels[sample_idx] == 0 else "RIGHT" if labels[sample_idx] == 1 else ""
        ax_time.set_title(f"Time Series Contrast: C3 and C4 ({label_text})")
        ax_time.legend(loc='upper right', frameon=True)
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude (µV)")
        ax_time.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        time_path = f"{base_name}_time.pdf"
        fig_time.savefig(time_path, bbox_inches='tight')
        print(f"Time-series plot saved to: {time_path}")
        plt.close(fig_time)

        # Still call show() if the user is running interactively, 
        # but we need to re-create for that or just show the last one.
        # Given it's for Overleaf, the files are most important.
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
