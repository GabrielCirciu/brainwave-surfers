import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

def apply_filters(data, fs=250.0):
    # 1. Demean (Subtract the DC offset before filtering to avoid edge transients)
    data = data - np.mean(data)
    
    # 2. Notch Filter (Remove 50Hz power line noise)
    # Most EEG software uses 50Hz for Europe/Asia or 60Hz for US.
    # We'll use 50Hz as it's the most common default for g.tec.
    f0 = 50.0
    # Quality factor, which is the ratio of the center frequency to the bandwidth, controlling narrowness of the notch.  
    Q = 30.0   
    b_notch, a_notch = iirnotch(f0, Q, fs)
    data = filtfilt(b_notch, a_notch, data)
    
    # 3. Bandpass Filter (1Hz - 30Hz), for MI this is the most important range of brain activity
    # This range captures the most important brain activity (Alpha/Beta)
    # and removes very slow drifts and high-frequency muscle noise.
    lowcut = 1.0
    highcut = 30.0
    nyq = 0.5 * fs
    b_band, a_band = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    data = filtfilt(b_band, a_band, data)
    
    return data

def apply_car(df, channels):
    """
    Common Average Reference (CAR):
    Subtracts the global average of all channels from each channel.
    This is excellent for removing common-mode noise.
    """
    channel_data = df[channels].values
    global_average = np.mean(channel_data, axis=1, keepdims=True)
    df[channels] = channel_data - global_average
    return df

def process_csv(input_path, output_path):
    print(f"Reading {input_path}...")
    try:
        df = pd.read_csv(input_path)
        fs = 250.0
        channels = [f'CH{i+1}' for i in range(8)]
        
        # Check if all channels exist
        existing_channels = [c for c in channels if c in df.columns]
        
        # Create a new DataFrame for adjusted data
        df_adjusted = df.copy()
        
        print("Applying filters (Demean -> 50Hz Notch -> 1-30Hz Bandpass)...")
        for ch in existing_channels:
            df_adjusted[ch] = apply_filters(df[ch].values, fs)
            
        print("Applying Common Average Reference (CAR)...")
        df_adjusted = apply_car(df_adjusted, existing_channels)
        
        # Save to new CSV
        df_adjusted.to_csv(output_file, index=False)
        print(f"Successfully saved EEG data to: {output_path}")
        
        # Summary report
        print("\nFinal Statistics")
        for ch in existing_channels[:2]: # Show first two as examples
            mean_val = df_adjusted[ch].mean()
            std_val = df_adjusted[ch].std()
            max_val = df_adjusted[ch].max()
            print(f"{ch}: Mean={mean_val:.4f} µV, StdDev={std_val:.2f} µV, Max={max_val:.2f} µV")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import os
    input_file = 'live_data.csv'
    output_file = 'live_data_adjusted.csv'
    
    if os.path.exists(input_file):
        process_csv(input_file, output_file)
    else:
        print(f"Error: {input_file} not found. Please run the live monitor first.")
