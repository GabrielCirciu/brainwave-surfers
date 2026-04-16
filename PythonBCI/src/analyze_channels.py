import pandas as pd
import numpy as np

def analyze_channels(file_path):
    try:
        df = pd.read_csv(file_path)
        channels = df.iloc[:, 1:].values
        diffs = np.diff(channels)
        mean_diff = np.mean(diffs)
        print(f"Channels shape: {channels.shape}")
        print(f"Channel Analysis for {file_path}")
        print(f"Minimum Diff:   {np.min(diffs):.6f} s")
        print(f"Maximum Diff:   {np.max(diffs):.6f} s")
        print(f"Mean Diff:      {mean_diff:.6f} s")
        print(f"Median Diff:    {np.median(diffs):.6f} s")

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = 'data/processed/live_data_2.csv'
    analyze_channels(file_path)
