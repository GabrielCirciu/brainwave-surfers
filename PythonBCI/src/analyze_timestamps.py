import pandas as pd
import numpy as np

def analyze_timestamps(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'Timestamp' not in df.columns:
            print("Error: 'Timestamp' column not found in CSV.")
            return
            
        timestamps = df['Timestamp'].values
        diffs = np.diff(timestamps)
        # Sampling rate calculation
        mean_diff = np.mean(diffs)
        estimated_hz = 1.0 / mean_diff if mean_diff > 0 else 0
        # Check for gaps (standard is 4ms for 250Hz)
        expected_diff = 0.004
        jitter = np.abs(diffs - expected_diff)
        
        print(f"Timestamp Analysis for {file_path}")
        print(f"Total Samples:  {len(timestamps)}")
        print(f"Minimum Diff:   {np.min(diffs):.6f} s")
        print(f"Maximum Diff:   {np.max(diffs):.6f} s")
        print(f"Mean Diff:      {mean_diff:.6f} s")
        print(f"Median Diff:    {np.median(diffs):.6f} s")
        print(f"Max Jitter:     {np.max(jitter):.6f} s")
        print(f"Estimated Rate: {estimated_hz:.2f} Hz")

    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    csv_file = input("Enter the CSV file name: ")
    file_path = 'PythonBCI/data/processed/' + csv_file
    analyze_timestamps(file_path)
