import numpy as np
import pandas as pd

def main():
    try:
        data_file = input('Input file to be converted: ')
        data = np.load('PythonBCI\\data\\raw\\' + data_file)
        chunks = data['chunk']
        timestamps = data['timestamps']
        
        print(f"Loaded live_data.npz: chunks shape {chunks.shape}, timestamps shape {timestamps.shape}")
        
        columns = ['Timestamp'] + [f'CH{i+1}' for i in range(chunks.shape[1])]
        combined_data = np.column_stack((timestamps, chunks))
        df = pd.DataFrame(combined_data, columns=columns)

        data_file = data_file[:-3]
        output_file = 'PythonBCI/data/processed/output_data_' + data_file + '.csv'

        df.to_csv(output_file, index=False)
        print(f"Successfully exported data to {output_file}")
        
    except FileNotFoundError:
        print("Error: live_data.npz not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
