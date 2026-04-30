import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_accuracy_trend(file_paths):
    plt.figure(figsize=(12, 8))
    
    for i, file_path in enumerate(file_paths):
        # Extract a short name from the folder name
        # e.g., kaje-26-04-29-13-26 -> kaje
        dir_name = os.path.basename(os.path.dirname(file_path))
        subject = dir_name.split('-')[0] if '-' in dir_name else dir_name
        
        df = pd.read_csv(file_path)
        
        # Sort by Batch to ensure chronological order
        df = df.sort_values('Batch')
        
        pipelines = df['Pipeline'].unique()
        
        for j, pipeline in enumerate(pipelines):
            pipeline_df = df[df['Pipeline'] == pipeline]
            label = f"{subject} - {pipeline}"
            plt.plot(pipeline_df['Batch'], pipeline_df['Accuracy'], 
                     linestyle='-',
                     linewidth=2,
                     label=label)

    plt.title('Model Accuracy Trend Over Time (Batches)', fontsize=16)
    plt.xlabel('Batch Number (-1 = Baseline, 0+ = Refined Batches)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(bottom=0.5)
    
    # Assume both files have similar batch numbers, extract all unique batches for x-ticks
    all_batches = sorted(df['Batch'].unique())
    plt.xticks(all_batches)
    
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(title='Subject & Pipeline', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    output_filename = 'accuracy_trend.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {os.path.abspath(output_filename)}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively (might be headless environment): {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot accuracy trend from multiple metrics.csv files")
    parser.add_argument('--files', nargs='+', help='List of metrics.csv file paths')
    args = parser.parse_args()

    if args.files:
        files_to_plot = args.files
    else:
        # Default files requested
        file1 = r"e:\GitHub\brainwave-surfers\PythonBCI\models\kaje-26-04-29-13-26\metrics.csv"
        file2 = r"e:\GitHub\brainwave-surfers\PythonBCI\models\vikt-26-04-29-14-07\metrics.csv"
        files_to_plot = [file1, file2]
        
    for f in files_to_plot:
        if not os.path.exists(f):
            print(f"Warning: File not found: {f}")
            
    plot_accuracy_trend(files_to_plot)
