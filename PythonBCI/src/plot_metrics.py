import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_metrics(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter out 'Final' rows so we only plot numerical batches
    df_batches = df[df['Batch'] != 'Final'].copy()
    
    # Convert 'Batch' to numeric for correct sorting and plotting
    df_batches['Batch'] = pd.to_numeric(df_batches['Batch'])
    
    # Set up the plot
    plt.figure(figsize=(10, 6))
    
    # Get all unique models/pipelines
    pipelines = df_batches['Pipeline'].unique()
    
    # Plot a line for each model
    for pipeline in pipelines:
        pipeline_data = df_batches[df_batches['Pipeline'] == pipeline]
        # Sort by batch just in case it's not sorted
        pipeline_data = pipeline_data.sort_values('Batch')
        plt.plot(pipeline_data['Batch'], pipeline_data['Accuracy'], marker='o', label=pipeline)
    
    # Customize the graph
    plt.title('Model Accuracy over Batches')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.legend(title='Pipeline')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    csv_file = os.path.join("PythonBCI", "models", "myta-hiamp-26-04-24-11-57", "metrics.csv")
    # csv_file = "/Users/gabrielcirciu/Documents/ITU University Files/Semester 6/Project/Game/brainwave-surfers/PythonBCI/models/myta-hiamp-26-04-24-11-57/metrics.csv"
    plot_metrics(csv_file)
