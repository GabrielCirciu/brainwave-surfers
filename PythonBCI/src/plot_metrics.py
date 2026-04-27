import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import seaborn as sns

def plot_metrics(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Split data
    df_batches = df[df['Batch'] != 'Final'].copy()
    df_final = df[df['Batch'] == 'Final'].copy()
    
    # Convert 'Batch' to numeric
    df_batches['Batch'] = pd.to_numeric(df_batches['Batch'])
    pipelines = sorted(df_batches['Pipeline'].unique()) # Sort for consistency
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f'BCI Model Performance: {Path(csv_path).parent.name}', fontsize=16, fontweight='bold')
    
    # Define consistent colors for pipelines
    color_palette = sns.color_palette("husl", len(pipelines))
    pipe_colors = {pipe: color_palette[i] for i, pipe in enumerate(pipelines)}
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 6, figure=fig)
    
    # --- Subplot 1: Accuracy Trend ---
    ax1 = fig.add_subplot(gs[0, 0:3])
    for pipe in pipelines:
        pipe_data = df_batches[df_batches['Pipeline'] == pipe].sort_values('Batch')
        ax1.plot(pipe_data['Batch'], pipe_data['Accuracy'], marker='o', label=pipe, linewidth=2, color=pipe_colors[pipe])
    ax1.set_title('Accuracy Trend per Batch')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Accuracy')
    ax1.axhline(y=0.55, color='r', linestyle='--', alpha=0.3, label='Threshold (0.55)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Subplot 2: AUC Trend ---
    ax2 = fig.add_subplot(gs[0, 3:6])
    for pipe in pipelines:
        pipe_data = df_batches[df_batches['Pipeline'] == pipe].sort_values('Batch')
        ax2.plot(pipe_data['Batch'], pipe_data['AUC'], marker='s', label=pipe, linewidth=2, color=pipe_colors[pipe])
    ax2.set_title('AUC Trend per Batch')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('AUC')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # --- Subplot 3: Final Comparison Bar Chart (Centered & Narrower) ---
    ax3 = fig.add_subplot(gs[1, 1:5])
    
    # Prepare data: Group by Metric then Pipeline
    final_melted = df_final.melt(id_vars=['Pipeline'], value_vars=['Accuracy', 'AUC'], 
                                var_name='Metric', value_name='Score')
    
    sns.barplot(data=final_melted, x='Metric', y='Score', hue='Pipeline', ax=ax3, palette=pipe_colors)
    
    ax3.set_title('Final Model Comparison (Merged Data)')
    ax3.set_ylim(0, 1.0)
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend(title='Pipeline', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add value labels on bars
    for p in ax3.patches:
        height = p.get_height()
        if height > 0:
            ax3.annotate(f'{height:.3f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points',
                        fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot in the same directory as CSV
    plot_path = Path(csv_path).parent / "performance_summary.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot BCI metrics from CSV")
    parser.add_argument("--csv", type=str, help="Path to metrics.csv")
    args = parser.parse_args()
    
    if args.csv:
        plot_metrics(args.csv)
    else:
        # Try to find the latest metrics.csv in the models folder
        models_dir = Path("PythonBCI/models")
        csv_files = list(models_dir.glob("**/metrics.csv"))
        if not csv_files:
            print("No metrics.csv files found in PythonBCI/models")
            return
        
        # Sort by modification time
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        print(f"No CSV specified. Using latest: {latest_csv}")
        plot_metrics(str(latest_csv))

if __name__ == "__main__":
    main()
