# Plots for Evaluation and Analysis

The `src/plots` directory contains scripts for evaluating pipeline performance and generating visualizations.

## Available Scripts

Here is a list of visualization scripts that will display different aspects of the pipeline performance and generate PDF files.

### `plot_all_subjects_boxplots.py` - All Subjects Boxplots
Evaluates all 9 subjects of the gold dataset across all pipelines using the stripped (8-channel) data. Generates boxplots of Accuracy and ROC AUC (`all_subjects_pipelines_boxplots.pdf`).


### `plot_compare_channels.py` - 22-Channel vs 8-Channel
Compares the performance of 22-channel (full) vs. 8-channel (stripped) data using LR, SVM, and MLP pipelines. Generates comparison boxplots (`compare_channels.pdf`). Update the `DATA_DIRS_22CH` and `DATA_DIRS_8CH` variables in the script to point to different datasets.

### `plot_compare_directories.py` - Cross-Environment Comparison
Compares pipeline performance across different dataset directories (environments/sessions). Generates boxplots of accuracy and AUC (`directory_comparison_both.pdf`). Update the `DATA_DIRS` array at the top of the script with your own paths.

### `plot_evaluate_pipelines.py` - Pipeline Evaluation
Evaluates pipelines across batches in multiple provided datasets. Generates a line plot of batch performance and a boxplot of final performance.
*   **Arguments:**
    *   `data_dirs` (Positional): Paths to dataset folders containing `batch_*.npz` files.
    *   `--save_plot`: Path to save the generated plot (default: `scores_environemnts.pdf`).
    *   `--metric`: Metric to evaluate and plot, choices are `accuracy` or `auc` (default: `accuracy`).

### `plot_score_distribution.py` - Theoretical Score Distribution
Plots the theoretical score distributions for the obstacle-avoiding game based on varying accuracy levels using a negative binomial distribution (`score_distribution.pdf`). Hardcoded `accuracies` and `lives` variables can be modified inside the script.

### `plot_scores.py` - Realtime Scores Trend
Reads `scores.csv` and plots the trend of actual player scores versus a simulated standard across multiple rounds (`scores_trend.pdf`). Hardcoded `csv_path` can be modified inside the script.

### `plot_subject_pipelines.py` - Single Subject Pipelines
Evaluates a specific single dataset/subject across all pipelines. Generates a bar chart comparing their Accuracy and ROC AUC (`subject_3_pipelines.pdf`). Hardcoded `data_dir` can be modified inside the script.

### `plot_visualize_erd.py` - PSD and ERD Visualization
Generates publication-quality plots of the Power Spectral Density (PSD) and time-series contrasts (e.g., C3 vs. C4) to visualize Event-Related Desynchronization (ERD) and signal quality. 
*   **Arguments:**
    *   `--dir` (Required): Directory containing `.npz` data batches.
    *   `--save`: Path to save the generated plot (default: `erd_visualization.pdf`).
    *   `--device`: EEG Device used, choices are `Unicorn`, `hiAmp` (default: `Unicorn`).