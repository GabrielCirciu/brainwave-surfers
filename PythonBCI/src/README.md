# Scripts and Pipelines

The `src` directory contains the core scripts for data collection, model training, live signal evaluation, and the signal processing pipelines.

Within here there is also a `notebooks` directory that contains a collection of all of the notebooks that were created during the development of the project, and a `plots` directory that contains a collection of all of the plot scripts that were used to generate the plots in the article.

## Available Scripts

Here is a list of the main scripts used for running the BCI system and their functions.

### `build_model.py` - Offline Model Builder
Merges eeg batches and trains a BCI model. Use this to re-train a model from a saved dataset folder.
*   **Arguments:**
    *   `--data_dir` (Required): Directory containing `batch_*.npz` files.
    *   `--pipeline`: Pipeline to train (default: `all`).
    *   `--use_grid`: Use GridSearchCV for hyperparameter tuning.
    *   `--use_aux`: Use auxiliary channels along with EEG, specifically for the Unicorn Hybrid Black.
    *   `--save_dir`: Directory to save the trained model.
    *   `--eeg_device`: Device used: `Unicorn`, `hiAmp`, `gold` (default: `Unicorn`).
    *   `--resample_fs`: Resample the data to a specific sampling frequency (default: `250Hz`).
    *   `--visualize`: Generate preprocessing verification plots.
    *   `--force_car`: Force Common Average Reference instead of CSD.
    *   `--no_save`: Don't save individual model files (only final model.pkl).
    *   `--combined`: Train on all batches combined, ignoring individual batch performance.

### `calibrate.py` - Realtime Calibration (Superseded by `online_refine.py`)
Connects to the active LSL EEG stream and Unity markers stream to record training data during the calibration phase. Saves data in batches and automatically trains initial models. Run interactively and provide `session name` and `device` when prompted.

### `download_gold_data_2.py` - Gold Data Downloader
Downloads gold data from MOABB, and saves it in the `PythonBCI/data/raw/gold-data-2` directory. Little easter egg for whoever reads this. The reason why there is a 2 at the end is because the old version got data in a different way and we chose to throw that out as this one was the right format to be in line with our other datasets.

### `live_signal_check.py` - Live Signal Quality Check
Connects to the LSL EEG stream and provides a live terminal view of electrode contact quality, noise levels, and DC offsets to help ensure a good setup before recording. Also does preprocessing and gives you an evaluation of the data.

### `mock_eeg_stream.py` - Mock EEG Stream
Creates a simulated LSL EEG stream from gold data for the game. May no longer work and needs to be updated to fit new data.

### `online_refine.py` - Online Model Refinement
Combines gold training data with live trials to refine an existing dataset/model continuously.
*   **Arguments:**
    *   `--base_dir` (Required): Directory containing base training data.
    *   `--session_name` (Required): Name for this new refinement session.
    *   `--device`: EEG Device: `Unicorn`, `hiAmp`.
    *   `--use_aux`: Use auxiliary channels along with EEG.

### `realtime_predict.py` - Realtime Prediction
Connects to the LSL EEG stream and uses a trained model to make continuous predictions, sending them to Unity as input for the game.
*   **Arguments:**
    *   `--model_path`: Path to the trained `model.pkl` file (default: loads from `PythonBCI/models/model.pkl`).

### `train.py` - Single Batch Evaluation
Trains and evaluates a specific pipeline on a single `.npz` dataset.
*   **Arguments:**
    *   `--data` (Required): Path to the `.npz` data file. This would normally be a batch but should be a full set of combined .npz files into one.
    *   `--pipeline`: Pipeline name (e.g. `aug_ts_mlp`). Find out more below.
    *   `--save_dir`: Directory to save the models.
    *   `--use_grid`: Use GridSearchCV for hyperparameter tuning.
    *   `--use_aux`: Use auxiliary channels along with EEG, specifically for the Unicorn Hybrid Black.
    *   `--resample_fs`: Resample the data to a specific sampling frequency (default: `250Hz`).
    *   `--eeg_device`: Device used: `Unicorn`, `hiAmp`, `gold`.
    *   `--visualize`: Generate preprocessing plots.
    *   `--no_save`: Don't save individual model files (only final model.pkl).
    *   `--force_car`: Force Common Average Reference instead of CSD.

#### Training Pipelines
*   `cov_ts_lr` - Covariance + Tangent Space + Standard Scaler + Logistic Regression
*   `cov_ts_svm` - Covariance + Tangent Space + Standard Scaler + Support Vector Machine
*   `cov_ts_mlp` - Covariance + Tangent Space + Standard Scaler + Multi-Layer Perceptron
*   `cov_mdm` - Covariance + Minimum Distance to Mean
*   `csp_svm` - Common Spatial Patterns + Support Vector Machine
*   `csp_lda` - Common Spatial Patterns + Linear Discriminant Analysis
*   `csp_rf` - Common Spatial Patterns + Random Forest
*   `csp_mlp` - Common Spatial Patterns + Multi-Layer Perceptron
*   `aug_ts_lr` - Augmented Data + Tangent Space + Standard Scaler + Logistic Regression
*   `aug_ts_svm` - Augmented Data + Tangent Space + Standard Scaler + Support Vector Machine
*   `aug_ts_mlp` - Augmented Data + Tangent Space + Standard Scaler + Multi-Layer Perceptron

#### Data Processing Steps
1.  **Cropping**: Crops part of the data from start and end.
2.  **Notch Filter**: Removes 50hz noise.
3.  **Bandpass Filter**: Filters 8-30 Hz for motor imagery.
4.  **Common Average Reference (CAR)**: Removes common noise. (alternatively CSD)