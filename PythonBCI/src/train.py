import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_validate, StratifiedKFold, KFold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

AVAILABLE_PIPELINES = [
    "cov_ts_lr", "cov_ts_svm", "cov_ts_mlp", "cov_mdm", 
    "aug_ts_lr", "aug_ts_svm", "aug_ts_mlp",
    "csp_svm", "csp_lda", "csp_rf", "csp_mlp"
]


class AugmentedDataset(BaseEstimator, TransformerMixin):
    """Dataset augmentation via delay embedding. 
    Matches MOABB's AugmentedDataset used in benchmarks.
    """
    def __init__(self, order=2, lag=1):
        self.order = order
        self.lag = lag

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.order <= 1:
            return X
        # X is (n_trials, n_channels, n_samples)
        # Concatenate delayed versions along the channel dimension
        chunks = []
        for p in range(self.order):
            start = p * self.lag
            end = X.shape[2] - (self.order - 1 - p) * self.lag
            chunks.append(X[:, :, start:end])
        
        return np.concatenate(chunks, axis=1)



def load_and_preprocess(data_path, use_aux=False, resample_fs=False, eeg_device="Unicorn", force_car=False):
    """Loads and preprocesses BCI data."""
    data = np.load(data_path)
    # not sure what config we use, for some data - epochs, for some - eeg, so this oone works for both
    if "epochs" in data:
        epochs_data = data["epochs"]
    elif "eeg" in data:
        eeg = data["eeg"]
        if use_aux and "aux" in data:
            aux = data["aux"]
            if eeg.ndim == 3:
                if eeg.shape[1] > eeg.shape[2]:
                    eeg = np.concatenate([eeg, aux], axis=2)
                else:
                    eeg = np.concatenate([eeg, aux], axis=1)
            elif eeg.ndim == 2:
                if eeg.shape[0] > eeg.shape[1]:
                    eeg = np.concatenate([eeg, aux], axis=1)
                else:
                    eeg = np.concatenate([eeg, aux], axis=0)

        # Normalize to (trials, channels, samples)
        if eeg.ndim == 3:
            # common layouts: (trials, samples, channels) or (trials, channels, samples)
            if eeg.shape[1] > eeg.shape[2]:
                epochs_data = np.transpose(eeg, (0, 2, 1))
            else:
                epochs_data = eeg
        elif eeg.ndim == 2:
            # single-trial: try to infer orientation
            if eeg.shape[0] > eeg.shape[1]:
                # assume (samples, channels)
                epochs_data = eeg.T[np.newaxis, ...]
            else:
                # assume (channels, samples)
                epochs_data = eeg[np.newaxis, ...]
    else:
        raise KeyError("NPZ file must contain either 'epochs' or 'eeg' arrays")

    if eeg_device == "hiAmp":
        epochs_data = epochs_data[:, [18, 27, 36, 29, 38, 22, 31, 40], :]
        print(f"hiAmp device detected. Selected motor-strip channels.")
        base_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
        ch_names = base_names + [f"EEG {i+1}" for i in range(len(base_names), epochs_data.shape[1])]
    elif eeg_device in ["gold", "BNCI2014_001"]:
        print(f"Gold/BNCI dataset detected. Channels: {epochs_data.shape[1]}")
        # Support both the 8-channel restricted set and the 22-channel full set
        if epochs_data.shape[1] == 8:
            base_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
        elif epochs_data.shape[1] == 22:
            # Standard BNCI2014_001 / Competition IV 2a channel order
            base_names = [
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz',
                'C2', 'C4', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz',
                'P2', 'POz', 'O1', 'O2'
            ]
        else:
            base_names = [f"EEG {i+1}" for i in range(epochs_data.shape[1])]
        
        ch_names = base_names + [f"EEG {i+1}" for i in range(len(base_names), epochs_data.shape[1])]
    elif eeg_device == "Unicorn":
        print("Unicorn device detected. Using all channels.")
        # We check if the amplitude is suspiciously large, if so - divide by 1000.
        if np.max(np.abs(epochs_data)) > 1e-4:
            print("Scaling Unicorn data from Volts to Microvolts...")
            epochs_data = epochs_data / 1000
        base_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
        ch_names = base_names + [f"EEG {i+1}" for i in range(len(base_names), epochs_data.shape[1])]
    elif eeg_device == "gold":
        print("Gold dataset (Schirrmeister2017) detected.")
        # Schirrmeister2017 data from MOABB is in Volts, but our pipeline expects Microvolts.
        # We check if the amplitude is suspiciously small.
        if np.max(np.abs(epochs_data)) < 1e-3: 
            print("Scaling gold data from Volts to Microvolts...")
            epochs_data = epochs_data * 1e6
        base_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
        ch_names = base_names + [f"EEG {i+1}" for i in range(len(base_names), epochs_data.shape[1])]
    else:
        print(f"Device {eeg_device} detected. Using all channels.")
        ch_names = [f"EEG {i+1}" for i in range(epochs_data.shape[1])]
        
    labels = data["labels"]
    # gather how many rows in a sample, divide it by 4 and that is the fs
    fs = epochs_data.shape[2] / 4
    print(f"Sample frequency: {fs} Hz")

    print(f"Loaded data shape: {epochs_data.shape} (Trials, Channels, Samples)")

    ch_types = ["eeg"] * epochs_data.shape[1]
    
    if "eeg" in data and use_aux and "aux" in data:
        orig_eeg = data["eeg"]
        if orig_eeg.ndim == 3:
            n_orig_eeg = orig_eeg.shape[2] if orig_eeg.shape[1] > orig_eeg.shape[2] else orig_eeg.shape[1]
        elif orig_eeg.ndim == 2:
            n_orig_eeg = orig_eeg.shape[1] if orig_eeg.shape[0] > orig_eeg.shape[1] else orig_eeg.shape[0]
        n_channels = epochs_data.shape[1]
        for i in range(n_orig_eeg, n_channels):
            ch_names[i] = f"AUX {i - n_orig_eeg + 1}"
            ch_types[i] = "misc" # AUX channels (gyro/accel) are misc type

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    print("Applying Notch Filter (50 Hz)...")
    # Apply notch filter to numpy data before creating EpochsArray
    # Using IIR method in a loop to handle multiple harmonics without filter length issues
    for freq in np.arange(50, fs / 2, 50):
        epochs_data = mne.filter.notch_filter(
            epochs_data.astype(np.float64), 
            fs, 
            freq, 
            method='iir',
            verbose=False
        )
    
    # Use events to keep labels synchronized if we drop epochs
    events = np.column_stack((
        np.arange(len(labels)), 
        np.zeros(len(labels), dtype=int), 
        labels.astype(int)
    ))
    
    epochs = mne.EpochsArray(epochs_data * 1e-6, info, events=events, verbose=False)
    
    if resample_fs:
        print(f"Resampling from {fs} Hz to 250.0 Hz...")
        epochs = epochs.resample(250.0, verbose=False)
        fs = 250.0

    print("Filtering data (8-15 Hz)...")
    epochs.filter(8.0, 15.0, picks=["eeg", "misc"], fir_design="firwin", skip_by_annotation="edge", verbose=False) 

    # Log amplitude to help debug scaling issues
    max_amp = np.max(np.abs(epochs.get_data(picks='eeg'))) * 1e6
    print(f"Max signal amplitude after filtering: {max_amp:.2f} uV")

    if eeg_device in ["gold", "BNCI2014_001"]:
        print("Gold/BNCI device detected. Filtering (4-40 Hz) for potentially better performance...")
        epochs.filter(4.0, 40.0, picks="eeg", fir_design="firwin", verbose=False)

    if eeg_device == "hiAmp":
        print("Applying Artifact Rejection (100mV)...")
        # Massive threshold (100mV) because your data is showing ~9mV peaks.
        # This will allow visualization so you can see what's happening.
        epochs.drop_bad(reject=dict(eeg=100000e-6), verbose=False)
        print(f"Remaining trials: {len(epochs)}")
        
        if len(epochs) == 0:
            print("WARNING: All epochs were dropped! Visualization will be skipped.")
    
    if len(epochs) > 0:
        if not force_car and eeg_device in ["Unicorn", "hiAmp", "gold", "BNCI2014_001"]:
            print(f"Applying Surface Laplacian (CSD) for {eeg_device}...")
            try:
                epochs.set_montage("standard_1020")
                epochs = mne.preprocessing.compute_current_source_density(epochs)
            except Exception as e:
                print(f"Surface Laplacian failed ({e}), falling back to CAR...")
                epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)
        else:
            print("Applying Common Average Reference (CAR)...")
            epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)

        # Log amplitude AFTER spatial reference to see improvement
        max_amp_post = np.max(np.abs(epochs.get_data(picks=['eeg', 'csd']))) * 1e6
        print(f"Max signal amplitude after {('CAR' if force_car else 'CSD')}: {max_amp_post:.2f} uV")

    if len(epochs) > 0:
        epochs.crop(tmin=0.5) #!TODO maybe make this one parametric?
    else:
        print("No epochs remaining to crop.")

    return epochs, fs


def plot_verification(epochs, save_dir, model_name):
    """Generates a diagnostic plot to verify filters and spatial resolution."""
    if len(epochs) == 0:
        print("Skipping verification plot: No epochs remaining after artifact rejection.")
        return

    try:
        fig = plt.figure(figsize=(12, 10))
        
        # 1. Plot PSD to verify filters (8-30Hz and 50Hz Notch)
        ax_psd = fig.add_subplot(2, 1, 1)
        epochs.compute_psd(fmin=1, fmax=60).plot(axes=ax_psd, show=False)
        ax_psd.set_title(f"PSD: Verify 8-30Hz Bandpass and 50Hz Notch ({model_name})")

        # 2. Plot time-series for a few motor channels to verify CSD/Cleaning
        ax_time = fig.add_subplot(2, 1, 2)
        # Pick C3 (index 1) and C4 (index 6) if they exist
        picks = [ch for ch in ["C3", "C4"] if ch in epochs.ch_names]
        if not picks: picks = [epochs.ch_names[1]]
        
        data = epochs.get_data(picks=picks)
        times = epochs.times
        for i, ch_name in enumerate(picks):
            ax_time.plot(times, data[0, i, :], label=f"Trial 1 - {ch_name}")
        
        ax_time.set_title("Time Series: Cleaned Motor Channels (Trial 1)")
        ax_time.legend()
        ax_time.set_xlabel("Time (s)")
        ax_time.set_ylabel("Amplitude (V)")
        
        report_path = Path(save_dir) / f"{model_name}_verification.png"
        latest_path = Path(save_dir) / "latest_verification.png"
        
        plt.savefig(report_path)
        plt.savefig(latest_path) # Also save as a fixed name for easy access
        plt.close(fig)
        print(f"Verification plots saved to: {report_path} and {latest_path}")
    except Exception as e:
        print(f"Could not generate verification plot: {e}")


def get_pipeline_and_grid(name):
    # Covariance, Tangent Space, Logistic Regression
    if name == "cov_ts_lr":
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=42)),
        ])
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "ts__metric": ["riemann", "logeuclid"],
            "clf__C": [0.1, 1.0, 10.0, 100.0],
        }

    # Covariance, Tangent Space, Support Vector Machine (MOABB Style)
    elif name == "cov_ts_svm":
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42)),
        ])
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "ts__metric": ["riemann", "logeuclid"],
            "clf__kernel": ["rbf", "linear"],
            "clf__C": [0.1, 1.0, 10.0, 100.0],
            "clf__gamma": ["scale", "auto", 0.1, 0.01],
        }

    # Covariance, Tangent Space, Multi-Layer Perceptron
    elif name == "cov_ts_mlp":
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(max_iter=5000, solver="lbfgs", random_state=42)),
        ])
        param_grid = {
            "cov__estimator": ["oas"],
            "ts__metric": ["riemann"],
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__alpha": [0.0001, 0.001],
            "clf__activation": ["relu", "tanh"]
        }

    # Covariance, Minimum Distance to Mean (Riemannian)
    elif name == "cov_mdm":
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("clf", MDM(metric="riemann")),
        ])
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "clf__metric": ["riemann", "logeuclid"],
        }
    
    # Common Spatial Patterns + Support Vector Machine
    elif name == "csp_svm":
        pipeline = Pipeline([
            ("csp", CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
            ("clf", SVC(probability=True, random_state=42)),
        ])
        param_grid = {
            "csp__n_components": [4, 6],
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1.0, 10.0],
        }

    # Common Spatial Patterns + Linear Discriminant Analysis
    elif name == "csp_lda":
        pipeline = Pipeline([
            ("csp", CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
            ("clf", LinearDiscriminantAnalysis()),
        ])
        param_grid = {
            "csp__n_components": [4, 6],
            "clf__solver": ["lsqr", "eigen"],
            "clf__shrinkage": ["auto", None],
        }

    # Common Spatial Patterns + Random Forest
    elif name == "csp_rf":
        pipeline = Pipeline([
            ("csp", CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
            ("clf", RandomForestClassifier(random_state=42)),
        ])
        param_grid = {
            "csp__n_components": [4, 6],
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 5, 10],
        }

    # Common Spatial Patterns + Multi-Layer Perceptron
    elif name == "csp_mlp":
        pipeline = Pipeline([
            ("csp", CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False)),
            ("clf", MLPClassifier(max_iter=2000, random_state=42)),
        ])
        param_grid = {
            "csp__n_components": [4, 6],
            "clf__hidden_layer_sizes": [(50,), (100,)],
            "clf__alpha": [0.0001, 0.001],
        }

    # Augmented Tangent Space + Logistic Regression
    elif name == "aug_ts_lr":
        pipeline = Pipeline([
            ("aug", AugmentedDataset(order=2)),
            ("cov", Covariances(estimator="lwf")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs", random_state=42)),
        ])
        param_grid = {
            "aug__order": [2],
            "cov__estimator": ["oas", "lwf"],
            "clf__C": [0.1, 1.0, 10.0],
        }

    # Augmented Tangent Space + SVM (The one the user wants)
    elif name == "aug_ts_svm":
        pipeline = Pipeline([
            ("aug", AugmentedDataset(order=2)),
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel='rbf', probability=True, random_state=42)),
        ])
        param_grid = {
            "aug__order": [2],
            "cov__estimator": ["oas", "lwf"],
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1.0, 10.0],
        }

    # Augmented Tangent Space + MLP
    elif name == "aug_ts_mlp":
        pipeline = Pipeline([
            ("aug", AugmentedDataset(order=2)),
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42)),
        ])
        param_grid = {
            "aug__order": [2],
            "clf__hidden_layer_sizes": [(50,), (100,)],
        }
    
    else:
        raise ValueError(f"pipeline must be one of: {AVAILABLE_PIPELINES}")

    return pipeline, param_grid


def train_model(data_path, pipeline_name="cov_ts_lr", save_dir="../models", use_grid=False, use_aux=False, resample_fs=False, eeg_device="", visualize=False, no_save=False, force_car=False):
    epochs, fs = load_and_preprocess(data_path, use_aux=use_aux, resample_fs=resample_fs, eeg_device=eeg_device, force_car=force_car)
    
    X = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    
    pipeline, param_grid = get_pipeline_and_grid(pipeline_name)

    # Handle tiny datasets: cross-validation can fail when train folds contain
    # only a single class (common when there are very few trials).
    n_samples = X.shape[0]
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Need at least two classes in the labels to train.")
    min_class_count = counts.min()

    scoring = {
        "auc": "roc_auc",
        "accuracy": "accuracy",
    }

    # Decide whether to run cross-validation/grid search based on data size
    run_cv = True
    if n_samples < 6 or min_class_count < 2:
        print("Warning: small dataset or insufficient class samples — skipping CV/grid and fitting on full data.")
        run_cv = False

    # Use 5-fold cross-validation with an 80/20 split as requested
    if run_cv and min_class_count >= 2:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print(f"Pipeline: {pipeline_name}")
    print(f"Grid search: {use_grid}")

    old_log_level = mne.get_config("MNE_LOGGING_LEVEL", "INFO")
    mne.set_log_level("ERROR")

    try:
        if run_cv and use_grid:
            n_candidates = np.prod([len(v) for v in param_grid.values()])
            print(f"Running GridSearchCV with {cv.get_n_splits()} folds over {n_candidates} candidates...")
            search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit="auc",
                cv=cv,
                n_jobs=1,
                verbose=1,
                return_train_score=False,
            )
            search.fit(X, y)

            best_model = search.best_estimator_
            best_idx = search.best_index_
            auc_mean = float(search.cv_results_["mean_test_auc"][best_idx])
            acc_mean = float(search.cv_results_["mean_test_accuracy"][best_idx])
            best_params = search.best_params_

        elif run_cv and not use_grid:
            print("Running cross-validation without grid search...")
            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                return_train_score=False,
            )
            pipeline.fit(X, y)

            best_model = pipeline
            auc_mean = float(np.mean(scores["test_auc"]))
            acc_mean = float(np.mean(scores["test_accuracy"]))
            best_params = {}

        else:
            # Small dataset: skip CV/grid and fit on full data
            pipeline.fit(X, y)
            best_model = pipeline
            auc_mean = float('nan')
            acc_mean = float('nan')
            best_params = {}

    finally:
        mne.set_log_level(old_log_level)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{pipeline_name}_{timestamp}"
    
    if visualize:
        plot_verification(epochs, save_dir, model_name)

    model_path = save_dir / f"{model_name}.pkl"
    report_path = save_dir / f"{model_name}_report.json"

    report = {
        "model_name": model_name,
        "pipeline": pipeline_name,
        "use_grid": use_grid,
        "data_path": str(data_path),
        "fs": fs,
        "auc": auc_mean,
        "accuracy": acc_mean,
        "best_params": {k: str(v) for k, v in best_params.items()},
        "model_path": str(model_path),
    }

    if not no_save:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"Model saved to: {model_path}")
        print(f"Report saved to: {report_path}")
    else:
        print("Model and report saving skipped (--no_save).")

    print(f"AUC: {auc_mean:.4f}")
    print(f"Accuracy: {acc_mean:.4f}")
    if best_params:
        print(f"Best params: {best_params}")

    return best_model, report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pipeline")
    parser.add_argument("--save_dir")
    parser.add_argument("--use_grid", action="store_true")
    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--resample_fs", default=False)
    parser.add_argument("--eeg_device", default="Unicorn", help="Device used: Unicorn, hiAmp, gold")
    parser.add_argument("--visualize", action="store_true", help="Generate preprocessing verification plots")
    parser.add_argument("--no_save", action="store_true", help="Don't save model and report files")
    parser.add_argument("--force_car", action="store_true", help="Force Common Average Reference instead of CSD")
    args = parser.parse_args()

    data_p = Path(args.data)
    if not args.save_dir:
        session_name = data_p.parent.name
        save_dir = Path(__file__).parents[1] / "models" / session_name
    else:
        save_dir = Path(args.save_dir)

    pipeline_choice = args.pipeline or "cov_ts_lr"
    
    if pipeline_choice == "all":
        print(f"Searching for absolute best model across: {AVAILABLE_PIPELINES}")
        best_overall_score = -1
        best_overall_report = None
        best_overall_name = ""
        
        for pipe_name in AVAILABLE_PIPELINES:
            try:
                _, report = train_model(
                    data_path=args.data,
                    pipeline_name=pipe_name,
                    save_dir=save_dir,
                    use_grid=args.use_grid,
                    use_aux=args.use_aux,
                    resample_fs=args.resample_fs,
                    eeg_device=args.eeg_device,
                    visualize=args.visualize,
                    no_save=args.no_save,
                    force_car=args.force_car,
                )
                
                # Compare based on AUC, fall back to Accuracy
                score = report.get("auc", 0) if not np.isnan(report.get("auc", 0)) else report.get("accuracy", 0)
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_name = pipe_name
                    best_overall_report = report
            except Exception as e:
                print(f"Pipeline {pipe_name} failed: {e}")
                
        if best_overall_report:
            print("\n" + "="*30)
            print(f"ABSOLUTE BEST MODEL: {best_overall_name}")
            print(f"Best AUC: {best_overall_report.get('auc', 0):.4f}")
            print(f"Best Accuracy: {best_overall_report.get('accuracy', 0):.4f}")
            print("="*30)
    else:
        train_model(
            data_path=args.data,
            pipeline_name=pipeline_choice,
            save_dir=save_dir,
            use_grid=args.use_grid,
            use_aux=args.use_aux,
            resample_fs=args.resample_fs,
            eeg_device=args.eeg_device,
            visualize=args.visualize,
            no_save=args.no_save,
            force_car=args.force_car,
        )


if __name__ == "__main__":
    main()