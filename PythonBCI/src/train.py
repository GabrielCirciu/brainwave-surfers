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

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ShuffleSplit, GridSearchCV, cross_validate


def load_and_preprocess(data_path, use_aux=False, resample_fs=False, eeg_device="Unicorn"):
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
        # Leave only channel 19, 28, 37, 30, 39, 23, 32, 41
        epochs_data = epochs_data[:, [18, 27, 36, 29, 38, 22, 31, 40], :]
        print(f"hiAmp device detected.Selected channels: {[18, 27, 36, 29, 38, 22, 31, 40]}")
    else:
        print("Unicorn device detected.Using all channels.")
        
    labels = data["labels"]
    # gather how many rows in a sample, divide it by 4 and that is the fs
    fs = epochs_data.shape[2] / 4
    print(f"Sample frequency: {fs} Hz")

    print(f"Loaded data shape: {epochs_data.shape} (Trials, Channels, Samples)")

    n_channels = epochs_data.shape[1]
    
    ch_names = [f"EEG {i+1}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    
    if "eeg" in data and use_aux and "aux" in data:
        orig_eeg = data["eeg"]
        if orig_eeg.ndim == 3:
            n_orig_eeg = orig_eeg.shape[2] if orig_eeg.shape[1] > orig_eeg.shape[2] else orig_eeg.shape[1]
        elif orig_eeg.ndim == 2:
            n_orig_eeg = orig_eeg.shape[1] if orig_eeg.shape[0] > orig_eeg.shape[1] else orig_eeg.shape[0]
        else:
            n_orig_eeg = n_channels
            
        for i in range(n_orig_eeg, n_channels):
            ch_names[i] = f"AUX {i - n_orig_eeg + 1}"
            ch_types[i] = "eeg" # Filter aux same as eeg

    info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types=ch_types)

    epochs = mne.EpochsArray(epochs_data * 1e-6, info, verbose=False)
    
    if resample_fs:
        print(f"Resampling from {fs} Hz to 250.0 Hz...")
        epochs = epochs.resample(250.0, verbose=False)
        fs = 250.0
        
    print("Filtering data (8-30 Hz)...")
    epochs.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge", verbose=False) 
    epochs.crop(tmin=0.5) #!TODO maybe make this one parametric?

    X = epochs.get_data(copy=True)
    y = labels

    return X, y, fs


def get_pipeline_and_grid(name):
    # Covariance, Tangent Space, Logistic Regression
    if name == "cov_ts_lr":
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ])
        param_grid = {
            "cov__estimator": ["oas", "lwf"],
            "ts__metric": ["riemann", "logeuclid"],
            "clf__C": [0.1, 1.0, 10.0],
        }

    # Common Spatial Patterns, Support Vector Machine
    elif name == "csp_svm":
        pipeline = Pipeline([
            ("csp", CSP(log=True, norm_trace=False)),
            ("clf", SVC(probability=True)),
        ])
        param_grid = {
            "csp__n_components": [2, 4],
            "csp__reg": [None, "ledoit_wolf"],
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1.0, 10.0],
        }

    # Common Spatial Patterns, Linear Discriminant Analysis
    elif name == "csp_lda":
        pipeline = Pipeline([
            ("csp", CSP(log=True, norm_trace=False)),
            ("clf", LinearDiscriminantAnalysis()),
        ])
        param_grid = {
            "csp__n_components": [2, 4],
            "csp__reg": [None, "ledoit_wolf"],
            "clf__solver": ["lsqr", "eigen"],
            "clf__shrinkage": ["auto"],
        }

    # Common Spatial Patterns, Random Forest
    elif name == "csp_rf":
        pipeline = Pipeline([
            ("csp", CSP(log=True, norm_trace=False)),
            ("clf", RandomForestClassifier(random_state=42)),
        ])
        param_grid = {
            "csp__n_components": [2, 4],
            "csp__reg": [None, "ledoit_wolf"],
            "clf__n_estimators": [50, 100],
            "clf__max_depth": [None, 3],
        }

    # Common Spatial Patterns, Multi-Layer Perceptron
    elif name == "csp_mlp":
        pipeline = Pipeline([
            ("csp", CSP(log=True, norm_trace=False)),
            ("clf", MLPClassifier(max_iter=1000, random_state=42)),
        ])
        param_grid = {
            "csp__n_components": [2, 4],
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__alpha": [0.0001, 0.001],
            "clf__activation": ["relu", "tanh"]
        }

    # Covariance, Tangent Space, Multi-Layer Perceptron
    elif name == "cov_ts_mlp":
        pipeline = Pipeline([
            ("cov", Covariances(estimator="oas")),
            ("ts", TangentSpace(metric="riemann")),
            ("clf", MLPClassifier(max_iter=1000, random_state=42)),
        ])
        param_grid = {
            "cov__estimator": ["oas"],
            "ts__metric": ["riemann"],
            "clf__hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "clf__alpha": [0.0001, 0.001],
            "clf__activation": ["relu", "tanh"]
        }

    else:
        raise ValueError("pipeline must be one of: cov_ts_lr, csp_svm, csp_lda, csp_rf, csp_mlp, cov_ts_mlp")

    return pipeline, param_grid


def train_model(data_path, pipeline_name="cov_ts_lr", save_dir="../models", use_grid=False, use_aux=False, resample_fs=False, eeg_device=""):
    X, y, fs = load_and_preprocess(data_path, use_aux=use_aux, resample_fs=resample_fs, eeg_device=eeg_device)
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

    # Prefer stratified splits when possible to avoid folds with single classes
    if run_cv and min_class_count >= 2:
        from sklearn.model_selection import StratifiedShuffleSplit
        cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    else:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

    print(f"Pipeline: {pipeline_name}")
    print(f"Grid search: {use_grid}")

    old_log_level = mne.get_config("MNE_LOGGING_LEVEL", "INFO")
    mne.set_log_level("ERROR")

    try:
        if run_cv and use_grid:
            print("Running GridSearchCV...")
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
    model_path = save_dir / f"{model_name}.pkl"
    report_path = save_dir / f"{model_name}_report.json"

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

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

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"AUC: {auc_mean:.4f}")
    print(f"Accuracy: {acc_mean:.4f}")
    if best_params:
        print(f"Best params: {best_params}")
    print(f"Model saved to: {model_path}")
    print(f"Report saved to: {report_path}")

    return best_model, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--pipeline")
    parser.add_argument("--save_dir")
    parser.add_argument("--use_grid", action="store_true")
    parser.add_argument("--use_aux", action="store_true")
    parser.add_argument("--resample_fs", default=False)
    parser.add_argument("--eeg_device", default="Unicorn")
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        pipeline_name=args.pipeline or "cov_ts_lr",
        save_dir=args.save_dir or "../models",
        use_grid=args.use_grid,
        use_aux=args.use_aux,
        resample_fs=args.resample_fs,
        eeg_device=args.eeg_device,
    )


if __name__ == "__main__":
    main()