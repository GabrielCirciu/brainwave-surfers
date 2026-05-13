# Brainwave Surfers

**An obstacle-avoiding game controlled by your brain using an EEG.**

This project is a collection of Python scripts for processing EEG data and training machine learning models to detect motor imagery. The ultimate goal is to create a seamless brain-computer interface (BCI) for an obstacle-avoiding game.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.12+**
*   **pip**

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/brainwave-surfers/brainwave-surfers.git
    cd brainwave-surfers/PythonBCI
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Data availability

We use the BNCI2014_001 motor imagery dataset from MOABB. You can download and prepare the dataset (in both 22-channel full and 8-channel stripped versions) using the included script:
```bash
python PythonBCI/src/download_gold_data_2.py
```

We also include our own datasets collected from motor imagery experiments in `PythonBCI/data/raw`, using the g.tec Unicorn Hybrid Black EEG device.

All files are in `.npz` format.

### Game setup

The obstacle-avoiding game is built with Unity.
1. Open the repository root folder in Unity Editor.
2. The game streams event markers to an LSL stream, with the name containing `Unity` in it.
3. To run the data collection pipeline, start your EEG LSL stream and execute the following script:
   ```bash
   python PythonBCI/src/online_refine.py
   ```
   Then you can press the `Calibrate` button in game and it will automatically collect the data.
4. To control the game using the trained model, start your EEG LSL stream and execute the realtime prediction script:
   ```bash
   python PythonBCI/src/realtime_predict.py
   ```
   Then you can press the `Start Game` button and you are ready to play.

All game related scripts can be found in `Assets/Scripts`.

### Pipeline scripts

The modular pipeline architecture is defined in `PythonBCI/src` (see `PythonBCI/src/README.md` for full details).

The signal processing and classification steps include:
- **Preprocessing**: It includes data filtering of cutting start and end of data off, Notch Filtering (50Hz), Bandpass Filtering (8-30 Hz), Common Average Reference (CAR).
- **Classification Pipelines**: We have several pipelines, such as Data Augmentation + Covariance Matrix + Tangent Space + Standard Scaler + Logistic Regression, to just name one of them.
- **Evaluation**: We evaluate the pipelines using cross-validation on the training data and report the accuracy. Best model gets saved.

### Visualizations

The `PythonBCI/src/plots` directory contains scripts for generating publication-ready visualizations (see `PythonBCI/src/plots/README.md` for full details). This is where most of our images for our publication were generated. 

## Contributing

Fork the project and feel free to do whatever you'd like with your fork.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.