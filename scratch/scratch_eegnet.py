import numpy as np
import os
import glob
import mne

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

def load_custom_mne(data_dir):
    session_files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
    all_eeg, all_labels = [], []
    for f in session_files:
        if "merged" in f: continue
        d = np.load(f)
        all_eeg.append(d['eeg'])
        all_labels.append(d['labels'])
    eeg = np.concatenate(all_eeg, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    if eeg.shape[1] == 1000:
        eeg = np.transpose(eeg, (0, 2, 1))

    for i in range(eeg.shape[0]):
        sample_axis = 1 if eeg.shape[1] == 1000 else 2
        centered = eeg[i] - np.mean(eeg[i], axis=sample_axis-1, keepdims=True)
        trial_std = np.std(centered)
        if trial_std > 0: eeg[i] = centered / trial_std

    ch_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
    info = mne.create_info(ch_names=ch_names, sfreq=250.0, ch_types=['eeg']*8)
    
    events = np.zeros((len(labels), 3), dtype=int)
    events[:, 0] = np.arange(len(labels)) * 1000
    events[:, 2] = labels
    
    epochs = mne.EpochsArray(eeg * 1e-6, info, events=events, verbose=False)
    epochs.set_montage("standard_1020")
    return epochs

def get_eegnet(nb_classes, Chans=8, Samples=1000, 
             dropoutRate=0.5, kernLength=125, F1=8, 
             D=2, F2=16, norm_rate=0.25):
    
    input1   = Input(shape=(Chans, Samples, 1))

    block1       = Conv2D(F1, (1, kernLength), padding='same',
                                   input_shape=(Chans, Samples, 1),
                                   use_bias=False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias=False, 
                                   depth_multiplier=D,
                                   depthwise_constraint=max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = Dropout(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias=False, padding='same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = Dropout(dropoutRate)(block2)
        
    flatten      = Flatten(name='flatten')(block2)
    
    dense        = Dense(nb_classes, name='dense', 
                         kernel_constraint=max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name='softmax')(dense)
    
    return Model(inputs=input1, outputs=softmax)

def run_eegnet():
    session_dir = r"PythonBCI\data\raw\vikt-26-04-29-14-07"
    epochs = load_custom_mne(session_dir)
    print("Loaded Raw Data. Applying 8-15 Hz Filter...")
    epochs.filter(8.0, 15.0, fir_design='firwin', verbose=False)
    
    # CAR
    epochs.set_eeg_reference("average", ch_type="eeg", verbose=False)
    
    X = epochs.get_data(copy=False)
    y = epochs.events[:, -1]
    
    # Drop bad to maximize chance
    # epochs.drop_bad(reject=dict(eeg=200e-6), verbose=False) # Skip strict rejection for DL, let it learn noise
    
    # Format for Keras (Trials, Channels, Samples, Kernels)
    X = np.expand_dims(X, axis=-1)
    
    print("\n" + "="*50)
    print("Training State-of-the-Art Deep Learning (EEGNet)...")
    print("="*50)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        model = get_eegnet(nb_classes=2, Chans=8, Samples=1000)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # Early stopping to prevent overfitting on tiny dataset
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        
        model.fit(X_train, y_train, batch_size=16, epochs=100, 
                  verbose=0, validation_data=(X_val, y_val), callbacks=[early_stop])
        
        preds = np.argmax(model.predict(X_val, verbose=0), axis=-1)
        probs = model.predict(X_val, verbose=0)[:, 1]
        
        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)
        acc_scores.append(acc)
        auc_scores.append(auc)
        print(f"Fold {fold+1}: Acc {acc:.4f} | AUC {auc:.4f}")
        
    print("\nFINAL DEEP LEARNING (EEGNet) SCORE:")
    print(f"Accuracy: {np.mean(acc_scores):.4f}")
    print(f"AUC:      {np.mean(auc_scores):.4f}")

if __name__ == '__main__':
    run_eegnet()
