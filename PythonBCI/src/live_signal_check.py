import numpy as np
import time
import os
import sys
from pylsl import resolve_streams, StreamInlet
import mne

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    print("Looking for LSL EEG stream...")
    streams = resolve_streams()
    eeg_streams = [s for s in streams if s.type() == 'Data' or s.type() == 'EEG']
    
    if not eeg_streams:
        print("No EEG streams found. Is the headset transmitting?")
        return
        
    inlet = StreamInlet(eeg_streams[0])
    info = inlet.info()
    n_channels = info.channel_count()
    fs = info.nominal_srate()
    if fs == 0: fs = 250.0
    
    print(f"Connected to stream: {info.name()} | Channels: {n_channels} | FS: {fs}Hz")
    print("Starting Live Signal Check...")
    time.sleep(2)
    
    buffer_seconds = 4.0
    buffer_size = int(fs * buffer_seconds)
    buffer = np.zeros((n_channels, buffer_size))
    
    ch_names = []
    desc = info.desc().child("channels")
    if not desc.empty():
        ch = desc.child("channel")
        for i in range(n_channels):
            name = ch.child_value("label")
            ch_names.append(name if name else f"CH {i+1}")
            ch = ch.next_sibling()
    
    if not ch_names or all("CH " in n for n in ch_names):
        unicorn_names = ["FC3", "C3", "CP3", "Cz", "CPz", "FC4", "C4", "CP4"]
        ch_names = unicorn_names + [f"AUX {i+1}" for i in range(n_channels - len(unicorn_names))]
    
    samples_received = 0
    last_update_time = time.time()
    
    while True:
        chunk, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=buffer_size)
        if chunk:
            chunk = np.array(chunk).T 
            n_samples = chunk.shape[1]
            
            buffer = np.roll(buffer, -n_samples, axis=1)
            buffer[:, -n_samples:] = chunk
            samples_received += n_samples
            
            current_time = time.time()
            if samples_received >= fs or (current_time - last_update_time) > 1.0:
                samples_received = 0
                last_update_time = current_time
                
                temp_std = np.std(buffer, axis=1)
                scale_factor = 1.0
                if np.median(temp_std) < 1e-3:
                    scale_factor = 1e6
                
                scaled_buffer = buffer * scale_factor
                
                try:
                    eeg_buffer = scaled_buffer[:8, :]
                    
                    f_notch = mne.filter.notch_filter(eeg_buffer, fs, [50], verbose=False)
                    f_band = mne.filter.filter_data(f_notch, fs, 1.0, 40.0, verbose=False)
                    
                    def apply_oscar(X):
                        cov = np.cov(X)
                        evals, evecs = np.linalg.eigh(cov)
                        threshold = np.median(evals) * 15.0
                        evals_capped = np.minimum(evals, threshold)
                        whitening_mat = evecs @ np.diag(np.sqrt(evals_capped / (evals + 1e-9))) @ evecs.T
                        return whitening_mat @ X

                    f_clean = apply_oscar(f_band)
                    f_car = f_clean - np.mean(f_clean, axis=0)
                    
                    eval_samples = int(fs * 1.0)
                    stable_data = f_car[:, -eval_samples:]
                    noise_levels = np.std(stable_data, axis=1)
                except Exception as e:
                    noise_levels = temp_std[:8] * scale_factor

                dc_offsets = np.mean(scaled_buffer[:8, :], axis=1)
                
                clear_console()
                
                bad_channels = []
                dead_channels = []
                
                for i in range(min(8, n_channels)):
                    noise = noise_levels[i]
                    offset = dc_offsets[i]
                    name = ch_names[i]
                    
                    if noise < 0.5:
                        status = "DEAD (under 0.5 uV)"
                        dead_channels.append(name)
                    elif noise < 50:
                        status = "GOOD (under 50 uV)"
                    elif noise < 100:
                        status = "NOISY (under 100 uV)"
                        bad_channels.append(name)
                    else:
                        status = "BAD CONTACT"
                        bad_channels.append(name)
                    
                    bar_length = min(40, int(noise / 2.5))
                    bar = "█" * bar_length
                    
                    print(f"{name.ljust(4)} | Noise: {noise:6.1f} uV | Offset: {offset:8.0f} uV | {status.ljust(16)} | {bar}")
                
                print("\Result:")
                if len(dead_channels) > 0:
                    print(f"DEAD: Channels {dead_channels} are dead.")
                elif len(bad_channels) > 0:
                    print(f"NOISY: Channels {bad_channels} are struggling.")
                else:
                    print("GOOD: Every electrode has a good connection.")
                
        time.sleep(0.01)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Signal Check.")
