import time
import numpy as np
from pylsl import StreamInlet, resolve_streams
import os
import sys

def main():
    # Enable ANSI escape sequences on Windows for flicker-free terminal updates
    # Allows the cursor to be moved in any windows version in terminal
    # This means we can overwrite the terminal texts without clearing it all the time
    # Right now we don't use it but it might be useful in the future
    if os.name == 'nt':
        os.system('') 

    print("Looking for EEG stream...")
    eeg_streams = []
    while not eeg_streams:
        streams = resolve_streams()
        
        print("\nFound the following streams on the network")
        for i, s in enumerate(streams):
            print(f"Stream {i}: {s.name()} | Type: {s.type()} | Channels: {s.channel_count()}")
        
        print("\nFiltering for Data streams...")
        eeg_streams = [s for s in streams if s.type() == 'Data']
        for s in eeg_streams:
            print(f"Found Data stream '{s.name()}'...")
        
        if len(eeg_streams) == 0:
            print("Still waiting for the Data stream... (Make sure LSL is set to 'send all signals in one stream')")
            time.sleep(1)
    
    print("\nPress any key to continue...")
    input()

    # We might have zombie streams that were not closed, so we must use the one that is transmitting data.
    target_stream = None
    eeg_inlet = None
    
    print("\nTesting streams for active data...")
    for s in reversed(eeg_streams):
        print(f"Testing stream '{s.name()}'...")
        inlet = StreamInlet(s)
        chunk, ts = inlet.pull_chunk(timeout=1.0, max_samples=250)
        
        if chunk:
            print("Success! Data is flowing.")
            target_stream = s
            eeg_inlet = inlet
            break
        else:
            print("No data. This is a zombie stream from earlier.")
            
    if target_stream is None:
        print("\nERROR: All located 8-channel streams are dead. Please restart the EEG, LSL stream, and then restart this process!")
        return
        
    stream_channels = eeg_inlet.info().channel_count()
    fs = int(eeg_inlet.info().nominal_srate())
    
    print(f"\nConnected to active stream! channels={stream_channels}, fs={fs}Hz")
    print("Press Ctrl+C to stop.")

    # Short wait helps collect stable data before starting to display it
    time.sleep(1)
    
    # Motor Imagery 8-channel labels for Unicorn EEG (CF3, C3, CP3, Cz, CPz, CF4, C4, CP4)
    labels = ["CF3", "C3", "CP3", "Cz", "CPz", "CF4", "C4", "CP4"]

    all_samples = []

    try:
        while True:
            # Pull a chunk of data (up to 1 second worth)
            chunk, timestamps = eeg_inlet.pull_chunk(timeout=1.0)
            
            if chunk:
                # We only care about displaying the most recent single sample from the chunk
                latest_sample = chunk[-1]

                # Add chunk and timestamp to a list
                all_samples.append((chunk, timestamps))
                
                print("\n")

                for i, val in enumerate(latest_sample):
                    label = labels[i] if i < len(labels) else f"Channel {i+1}"
                    # Format standard values, but print counter/battery as integers if possible
                    if i in [14, 15, 16]:
                        print(f"{label:>15}: {int(val):>10}")
                    else:
                        print(f"{label:>15}: {val:>10.3f}")
                
            else:
                print("No data chunk received. Retrying...")
                
            time.sleep(1.0)  # Only print once a second so it doesn't spam too fast
            
    except KeyboardInterrupt:

        # Save all chunks and timestamps
        if all_samples:

            # Check for output_data_version.txt and read the version from it
            # then save npz with the appropriate version in the name
            with open('PythonBCI/data/config/live_data_version.txt', 'r') as f:
                current_version = f.read()

            output_file = 'PythonBCI/data/raw/live_data_' + current_version + '.npz'
            with open('PythonBCI/data/config/live_data_version.txt', 'w') as f:
                f.write(str(int(current_version) + 1))

            # Flatten lists of chunks and timestamps into single consistent arrays
            all_chunks = np.concatenate([s[0] for s in all_samples], axis=0)
            all_ts = np.concatenate([s[1] for s in all_samples], axis=0)

            np.savez(output_file, chunk=all_chunks, timestamps=all_ts)
            print(f"\n\nSaved {len(all_ts)} samples to {output_file}")

        else:
            print("\n\nNo data collected to save.")

        print("Stopped live monitor.")

if __name__ == '__main__':
    main()
