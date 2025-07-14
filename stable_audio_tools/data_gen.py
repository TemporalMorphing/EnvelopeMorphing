import os
import numpy as np
import random
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt


def extract_envelope(filepath, target_length=2048, cutoff_freq=30, plot=True):
    """
    Extract a fixed-length temporal envelope from an audio file.

    Parameters:
    -----------
    filepath : str
        Path to the WAV file
    target_length : int
        Desired length of the output envelope (default: 2048)
    cutoff_freq : float
        Cutoff frequency for the lowpass filter in Hz (default: 30)
    plot : bool
        Whether to create visualization plots (default: True)

    Returns:
    --------
    tuple
        (original_envelope, smoothed_envelope, resampled_envelope)
    """
    # Read audio file
    sample_rate, audio = wavfile.read(filepath)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio
    audio = audio.astype(float) / np.max(np.abs(audio))

    # Calculate temporal envelope using Hilbert transform
    analytic_signal = hilbert(audio)
    envelope = np.abs(analytic_signal)

    # Design and apply lowpass filter
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normalized_cutoff, btype='low')
    smoothed_envelope = filtfilt(b, a, envelope)

    # Resample to target length
    original_indices = np.linspace(0, len(smoothed_envelope)-1, len(smoothed_envelope))
    target_indices = np.linspace(0, len(smoothed_envelope)-1, target_length)
    resampled_envelope = np.interp(target_indices, original_indices, smoothed_envelope)

    if plot:
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))

        # Plot original envelope
        time = np.arange(len(envelope)) / sample_rate
        ax1.plot(time, envelope)
        ax1.set_title('Original Temporal Envelope')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')

        # Plot smoothed envelope
        ax2.plot(time, smoothed_envelope)
        ax2.set_title(f'Smoothed Envelope (Lowpass {cutoff_freq}Hz)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')

        # Plot resampled envelope
        resampled_time = np.linspace(0, time[-1], target_length)
        ax3.plot(resampled_time, resampled_envelope)
        ax3.set_title(f'Resampled Envelope ({target_length} samples)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude')

        plt.tight_layout()
        plt.show()

    return envelope, smoothed_envelope, resampled_envelope

# Function to create the directory if it does not exist
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to process all files in a directory and save envelopes
def process_audio_directory(input_dir, output_dir, plot_dir, num_plots=20):
    # Ensure output directories exist
    ensure_directory_exists(output_dir)
    ensure_directory_exists(plot_dir)

    # Get list of all .wav files in the input directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    # Randomly select files for which to save plots
    plot_files = random.sample(wav_files, min(num_plots, len(wav_files)))

    # Process each file
    for wav_file in wav_files:
        input_path = os.path.join(input_dir, wav_file)
        output_path = os.path.join(output_dir, wav_file)

        # Extract the envelope
        _, _, resampled_envelope = extract_envelope(input_path, plot=wav_file in plot_files)

        # Save the resampled envelope as a 2048-sample .wav file
        # Use a sample rate of 16000 (arbitrary choice, since envelope is normalized)
        sample_rate = 200
        wavfile.write(output_path, sample_rate, resampled_envelope.astype(np.float32))

        # Save the plot if the file is selected
        if wav_file in plot_files:
            plot_output_path = os.path.join(plot_dir, wav_file.replace('.wav', '.png'))

            # Plot and save the figure
            plt.figure(figsize=(10, 4))
            plt.plot(resampled_envelope)
            plt.title(f'Envelope for {wav_file}')
            plt.xlabel('Sample Index')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(plot_output_path)
            plt.close()

    print(f"Processing complete. Envelopes saved to {output_dir}, plots saved to {plot_dir}.")

# Paths for input, output, and plot directories
input_directory = "audio_samples_directory_path"  # Replace with your actual path
output_directory = "envelopes"
plot_directory = "plots"

# Process the directory
process_audio_directory(input_directory, output_directory, plot_directory)
