import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import hilbert, butter, filtfilt
import os
import pandas as pd
import random
from tqdm import tqdm
import math
import shutil
from datetime import datetime

# Create timestamp for the run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create main directories
output_dir = f"morphing_dataset"
os.makedirs(output_dir, exist_ok=True)

# Create subdirectories for different types of data
gaussian_dir = os.path.join(output_dir, "gaussian_envelopes")
realworld_dir = os.path.join(output_dir, "real_world_envelopes")
plot_dir = os.path.join(output_dir, "plots")

# Create specific plot directories
plot_dirs = {
    "train_simple": os.path.join(plot_dir, "train_simple"),
    "test_simple": os.path.join(plot_dir, "test_simple"),
    "test_composed": os.path.join(plot_dir, "test_composed"),
    "test_real_simple": os.path.join(plot_dir, "test_real_simple"),
    "test_real_composed": os.path.join(plot_dir, "test_real_composed")
}

# Create all directories
os.makedirs(gaussian_dir, exist_ok=True)
os.makedirs(realworld_dir, exist_ok=True)
for dir_path in plot_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Parameters
SAMPLE_RATE = 200  # Hz
NUM_SAMPLES = 2048
DURATION = NUM_SAMPLES / SAMPLE_RATE  # seconds

# Alpha values for morphing
ALPHA_VALUES = [0, 0.25, 0.5, 0.75, 1]

# Define the composite morph configurations
COMPOSITE_CONFIGURATIONS = [
    # 2 properties
    {"amplitude": True, "placement": True, "inter_impulse_distance": False, "n_impulses": False},
    {"amplitude": True, "placement": False, "inter_impulse_distance": True, "n_impulses": False},
    {"amplitude": True, "placement": False, "inter_impulse_distance": False, "n_impulses": True},
    {"amplitude": False, "placement": True, "inter_impulse_distance": True, "n_impulses": False},
    {"amplitude": False, "placement": True, "inter_impulse_distance": False, "n_impulses": True},
    {"amplitude": False, "placement": False, "inter_impulse_distance": True, "n_impulses": True},
    
    # 3 properties
    {"amplitude": True, "placement": True, "inter_impulse_distance": True, "n_impulses": False},
    {"amplitude": False, "placement": True, "inter_impulse_distance": True, "n_impulses": True},
    {"amplitude": True, "placement": False, "inter_impulse_distance": True, "n_impulses": True},
    {"amplitude": True, "placement": True, "inter_impulse_distance": False, "n_impulses": True},
    
    # 4 properties
    {"amplitude": True, "placement": True, "inter_impulse_distance": True, "n_impulses": True},
]

# Define the audio files and their segments
audio_files = {
    "glass": {"file": "audio_files/breaking_glass.wav", "start": 8.8, "end": 9.9},
    "duck": {"file": "audio_files/duck.wav", "start": 0.5, "end": 0.9},
    "dog": {"file": "audio_files/dog.wav", "start": 5.7, "end": 6},
    "stapler": {"file": "audio_files/stapler.wav", "start": 0, "end": 0.2},
    "bell": {"file": "audio_files/bicycle_bell.wav", "start": 0, "end": 0.5},
    "alarm": {"file": "audio_files/alarm.wav", "start": 0, "end": 0.3}, 
    "meow": {"file": "audio_files/cat_meow.wav", "start": 1.2, "end": 2.6},
    "door": {"file": "audio_files/door_creak.wav", "start": 0, "end": 0.8},
    "church": {"file": "audio_files/church_bell.wav", "start": 0, "end": 1},
    "goat": {"file": "audio_files/goat.wav", "start": 0.3, "end": 1.2}
}


# Cache for extracted audio envelopes
envelope_cache = {}

###########################################
# Part 1: Gaussian Envelope Functions
###########################################



def gaussian_impulse(t, center, width, amplitude):
    """Generate a Gaussian impulse that goes to zero between impulses."""
    gaussian = amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)
    # Apply threshold to ensure clean zeros
    threshold = amplitude * 0.001  # 0.1% of peak amplitude
    gaussian[gaussian < threshold] = 0
    return gaussian


def gaussian_env(n_impulses, placement, inter_impulse_distance, amplitude):
    """
    Generate a Gaussian audio envelope with clean zeros between impulses.
    """
    # Generate time points
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    
    # Initialize envelope
    envelope = np.zeros(NUM_SAMPLES)
    
    # Use a narrower width for cleaner separation
    width = 0.05  # Even narrower than before
    
    # Calculate start position so that the center impulse is at placement
    if n_impulses % 2 == 1:  # Odd number of impulses
        start_pos = placement - (n_impulses // 2) * inter_impulse_distance
    else:  # Even number of impulses
        start_pos = placement - (n_impulses / 2 - 0.5) * inter_impulse_distance
    
    # Check if any impulse would go outside our time range
    for i in range(n_impulses):
        center_time = start_pos + i * inter_impulse_distance
        if center_time < width * 4 or center_time > DURATION - width * 4:
            return None
    
    # Check for potential overlap - increase minimum distance
    if inter_impulse_distance < width * 8:  # Much larger separation
        return None
    
    # Add each impulse using the fixed function
    for i in range(n_impulses):
        center_time = start_pos + i * inter_impulse_distance
        envelope += gaussian_impulse(t, center_time, width, amplitude)
    
    # Ensure clean zeros and no overlap
    if np.max(envelope) > 1.0 + 1e-6:
        return None
    
    # Apply final threshold to ensure clean zeros
    threshold = 0.001
    envelope[envelope < threshold] = 0
    envelope = np.clip(envelope, 0, 1)
    
    return envelope

def power_interpolate_amplitude(amplitude_1, amplitude_2, alpha):
    """
    Interpolate amplitude using x1^alpha * x2^(1-alpha) formula.
    
    Parameters:
    -----------
    amplitude_1, amplitude_2 : float
        Amplitude values to interpolate between (0-1)
    alpha : float
        Morphing parameter (0 to 1), where 0 means fully amplitude_2, 
        1 means fully amplitude_1
        
    Returns:
    --------
    float
        Power-interpolated amplitude value
    """
    # Ensure amplitudes are not zero (to avoid issues with power function)
    epsilon = 1e-6
    amp1 = max(amplitude_1, epsilon)
    amp2 = max(amplitude_2, epsilon)
    
    # Power interpolation: x1^alpha * x2^(1-alpha)
    interpolated = (amp1 ** alpha) * (amp2 ** (1 - alpha))
    
    # Clip to ensure we're in [0, 1] range
    return min(max(interpolated, 0), 1)

def save_envelope(envelope, filename):
    """Save the envelope as a .wav file."""
    # Convert to int16 format for wav file
    audio_data = (envelope * 32767).astype(np.int16)
    wavfile.write(filename, SAMPLE_RATE, audio_data)
    return filename

def plot_envelope(envelope, title, save_path=None):
    """Plot the envelope."""
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    plt.figure(figsize=(10, 4))
    plt.plot(t, envelope)
    plt.title(title, fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(False)  # Remove gridlines
    plt.ylim(0, 1.05)  # Set y-axis limits
    plt.tick_params(axis='both', which='major', labelsize=10)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_all_directories():
    """Generate 100 plots for each envelope directory."""
    directories_to_plot = [
        (gaussian_dir, os.path.join(plot_dir, "gaussian_samples")),
        (realworld_dir, os.path.join(plot_dir, "realworld_samples"))
    ]
    
    for wav_dir, plot_output_dir in directories_to_plot:
        if os.path.exists(wav_dir):
            print(f"Generating plots for {wav_dir}...")
            generate_plots_for_directory(wav_dir, plot_output_dir, num_plots=100)
            print(f"Plots saved to {plot_output_dir}")

def plot_morph_sequence(filepaths, alphas, title, save_path=None):
    """Plot a sequence of envelopes to show morphing progression."""
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    fig, axs = plt.subplots(len(alphas), 1, figsize=(12, 10))
    
    for i, (alpha, filepath) in enumerate(zip(alphas, filepaths)):
        # Load wav file
        _, audio_data = wavfile.read(filepath)
        envelope = audio_data / 32767.0  # Convert to [0,1] range
        
        # Plot on the corresponding subplot
        axs[i].plot(t, envelope)
        axs[i].set_title(f"Alpha = {alpha}", fontsize=12)
        axs[i].grid(False)  # Remove gridlines
        axs[i].set_ylim(0, 1.05)
        axs[i].tick_params(axis='both', which='major', labelsize=10)
        
        # Add x-label only to the bottom plot
        if i == len(alphas) - 1:
            axs[i].set_xlabel('Time (s)', fontsize=12)
        
        axs[i].set_ylabel('Amplitude', fontsize=12)
    
    # Add an overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return fig

###########################################
# Part 1.5: Square Wave Envelope Functions
###########################################

def square_impulse(t, center, width, amplitude):
    """Generate a square impulse."""
    # Calculate distance from center
    distance = np.abs(t - center)
    # Create square pulse with half-width
    return amplitude * (distance <= width/2).astype(float)

def square_env(n_impulses, placement, inter_impulse_distance, amplitude):
    """
    Generate a square wave audio envelope with the specified parameters.
    
    Parameters:
    -----------
    n_impulses : int
        Number of impulses (1-10)
    placement : float
        Center time point of the impulses (in seconds)
    inter_impulse_distance : float
        Distance between impulses (in seconds)
    amplitude : float
        Amplitude of impulses (0-1)
        
    Returns:
    --------
    numpy.ndarray or None
        Audio envelope with 2048 samples, or None if constraints cannot be satisfied
    """
    # Generate time points
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    
    # Initialize envelope
    envelope = np.zeros(NUM_SAMPLES)
    
    # Use a fixed, narrow width for all impulses
    width = 0.1  # in seconds - much narrower than before
    
    # Calculate start position so that the center impulse is at placement
    if n_impulses % 2 == 1:  # Odd number of impulses
        start_pos = placement - (n_impulses // 2) * inter_impulse_distance
    else:  # Even number of impulses
        start_pos = placement - (n_impulses / 2 - 0.5) * inter_impulse_distance
    
    # Check if any impulse would go outside our time range
    for i in range(n_impulses):
        center_time = start_pos + i * inter_impulse_distance
        if center_time - width/2 < 0 or center_time + width/2 > DURATION:
            # Too close to boundary
            return None
    
    # Check for potential overlap between impulses
    if inter_impulse_distance < width * 2:  # Increased minimum distance to ensure no overlap
        # Impulses are too close to each other and would overlap
        return None
    
    # Add each impulse
    for i in range(n_impulses):
        center_time = start_pos + i * inter_impulse_distance
        envelope += square_impulse(t, center_time, width, amplitude)
    
    # If maximum value exceeds 1.0, this configuration produces overlap
    if np.max(envelope) > 1.0 + 1e-6:  # Small tolerance for floating point errors
        return None
    
    # Clip to ensure no values exceed 1.0
    envelope = np.clip(envelope, 0, 1)
    
    return envelope

def generate_square_envelope(n_impulses, placement, inter_impulse_distance, amplitude):
    """
    Generate a square wave envelope and save it with a descriptive filename.
    
    Returns (filename, envelope) if successful, (None, None) if constraints cannot be satisfied.
    """
    envelope = square_env(n_impulses, placement, inter_impulse_distance, amplitude)
    
    # If the envelope couldn't be created with the constraints, return None
    if envelope is None:
        return None, None
    
    # Create filename with parameters
    filename = os.path.join(gaussian_dir, f"square_n{n_impulses}_p{placement:.2f}_i{inter_impulse_distance:.2f}_a{amplitude:.2f}.wav")
    save_envelope(envelope, filename)
    
    return filename, envelope

# Fix 1: Generate more plots (100 per directory)
def generate_plots_for_directory(wav_dir, plot_dir, num_plots=100):
    """Generate plots for a random sample of wav files in a directory."""
    # Get all wav files
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    if len(wav_files) == 0:
        print(f"No wav files found in {wav_dir}")
        return
    
    # Sample randomly
    sample_size = min(num_plots, len(wav_files))
    sampled_files = random.sample(wav_files, sample_size)
    
    os.makedirs(plot_dir, exist_ok=True)
    
    for i, wav_file in enumerate(sampled_files):
        try:
            # Load wav file
            filepath = os.path.join(wav_dir, wav_file)
            sample_rate, audio_data = wavfile.read(filepath)
            envelope = audio_data / 32767.0  # Convert to [0,1] range
            
            # Create plot
            t = np.linspace(0, DURATION, NUM_SAMPLES)
            plt.figure(figsize=(10, 4))
            plt.plot(t, envelope)
            plt.title(f"Envelope: {wav_file}", fontsize=12)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.grid(False)
            plt.ylim(0, 1.05)
            
            # Save plot
            plot_filename = os.path.join(plot_dir, f"plot_{i+1:03d}_{wav_file.replace('.wav', '.png')}")
            plt.savefig(plot_filename)
            plt.close()
            
        except Exception as e:
            print(f"Error plotting {wav_file}: {e}")

###########################################
# Part 2: Real-World Audio Envelope Functions
###########################################

def extract_audio_envelope(audio_type):
    """
    Extract the envelope from the specified audio file using Hilbert transform and smoothing.
    
    Parameters:
    -----------
    audio_type : str
        Key in the audio_files dictionary
        
    Returns:
    --------
    numpy.ndarray
        Normalized envelope of the audio segment
    """
    # Check if already in cache
    if audio_type in envelope_cache:
        return envelope_cache[audio_type]
    
    # Get file info
    file_info = audio_files[audio_type]
    file_path = file_info["file"]
    start_time = file_info["start"]
    end_time = file_info["end"]
    
    try:
        # Read audio file
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Convert to float
        audio_data = audio_data.astype(float)
        
        # Normalize audio data
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Extract the segment
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = audio_data[start_sample:end_sample]
        
        # Apply Hilbert transform to get the envelope
        analytic_signal = hilbert(audio_segment)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Apply Butterworth filter for smoothing
        b, a = butter(6, 30.0 / (sample_rate / 2), 'low')
        smooth_envelope = filtfilt(b, a, amplitude_envelope)
        
        # Normalize the envelope to always have max amplitude of 1.0
        smooth_envelope = smooth_envelope / np.max(smooth_envelope)
        
        # Resample to our target length (NUM_SAMPLES)
        indices = np.linspace(0, len(smooth_envelope) - 1, NUM_SAMPLES)
        resampled_envelope = np.interp(indices, np.arange(len(smooth_envelope)), smooth_envelope)
        
        # Store in cache
        envelope_cache[audio_type] = resampled_envelope
        
        return resampled_envelope
    
    except Exception as e:
        print(f"Error extracting envelope from {file_path}: {e}")
        # Return a fallback envelope if there's an error
        t = np.linspace(0, 1, NUM_SAMPLES)
        return np.exp(-0.5 * ((t - 0.5) / 0.1) ** 2)

def audio_impulse(t, center, width, amplitude, audio_type):
    """
    Generate an impulse using the envelope of the specified audio type.
    
    Parameters:
    -----------
    t : numpy.ndarray
        Time points
    center : float
        Center time point of the impulse
    width : float
        Width scaling factor - used only for certain audio types
    amplitude : float
        Amplitude scaling factor
    audio_type : str
        Type of audio envelope to use
        
    Returns:
    --------
    numpy.ndarray
        Scaled and positioned audio envelope
    """
    # Get the base envelope
    base_envelope = extract_audio_envelope(audio_type)
    
    # Get the file info to determine original duration
    file_info = audio_files[audio_type]
    original_duration = file_info["end"] - file_info["start"]
    
    # Decide whether this audio type needs width scaling
    # Keep original durations for short sounds, scale longer ones to 0.5s
    narrow_types = ['duck', 'dog', 'stapler', 'alarm', 'bell']
    wide_types = ['glass', 'meow', 'door', 'church', 'goat']
    
    if audio_type in wide_types:
        # Scale duration to 0.5 seconds for wide types
        effective_duration = 0.5
        # We'll need to stretch/compress the envelope along time axis
        scale_factor = original_duration / effective_duration
    else:
        # Use original duration for narrow types
        effective_duration = original_duration
        scale_factor = 1.0
    
    # Create time indices for the envelope
    envelope_time = np.linspace(0, original_duration, NUM_SAMPLES)
    
    # Calculate time offset to position the envelope
    # Center point of envelope should be at 'center'
    time_offset = center - effective_duration / 2
    
    # Position the envelope
    positioned_time = t - time_offset
    
    if audio_type in wide_types:
        # For wide types, apply time scaling to fit into 0.5s
        positioned_time = positioned_time * scale_factor
    
    # Interpolate the envelope at the positioned time points
    # Any points outside the original envelope time range will be set to 0
    impulse = np.interp(positioned_time, envelope_time, base_envelope, left=0, right=0)
    
    # Scale by amplitude
    impulse *= amplitude
    
    return impulse

def real_world_env(audio_type, n_impulses, placement, inter_impulse_distance, amplitude):
    """
    Generate an audio envelope with the specified parameters.
    
    Parameters:
    -----------
    audio_type : str
        Type of audio envelope to use
    n_impulses : int
        Number of impulses (1-10)
    placement : float
        Center time point of the impulses (in seconds)
    inter_impulse_distance : float
        Distance between impulses (in seconds)
    amplitude : float
        Amplitude of impulses (0-1)
        
    Returns:
    --------
    numpy.ndarray or None
        Audio envelope with 2048 samples, or None if constraints cannot be satisfied
    """
    # Generate time points
    t = np.linspace(0, DURATION, NUM_SAMPLES)
    
    # Initialize envelope
    envelope = np.zeros(NUM_SAMPLES)
    
    # Determine effective envelope duration
    if audio_type in audio_files:
        file_info = audio_files[audio_type]
        original_duration = file_info["end"] - file_info["start"]
        
        # Narrow types keep original duration, wide types use 0.5s
        narrow_types = ['duck', 'dog', 'stapler', 'alarm', 'bell']
        wide_types = ['glass', 'meow', 'door', 'church', 'goat']
        
        if audio_type in wide_types:
            envelope_duration = 0.5  # Scale to 0.5s for wide types
        else:
            envelope_duration = original_duration
    else:
        # Use default width for non-audio types (e.g., 'square')
        envelope_duration = 0.2
    
    # Calculate start position so that the center impulse is at placement
    if n_impulses % 2 == 1:  # Odd number of impulses
        start_pos = placement - (n_impulses // 2) * inter_impulse_distance
    else:  # Even number of impulses
        start_pos = placement - (n_impulses / 2 - 0.5) * inter_impulse_distance
    
    # Check if any impulse would go outside our time range
    for i in range(n_impulses):
        center_time = start_pos + i * inter_impulse_distance
        if center_time - envelope_duration/2 < 0 or center_time + envelope_duration/2 > DURATION:
            # Too close to boundary - need room on each side
            return None
    
    # Check for potential overlap between impulses
    if inter_impulse_distance < envelope_duration:
        # Impulses are too close to each other and would overlap significantly
        return None
    
    # Add each impulse
    for i in range(n_impulses):
        center_time = start_pos + i * inter_impulse_distance
        
        # For real audio types, use the audio_impulse function
        if audio_type in audio_files:
            # Pass 1.0 as width as we handle scaling in audio_impulse now
            envelope += audio_impulse(t, center_time, 1.0, amplitude, audio_type)
        else:
            # For square waves or other non-audio types, use the original approach
            width = envelope_duration
            if audio_type == 'square':
                envelope += square_impulse(t, center_time, width, amplitude)
            else:
                # Fallback to Gaussian for unknown types
                envelope += gaussian_impulse(t, center_time, width/4, amplitude)
    
    # If maximum value exceeds 1.0, this configuration produces overlap
    if np.max(envelope) > 1.0 + 1e-6:  # Small tolerance for floating point errors
        return None
    
    # Clip to ensure no values exceed 1.0
    envelope = np.clip(envelope, 0, 1)
    
    return envelope

###########################################
# Part 3: Gaussian Simple Envelope Dataset Generation
###########################################

def generate_gaussian_envelope(n_impulses, placement, inter_impulse_distance, amplitude):
    """
    Generate a Gaussian envelope and save it with a descriptive filename.
    
    Returns (filename, envelope) if successful, (None, None) if constraints cannot be satisfied.
    """
    envelope = gaussian_env(n_impulses, placement, inter_impulse_distance, amplitude)
    
    # If the envelope couldn't be created with the constraints, return None
    if envelope is None:
        return None, None
    
    # Create filename with parameters
    filename = os.path.join(gaussian_dir, f"gaussian_n{n_impulses}_p{placement:.2f}_i{inter_impulse_distance:.2f}_a{amplitude:.2f}.wav")
    save_envelope(envelope, filename)
    
    return filename, envelope




def generate_gaussian_simple_morph(morph_type, max_attempts=50, envelope_type='gaussian'):
    """
    Generate a set of envelopes with varying alpha values for the specified morph_type.
    
    Parameters:
    -----------
    morph_type : str
        Type of morphing to apply ('n_impulses', 'placement', 'inter_impulse_distance', or 'amplitude')
    max_attempts : int
        Maximum number of attempts to generate a valid set
    envelope_type : str
        Type of envelope to generate ('gaussian' or 'square')
        
    Returns:
    --------
    dict or None
        Dictionary with envelope data if successful, None if constraints cannot be satisfied.
    """
    for attempt in range(max_attempts):
        # Initialize parameters with random values
        # For inter_impulse_distance morph type, ensure n_impulses > 1
        if morph_type == 'inter_impulse_distance':
            n_impulses_1 = random.randint(2, 8)  # Start with at least 2 impulses
        else:
            n_impulses_1 = random.randint(1, 8)  # Using 8 as max to allow room for variation
            
        placement_1 = random.uniform(1.0, DURATION - 1.0)  # Keep further away from edges
        
        # Choose parameters that are more likely to avoid overlap
        width_estimate = min(0.2, 0.5 / n_impulses_1)
        min_distance = width_estimate * 3  # At least 3 widths separation for gaussians
        
        inter_impulse_distance_1 = random.uniform(min_distance, min(0.8, min_distance * 2))
        amplitude_1 = random.uniform(0.2, 0.8)  # Slightly lower max amplitude to reduce overlap chance
        
        # Set up values for envelope 2 based on the morphing parameter
        if morph_type == 'n_impulses':
            # Need at least 2 steps difference for integer morphing
            max_diff = min(4, 10 - n_impulses_1)  # Ensure we don't exceed 10 impulses
            min_diff = min(2, n_impulses_1 - 1)   # Ensure we don't go below 1 impulse
            
            if max_diff < 2 and min_diff < 2:
                continue  # Skip this attempt if we can't make a valid difference
            
            if random.random() < 0.5 and max_diff >= 2:
                # Increase n_impulses
                diff = random.randint(2, max_diff)
                n_impulses_2 = n_impulses_1 + diff
            elif min_diff >= 2:
                # Decrease n_impulses
                diff = random.randint(2, min_diff)
                n_impulses_2 = n_impulses_1 - diff
            else:
                continue  # Skip if we can't make a valid difference
            
            # Keep other parameters the same
            placement_2 = placement_1
            inter_impulse_distance_2 = inter_impulse_distance_1
            amplitude_2 = amplitude_1
            
        elif morph_type == 'placement':
            # Vary placement, making sure there's room on both sides
            max_diff = min(placement_1 - 1.0, DURATION - placement_1 - 1.0) * 0.8  # 80% of available space
            if max_diff < 0.2:
                continue  # Not enough room to make a good difference
            
            diff = random.uniform(0.2, max_diff)
            if random.random() < 0.5:
                placement_2 = max(0.5, placement_1 - diff)
            else:
                placement_2 = min(DURATION - 0.5, placement_1 + diff)
            
            # Keep other parameters the same
            n_impulses_2 = n_impulses_1
            inter_impulse_distance_2 = inter_impulse_distance_1
            amplitude_2 = amplitude_1
            
        elif morph_type == 'inter_impulse_distance':
            # Need wider range for effective morphing while avoiding overlap
            min_valid_distance = width_estimate * 2  # Minimum to avoid overlap
            max_valid_distance = min(0.8, (DURATION - 1.0) / n_impulses_1)  # Maximum to fit all impulses
            
            if max_valid_distance < min_valid_distance * 1.4:  # Not enough range for a good difference
                continue
            
            diff = random.uniform(min_valid_distance * 0.3, min_valid_distance * 0.7)
            if random.random() < 0.5 and inter_impulse_distance_1 - diff >= min_valid_distance:
                inter_impulse_distance_2 = inter_impulse_distance_1 - diff
            elif inter_impulse_distance_1 + diff <= max_valid_distance:
                inter_impulse_distance_2 = inter_impulse_distance_1 + diff
            else:
                continue  # Skip if we can't make a valid difference
            
            # Keep other parameters the same
            n_impulses_2 = n_impulses_1
            placement_2 = placement_1
            amplitude_2 = amplitude_1
            
        else:  # amplitude
            # Vary amplitude, keeping sufficient difference but avoiding overlap
            diff = random.uniform(0.1, 0.3)
            if random.random() < 0.5 and amplitude_1 - diff >= 0.1:
                amplitude_2 = amplitude_1 - diff
            elif amplitude_1 + diff <= 0.9:
                amplitude_2 = amplitude_1 + diff
            else:
                continue  # Skip if we can't make a valid difference
            
            # Keep other parameters the same
            n_impulses_2 = n_impulses_1
            placement_2 = placement_1
            inter_impulse_distance_2 = inter_impulse_distance_1
        
        # Generate the two base envelopes based on envelope_type
        if envelope_type == 'gaussian':
            env1_result = generate_gaussian_envelope(n_impulses_1, placement_1, inter_impulse_distance_1, amplitude_1)
        else:  # square
            env1_result = generate_square_envelope(n_impulses_1, placement_1, inter_impulse_distance_1, amplitude_1)
            
        if env1_result[0] is None:
            continue
        env1_filepath, env1 = env1_result
            
        if envelope_type == 'gaussian':
            env2_result = generate_gaussian_envelope(n_impulses_2, placement_2, inter_impulse_distance_2, amplitude_2)
        else:  # square
            env2_result = generate_square_envelope(n_impulses_2, placement_2, inter_impulse_distance_2, amplitude_2)
            
        if env2_result[0] is None:
            continue
        env2_filepath, env2 = env2_result
        
        # Generate morphed envelopes for each alpha value
        morph_filepaths = {}
        morph_params = {}
        valid_set = True
        
        for alpha in ALPHA_VALUES:
            # Skip alpha=0 and alpha=1 as they're already covered by env2 and env1
            if alpha == 0:
                morph_filepaths[alpha] = env2_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_2,
                    'placement': placement_2,
                    'inter_impulse_distance': inter_impulse_distance_2,
                    'amplitude': amplitude_2
                }
                continue
            elif alpha == 1:
                morph_filepaths[alpha] = env1_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_1,
                    'placement': placement_1,
                    'inter_impulse_distance': inter_impulse_distance_1,
                    'amplitude': amplitude_1
                }
                continue
            
            # Calculate morphed parameters
            if morph_type == 'n_impulses':
                n_impulses_alpha = round(alpha * n_impulses_1 + (1 - alpha) * n_impulses_2)
                placement_alpha = placement_1
                inter_impulse_distance_alpha = inter_impulse_distance_1
                amplitude_alpha = amplitude_1
            elif morph_type == 'placement':
                n_impulses_alpha = n_impulses_1
                placement_alpha = alpha * placement_1 + (1 - alpha) * placement_2
                inter_impulse_distance_alpha = inter_impulse_distance_1
                amplitude_alpha = amplitude_1
            elif morph_type == 'inter_impulse_distance':
                n_impulses_alpha = n_impulses_1
                placement_alpha = placement_1
                inter_impulse_distance_alpha = alpha * inter_impulse_distance_1 + (1 - alpha) * inter_impulse_distance_2
                amplitude_alpha = amplitude_1
            else:  # amplitude
                n_impulses_alpha = n_impulses_1
                placement_alpha = placement_1
                inter_impulse_distance_alpha = inter_impulse_distance_1
                # Use power interpolation for amplitude
                amplitude_alpha = power_interpolate_amplitude(amplitude_1, amplitude_2, alpha)
            
            # Generate the morphed envelope based on envelope_type
            if envelope_type == 'gaussian':
                morphed_envelope = gaussian_env(n_impulses_alpha, placement_alpha, inter_impulse_distance_alpha, amplitude_alpha)
            else:  # square
                morphed_envelope = square_env(n_impulses_alpha, placement_alpha, inter_impulse_distance_alpha, amplitude_alpha)
            
            if morphed_envelope is None:
                valid_set = False
                break
            
            # Create filename with parameters
            morph_filename = os.path.join(gaussian_dir, f"{envelope_type}_morph_{morph_type}_a{alpha:.2f}_n{n_impulses_alpha}_p{placement_alpha:.2f}_i{inter_impulse_distance_alpha:.2f}_a{amplitude_alpha:.2f}.wav")
            save_envelope(morphed_envelope, morph_filename)
            
            morph_filepaths[alpha] = morph_filename
            morph_params[alpha] = {
                'n_impulses': n_impulses_alpha,
                'placement': placement_alpha,
                'inter_impulse_distance': inter_impulse_distance_alpha,
                'amplitude': amplitude_alpha
            }
        
        if not valid_set:
            continue  # If any morph couldn't be created, try again
        
        # Assign to train or test with appropriate probability
        split = 'test' if random.random() < 0.1 else 'train'
        
        # Create the full triplet set data
        triplet_set = {
            'envelope_type': envelope_type,
            'morph_type': morph_type,
            'env1_filepath': env1_filepath,
            'env2_filepath': env2_filepath,
            'env1_n_impulses': n_impulses_1,
            'env1_placement': placement_1,
            'env1_inter_impulse_distance': inter_impulse_distance_1,
            'env1_amplitude': amplitude_1,
            'env2_n_impulses': n_impulses_2,
            'env2_placement': placement_2,
            'env2_inter_impulse_distance': inter_impulse_distance_2,
            'env2_amplitude': amplitude_2,
            'split': split
        }
        
        # Add morph filepaths and parameters for each alpha
        for alpha in ALPHA_VALUES:
            triplet_set[f'morph_alpha_{alpha}_filepath'] = morph_filepaths[alpha]
            triplet_set[f'morph_alpha_{alpha}_n_impulses'] = morph_params[alpha]['n_impulses']
            triplet_set[f'morph_alpha_{alpha}_placement'] = morph_params[alpha]['placement']
            triplet_set[f'morph_alpha_{alpha}_inter_impulse_distance'] = morph_params[alpha]['inter_impulse_distance']
            triplet_set[f'morph_alpha_{alpha}_amplitude'] = morph_params[alpha]['amplitude']
        
        return triplet_set
    
    # If we tried max_attempts times and couldn't create a valid set, return None
    return None


def generate_gaussian_simple_dataset(num_sets=100000):
    """Generate a dataset of simple morphs with Gaussian and Square envelopes."""
    data = []
    
    # Morph types and envelope types
    morph_types = ['n_impulses', 'placement', 'inter_impulse_distance', 'amplitude']
    envelope_types = ['gaussian', 'square']
    
    # Try to distribute sets evenly across morph types and envelope types
    target_per_combination = num_sets // (len(morph_types) * len(envelope_types))
    
    # Keep count of successfully generated sets for each combination
    combination_counts = {(et, mt): 0 for et in envelope_types for mt in morph_types}
    
    # Track train/test splits
    split_counts = {'train': 0, 'test': 0}
    
    # Calculate number of plots to create (0.1% of data)
    plot_target = max(1, int(num_sets * 0.001))
    plots_per_combination = {
        'train': {(et, mt): max(1, plot_target // (2 * len(morph_types) * len(envelope_types))) 
                 for et in envelope_types for mt in morph_types},
        'test': {(et, mt): max(1, plot_target // (2 * len(morph_types) * len(envelope_types))) 
                for et in envelope_types for mt in morph_types}
    }
    plot_counts = {
        'train': {(et, mt): 0 for et in envelope_types for mt in morph_types},
        'test': {(et, mt): 0 for et in envelope_types for mt in morph_types}
    }
    
    # Track progress
    total_generated = 0
    total_plots = 0
    attempts = 0
    max_attempts = num_sets * 10  # Allow up to 10x the requested number as attempts
    
    with tqdm(total=num_sets, desc="Generating simple morphs") as pbar:
        while total_generated < num_sets and attempts < max_attempts:
            # Choose the combinations that need more sets
            combinations_needed = [(et, mt) for et, mt in combination_counts.keys() 
                                  if combination_counts[(et, mt)] < target_per_combination]
            
            # If all combinations have reached their targets, choose randomly
            if not combinations_needed:
                envelope_type = random.choice(envelope_types)
                morph_type = random.choice(morph_types)
            else:
                envelope_type, morph_type = random.choice(combinations_needed)
            
            triplet_set = generate_gaussian_simple_morph(morph_type, max_attempts=50, envelope_type=envelope_type)
            attempts += 1
            
            if triplet_set is not None:
                data.append(triplet_set)
                combination_counts[(envelope_type, morph_type)] += 1
                split = triplet_set['split']
                split_counts[split] += 1
                total_generated += 1
                pbar.update(1)
                
                # Decide if we should create a plot
                should_plot = (plot_counts[split][(envelope_type, morph_type)] < plots_per_combination[split][(envelope_type, morph_type)])
                
                if should_plot:
                    plot_counts[split][(envelope_type, morph_type)] += 1
                    total_plots += 1
                    
                    # Get filepaths for all alpha values
                    morph_filepaths = [triplet_set[f'morph_alpha_{alpha}_filepath'] for alpha in ALPHA_VALUES]
                    
                    # Create the plot
                    plot_title = f"{envelope_type.capitalize()} {morph_type.capitalize()} Morphing"
                    plot_filename = os.path.join(
                        plot_dirs[f"{split}_simple"], 
                        f"{envelope_type}_{morph_type}_{plot_counts[split][(envelope_type, morph_type)]}.png"
                    )
                    plot_morph_sequence(morph_filepaths, ALPHA_VALUES, plot_title, plot_filename)
                
                # Print progress periodically
                if total_generated % 1000 == 0:
                    success_rate = total_generated / attempts * 100
                    print(f"\nSuccess rate: {success_rate:.1f}% ({total_generated}/{attempts} attempts)")
                    print(f"Plots created: {total_plots}/{total_generated}")
                    print(f"Split distribution: Train {split_counts['train']}, Test {split_counts['test']}")
    
    # Create DataFrame and separate train/test
    df = pd.DataFrame(data)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # Create simplified dataframes
    train_simple_df = create_simplified_dataframe(train_df)
    test_simple_df = create_simplified_dataframe(test_df)
    
    # Save to CSV
    train_path = os.path.join(output_dir, "train_simple.csv")
    test_path = os.path.join(output_dir, "test_simple.csv")
    
    train_simple_df.to_csv(train_path, index=False)
    test_simple_df.to_csv(test_path, index=False)
    
    # Print statistics
    print(f"\nGenerated {len(df)} simple morph sets:")
    print(f"  Train: {len(train_df)} sets ({len(train_simple_df)} rows)")
    print(f"  Test: {len(test_df)} sets ({len(test_simple_df)} rows)")
    print(f"Morph type distribution:\n{df['morph_type'].value_counts()}")
    print(f"Envelope type distribution:\n{df['envelope_type'].value_counts()}")
    
    return train_simple_df, test_simple_df

###########################################
# Part 4: Gaussian Composite Envelope Dataset Generation
###########################################

def generate_gaussian_composite_morph(config_idx, max_attempts=50, envelope_type='gaussian'):
    """
    Generate a set of envelopes with varying alpha values for the specified composite configuration.
    
    Parameters:
    -----------
    config_idx : int
        Index of the composite configuration to use
    max_attempts : int
        Maximum number of attempts to generate a valid set
    envelope_type : str
        Type of envelope to generate ('gaussian' or 'square')
        
    Returns:
    --------
    dict or None
        Dictionary with envelope data if successful, None if constraints cannot be satisfied.
    """
    config = COMPOSITE_CONFIGURATIONS[config_idx]
    
    # Create a short string representation of the configuration to use in filenames
    config_str = ''.join(['1' if config[param] else '0' for param in ['amplitude', 'placement', 'inter_impulse_distance', 'n_impulses']])
    
    # Count the number of properties being morphed
    num_morphed_props = sum(config.values())
    
    for attempt in range(max_attempts):
        # Initialize base parameters for both envelopes
        # For configurations involving inter_impulse_distance, ensure n_impulses > 1
        if config["inter_impulse_distance"]:
            n_impulses_1 = random.randint(2, 8)  # Start with at least 2 impulses
        else:
            n_impulses_1 = random.randint(1, 8)  # Using 8 as max to allow room for variation
            
        placement_1 = random.uniform(1.0, DURATION - 1.0)
        
        # Choose parameters that are more likely to avoid overlap
        width_estimate = min(0.2, 0.5 / n_impulses_1)
        min_distance = width_estimate * 3  # At least 3 widths separation for gaussians
        
        inter_impulse_distance_1 = random.uniform(min_distance, min(0.8, min_distance * 2))
        amplitude_1 = random.uniform(0.2, 0.8)
        
        # Set up random differences for morphing
        n_impulses_diff = 0
        placement_diff = 0
        inter_impulse_distance_diff = 0
        amplitude_diff = 0
        
        # Generate parameter differences for morphing based on the configuration
        if config["n_impulses"]:
            max_diff = min(4, 10 - n_impulses_1)  # Ensure we don't exceed 10 impulses
            min_diff = min(2, n_impulses_1 - 1)   # Ensure we don't go below 1 impulse
            
            if max_diff >= 2:  # Can increase
                n_impulses_diff = random.randint(2, max_diff)
            elif min_diff >= 2:  # Can decrease
                n_impulses_diff = -random.randint(2, min_diff)
            else:
                continue  # Skip if we can't make a valid difference
        
        if config["placement"]:
            max_diff = min(placement_1 - 1.0, DURATION - placement_1 - 1.0) * 0.8  # 80% of available space
            if max_diff < 0.2:
                continue  # Not enough room to make a good difference
            
            placement_diff = random.uniform(0.2, max_diff)
            if random.random() < 0.5:
                placement_diff = -placement_diff
        
        if config["inter_impulse_distance"]:
            min_valid_distance = width_estimate * 2  # Minimum to avoid overlap
            max_valid_distance = min(0.8, (DURATION - 1.0) / n_impulses_1)  # Maximum to fit all impulses
            
            if max_valid_distance < min_valid_distance * 1.4:  # Not enough range for a good difference
                continue
            
            inter_impulse_distance_diff = random.uniform(min_valid_distance * 0.3, min_valid_distance * 0.7)
            if random.random() < 0.5:
                inter_impulse_distance_diff = -inter_impulse_distance_diff
            
            # Ensure the result stays within valid bounds
            if (inter_impulse_distance_1 + inter_impulse_distance_diff < min_valid_distance or 
                inter_impulse_distance_1 + inter_impulse_distance_diff > max_valid_distance):
                continue
        
        if config["amplitude"]:
            amplitude_diff = random.uniform(0.1, 0.3)
            if random.random() < 0.5:
                amplitude_diff = -amplitude_diff
            
            # Ensure the result stays within valid bounds
            if (amplitude_1 + amplitude_diff < 0.1 or amplitude_1 + amplitude_diff > 0.9):
                continue
        
        # Calculate parameters for envelope 2
        n_impulses_2 = n_impulses_1 + n_impulses_diff if config["n_impulses"] else n_impulses_1
        placement_2 = placement_1 + placement_diff if config["placement"] else placement_1
        inter_impulse_distance_2 = inter_impulse_distance_1 + inter_impulse_distance_diff if config["inter_impulse_distance"] else inter_impulse_distance_1
        amplitude_2 = amplitude_1 + amplitude_diff if config["amplitude"] else amplitude_1
        
        # Generate the two base envelopes based on envelope_type
        if envelope_type == 'gaussian':
            env1_result = generate_gaussian_envelope(n_impulses_1, placement_1, inter_impulse_distance_1, amplitude_1)
        else:  # square
            env1_result = generate_square_envelope(n_impulses_1, placement_1, inter_impulse_distance_1, amplitude_1)
            
        if env1_result[0] is None:
            continue
        env1_filepath, env1 = env1_result
            
        if envelope_type == 'gaussian':
            env2_result = generate_gaussian_envelope(n_impulses_2, placement_2, inter_impulse_distance_2, amplitude_2)
        else:  # square
            env2_result = generate_square_envelope(n_impulses_2, placement_2, inter_impulse_distance_2, amplitude_2)
            
        if env2_result[0] is None:
            continue
        env2_filepath, env2 = env2_result
        
        # Generate morphed envelopes for each alpha value
        morph_filepaths = {}
        morph_params = {}
        valid_set = True
        
        for alpha in ALPHA_VALUES:
            # For alpha=0 and alpha=1, we already have the envelopes
            if alpha == 0:
                morph_filepaths[alpha] = env2_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_2,
                    'placement': placement_2,
                    'inter_impulse_distance': inter_impulse_distance_2,
                    'amplitude': amplitude_2
                }
                continue
            elif alpha == 1:
                morph_filepaths[alpha] = env1_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_1,
                    'placement': placement_1,
                    'inter_impulse_distance': inter_impulse_distance_1,
                    'amplitude': amplitude_1
                }
                continue
            
            # Calculate parameters for this alpha
            # For n_impulses (integer), we need special handling
            if config["n_impulses"]:
                n_impulses_alpha = round(alpha * n_impulses_1 + (1 - alpha) * n_impulses_2)
            else:
                n_impulses_alpha = n_impulses_1  # Keep constant
            
            if config["placement"]:
                placement_alpha = alpha * placement_1 + (1 - alpha) * placement_2
            else:
                placement_alpha = placement_1  # Keep constant
            
            if config["inter_impulse_distance"]:
                inter_impulse_distance_alpha = alpha * inter_impulse_distance_1 + (1 - alpha) * inter_impulse_distance_2
            else:
                inter_impulse_distance_alpha = inter_impulse_distance_1  # Keep constant
            
            if config["amplitude"]:
                # Use power interpolation for amplitude
                amplitude_alpha = power_interpolate_amplitude(amplitude_1, amplitude_2, alpha)
            else:
                amplitude_alpha = amplitude_1  # Keep constant
            
            # Generate the morphed envelope based on envelope_type
            if envelope_type == 'gaussian':
                morphed_envelope = gaussian_env(n_impulses_alpha, placement_alpha, inter_impulse_distance_alpha, amplitude_alpha)
            else:  # square
                morphed_envelope = square_env(n_impulses_alpha, placement_alpha, inter_impulse_distance_alpha, amplitude_alpha)
            
            if morphed_envelope is None:
                valid_set = False
                break
            
            # Create a descriptive filename
            morph_filename = os.path.join(gaussian_dir, f"{envelope_type}_composite_{config_str}_a{alpha:.2f}_n{n_impulses_alpha}_p{placement_alpha:.2f}_i{inter_impulse_distance_alpha:.2f}_a{amplitude_alpha:.2f}.wav")
            save_envelope(morphed_envelope, morph_filename)
            
            morph_filepaths[alpha] = morph_filename
            morph_params[alpha] = {
                'n_impulses': n_impulses_alpha,
                'placement': placement_alpha,
                'inter_impulse_distance': inter_impulse_distance_alpha,
                'amplitude': amplitude_alpha
            }
        
        if not valid_set:
            continue  # If any morph couldn't be created, try again
        
        # Assign to test set
        split = 'test'
        
        # Create a descriptive morph type string
        morph_type_str = '+'.join([param for param, value in config.items() if value])
        
        # Create the full envelope set data
        envelope_set = {
            'envelope_type': envelope_type,
            'config_idx': config_idx,
            'config_str': config_str,
            'morph_type': morph_type_str,
            'num_morphed_props': num_morphed_props,
            'morph_amplitude': int(config["amplitude"]),
            'morph_placement': int(config["placement"]),
            'morph_inter_impulse_distance': int(config["inter_impulse_distance"]),
            'morph_n_impulses': int(config["n_impulses"]),
            'env1_filepath': env1_filepath,
            'env2_filepath': env2_filepath,
            'env1_n_impulses': n_impulses_1,
            'env1_placement': placement_1,
            'env1_inter_impulse_distance': inter_impulse_distance_1,
            'env1_amplitude': amplitude_1,
            'env2_n_impulses': n_impulses_2,
            'env2_placement': placement_2,
            'env2_inter_impulse_distance': inter_impulse_distance_2,
            'env2_amplitude': amplitude_2,
            'split': split
        }
        
        # Add morph filepaths and parameters for each alpha
        for alpha in ALPHA_VALUES:
            envelope_set[f'morph_alpha_{alpha}_filepath'] = morph_filepaths[alpha]
            envelope_set[f'morph_alpha_{alpha}_n_impulses'] = morph_params[alpha]['n_impulses']
            envelope_set[f'morph_alpha_{alpha}_placement'] = morph_params[alpha]['placement']
            envelope_set[f'morph_alpha_{alpha}_inter_impulse_distance'] = morph_params[alpha]['inter_impulse_distance']
            envelope_set[f'morph_alpha_{alpha}_amplitude'] = morph_params[alpha]['amplitude']
        
        return envelope_set
    
    # If we tried max_attempts times and couldn't create a valid set, return None
    return None

def generate_gaussian_composite_dataset(num_sets=10000):
    """Generate a dataset of composite morphs with Gaussian and Square envelopes."""
    data = []
    
    # Total number of configurations
    num_configs = len(COMPOSITE_CONFIGURATIONS)
    envelope_types = ['gaussian', 'square']
    
    # Try to distribute sets evenly across configurations and envelope types
    target_per_combination = num_sets // (num_configs * len(envelope_types))
    
    # Keep count of successfully generated sets for each combination
    combination_counts = {(et, i): 0 for et in envelope_types for i in range(num_configs)}
    
    # Calculate number of plots to create (0.5% of data)
    plot_target = max(1, int(num_sets * 0.005))
    plots_per_combination = max(1, plot_target // (num_configs * len(envelope_types)))
    plot_counts = {(et, i): 0 for et in envelope_types for i in range(num_configs)}
    
    # Track progress
    total_generated = 0
    total_plots = 0
    attempts = 0
    max_attempts = num_sets * 10  # Allow up to 10x the requested number as attempts
    
    with tqdm(total=num_sets, desc="Generating composite morphs") as pbar:
        while total_generated < num_sets and attempts < max_attempts:
            # Choose the combinations that need more sets
            combinations_needed = [(et, i) for et, i in combination_counts.keys() 
                                  if combination_counts[(et, i)] < target_per_combination]
            
            # If all combinations have reached their targets, choose randomly
            if not combinations_needed:
                envelope_type = random.choice(envelope_types)
                config_idx = random.choice(list(range(num_configs)))
            else:
                envelope_type, config_idx = random.choice(combinations_needed)
            
            envelope_set = generate_gaussian_composite_morph(config_idx, envelope_type=envelope_type)
            attempts += 1
            
            if envelope_set is not None:
                data.append(envelope_set)
                combination_counts[(envelope_type, config_idx)] += 1
                total_generated += 1
                pbar.update(1)
                
                # Decide if we should create a plot
                should_plot = (plot_counts[(envelope_type, config_idx)] < plots_per_combination)
                
                if should_plot:
                    plot_counts[(envelope_type, config_idx)] += 1
                    total_plots += 1
                    
                    # Get filepaths for all alpha values
                    morph_filepaths = [envelope_set[f'morph_alpha_{alpha}_filepath'] for alpha in ALPHA_VALUES]
                    
                    # Get a descriptive string for the config
                    config = COMPOSITE_CONFIGURATIONS[config_idx]
                    config_desc = '+'.join([k for k, v in config.items() if v])
                    
                    # Create the plot
                    plot_title = f"{envelope_type.capitalize()} Composite Morphing: {config_desc}"
                    plot_filename = os.path.join(
                        plot_dirs["test_composed"], 
                        f"{envelope_type}_composite_{config_idx}_{plot_counts[(envelope_type, config_idx)]}.png"
                    )
                    plot_morph_sequence(morph_filepaths, ALPHA_VALUES, plot_title, plot_filename)
                
                # Print progress periodically
                if total_generated % 500 == 0:
                    success_rate = total_generated / attempts * 100
                    print(f"\nSuccess rate: {success_rate:.1f}% ({total_generated}/{attempts} attempts)")
                    print(f"Plots created: {total_plots}/{total_generated}")
                    
                    # Print per-configuration progress
                    print("\nProgress per configuration:")
                    for env_type in envelope_types:
                        for i in range(num_configs):
                            print(f"  {env_type} Config {i}: {combination_counts[(env_type, i)]}/{target_per_combination} sets")
    
    # Create DataFrame 
    df = pd.DataFrame(data)
    
    # Create simplified dataframe
    test_composed_df = create_simplified_dataframe(df)
    
    # Save to CSV
    test_path = os.path.join(output_dir, "test_composed.csv")
    test_composed_df.to_csv(test_path, index=False)
    
    # Print statistics
    print(f"\nGenerated {len(df)} composite morph sets:")
    print(f"  Test set with composed morphs: {len(df)} sets ({len(test_composed_df)} rows)")
    
    # Print distribution by configuration and envelope type
    print("\nDistribution by configuration and envelope type:")
    for env_type in envelope_types:
        for i in range(num_configs):
            count = len(df[(df['envelope_type'] == env_type) & (df['config_idx'] == i)])
            config = COMPOSITE_CONFIGURATIONS[i]
            config_desc = '+'.join([k for k, v in config.items() if v])
            print(f"  {env_type} Config {i} ({config_desc}): {count}/{target_per_combination} sets")
    
    return test_composed_df

###########################################
# Part 5: Real-World Audio Dataset Generation
###########################################

def generate_real_world_envelope(audio_type, n_impulses, placement, inter_impulse_distance, amplitude):
    """
    Generate a real-world audio envelope and save it with a descriptive filename.
    
    Returns (filename, envelope) if successful, (None, None) if constraints cannot be satisfied.
    """
    envelope = real_world_env(audio_type, n_impulses, placement, inter_impulse_distance, amplitude)
    
    # If the envelope couldn't be created with the constraints, return None
    if envelope is None:
        return None, None
    
    # Create filename with parameters
    filename = os.path.join(realworld_dir, f"{audio_type}_n{n_impulses}_p{placement:.2f}_i{inter_impulse_distance:.2f}_a{amplitude:.2f}.wav")
    save_envelope(envelope, filename)
    
    return filename, envelope

def generate_real_world_simple_morph(audio_type, morph_type, max_attempts=50):
    """
    Generate a set of real-world audio envelopes with varying alpha values for the specified morph_type.
    
    Parameters:
    -----------
    audio_type : str
        Type of audio envelope to use
    morph_type : str
        Type of morphing to apply ('n_impulses', 'placement', 'inter_impulse_distance', or 'amplitude')
    max_attempts : int
        Maximum number of attempts to generate a valid set
        
    Returns:
    --------
    dict or None
        Dictionary with envelope data if successful, None if constraints cannot be satisfied.
    """
    for attempt in range(max_attempts):
        # Initialize parameters with random values
        # For inter_impulse_distance morph type, ensure n_impulses > 1
        if morph_type == 'inter_impulse_distance':
            n_impulses_1 = random.randint(2, 8)  # Start with at least 2 impulses
        else:
            n_impulses_1 = random.randint(1, 8)  # Using 8 as max to allow room for variation
            
        placement_1 = random.uniform(1.0, DURATION - 1.0)  # Keep further away from edges
        
        # Choose parameters that are more likely to avoid overlap
        width_estimate = min(0.2, 0.5 / n_impulses_1)
        min_distance = width_estimate * 3  # At least 3 widths separation for audio impulses
        
        inter_impulse_distance_1 = random.uniform(min_distance, min(0.8, min_distance * 2))
        amplitude_1 = random.uniform(0.2, 0.8)  # Slightly lower max amplitude to reduce overlap chance
        
        # Set up values for envelope 2 based on the morphing parameter
        if morph_type == 'n_impulses':
            # Need at least 2 steps difference for integer morphing
            max_diff = min(4, 10 - n_impulses_1)  # Ensure we don't exceed 10 impulses
            min_diff = min(2, n_impulses_1 - 1)   # Ensure we don't go below 1 impulse
            
            if max_diff < 2 and min_diff < 2:
                continue  # Skip this attempt if we can't make a valid difference
            
            if random.random() < 0.5 and max_diff >= 2:
                # Increase n_impulses
                diff = random.randint(2, max_diff)
                n_impulses_2 = n_impulses_1 + diff
            elif min_diff >= 2:
                # Decrease n_impulses
                diff = random.randint(2, min_diff)
                n_impulses_2 = n_impulses_1 - diff
            else:
                continue  # Skip if we can't make a valid difference
            
            # Keep other parameters the same
            placement_2 = placement_1
            inter_impulse_distance_2 = inter_impulse_distance_1
            amplitude_2 = amplitude_1
            
        elif morph_type == 'placement':
            # Vary placement, making sure there's room on both sides
            max_diff = min(placement_1 - 1.0, DURATION - placement_1 - 1.0) * 0.8  # 80% of available space
            if max_diff < 0.2:
                continue  # Not enough room to make a good difference
            
            diff = random.uniform(0.2, max_diff)
            if random.random() < 0.5:
                placement_2 = max(0.5, placement_1 - diff)
            else:
                placement_2 = min(DURATION - 0.5, placement_1 + diff)
            
            # Keep other parameters the same
            n_impulses_2 = n_impulses_1
            inter_impulse_distance_2 = inter_impulse_distance_1
            amplitude_2 = amplitude_1
            
        elif morph_type == 'inter_impulse_distance':
            # Need wider range for effective morphing while avoiding overlap
            min_valid_distance = width_estimate * 2  # Minimum to avoid overlap
            max_valid_distance = min(0.8, (DURATION - 1.0) / n_impulses_1)  # Maximum to fit all impulses
            
            if max_valid_distance < min_valid_distance * 1.4:  # Not enough range for a good difference
                continue
            
            diff = random.uniform(min_valid_distance * 0.3, min_valid_distance * 0.7)
            if random.random() < 0.5 and inter_impulse_distance_1 - diff >= min_valid_distance:
                inter_impulse_distance_2 = inter_impulse_distance_1 - diff
            elif inter_impulse_distance_1 + diff <= max_valid_distance:
                inter_impulse_distance_2 = inter_impulse_distance_1 + diff
            else:
                continue  # Skip if we can't make a valid difference
            
            # Keep other parameters the same
            n_impulses_2 = n_impulses_1
            placement_2 = placement_1
            amplitude_2 = amplitude_1
            
        else:  # amplitude
            # Vary amplitude, keeping sufficient difference but avoiding overlap
            diff = random.uniform(0.1, 0.3)
            if random.random() < 0.5 and amplitude_1 - diff >= 0.1:
                amplitude_2 = amplitude_1 - diff
            elif amplitude_1 + diff <= 0.9:
                amplitude_2 = amplitude_1 + diff
            else:
                continue  # Skip if we can't make a valid difference
            
            # Keep other parameters the same
            n_impulses_2 = n_impulses_1
            placement_2 = placement_1
            inter_impulse_distance_2 = inter_impulse_distance_1
        
        # Generate the two base envelopes
        env1_result = generate_real_world_envelope(audio_type, n_impulses_1, placement_1, inter_impulse_distance_1, amplitude_1)
        if env1_result[0] is None:
            continue
        env1_filepath, env1 = env1_result
            
        env2_result = generate_real_world_envelope(audio_type, n_impulses_2, placement_2, inter_impulse_distance_2, amplitude_2)
        if env2_result[0] is None:
            continue
        env2_filepath, env2 = env2_result
        
        # Generate morphed envelopes for each alpha value
        morph_filepaths = {}
        morph_params = {}
        valid_set = True
        
        for alpha in ALPHA_VALUES:
            # Skip alpha=0 and alpha=1 as they're already covered by env2 and env1
            if alpha == 0:
                morph_filepaths[alpha] = env2_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_2,
                    'placement': placement_2,
                    'inter_impulse_distance': inter_impulse_distance_2,
                    'amplitude': amplitude_2
                }
                continue
            elif alpha == 1:
                morph_filepaths[alpha] = env1_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_1,
                    'placement': placement_1,
                    'inter_impulse_distance': inter_impulse_distance_1,
                    'amplitude': amplitude_1
                }
                continue
            
            # Calculate morphed parameters
            if morph_type == 'n_impulses':
                n_impulses_alpha = round(alpha * n_impulses_1 + (1 - alpha) * n_impulses_2)
                placement_alpha = placement_1
                inter_impulse_distance_alpha = inter_impulse_distance_1
                amplitude_alpha = amplitude_1
            elif morph_type == 'placement':
                n_impulses_alpha = n_impulses_1
                placement_alpha = alpha * placement_1 + (1 - alpha) * placement_2
                inter_impulse_distance_alpha = inter_impulse_distance_1
                amplitude_alpha = amplitude_1
            elif morph_type == 'inter_impulse_distance':
                n_impulses_alpha = n_impulses_1
                placement_alpha = placement_1
                inter_impulse_distance_alpha = alpha * inter_impulse_distance_1 + (1 - alpha) * inter_impulse_distance_2
                amplitude_alpha = amplitude_1
            else:  # amplitude
                n_impulses_alpha = n_impulses_1
                placement_alpha = placement_1
                inter_impulse_distance_alpha = inter_impulse_distance_1
                # Use power interpolation for amplitude
                amplitude_alpha = power_interpolate_amplitude(amplitude_1, amplitude_2, alpha)
            
            # Generate the morphed envelope
            morphed_envelope = real_world_env(audio_type, n_impulses_alpha, placement_alpha, inter_impulse_distance_alpha, amplitude_alpha)
            
            if morphed_envelope is None:
                valid_set = False
                break
            
            # Create filename with parameters
            morph_filename = os.path.join(realworld_dir, f"{audio_type}_morph_{morph_type}_a{alpha:.2f}_n{n_impulses_alpha}_p{placement_alpha:.2f}_i{inter_impulse_distance_alpha:.2f}_a{amplitude_alpha:.2f}.wav")
            save_envelope(morphed_envelope, morph_filename)
            
            morph_filepaths[alpha] = morph_filename
            morph_params[alpha] = {
                'n_impulses': n_impulses_alpha,
                'placement': placement_alpha,
                'inter_impulse_distance': inter_impulse_distance_alpha,
                'amplitude': amplitude_alpha
            }
        
        if not valid_set:
            continue  # If any morph couldn't be created, try again
        
        # Create the full envelope set data
        envelope_set = {
            'audio_type': audio_type,
            'morph_type': morph_type,
            'env1_filepath': env1_filepath,
            'env2_filepath': env2_filepath,
            'env1_n_impulses': n_impulses_1,
            'env1_placement': placement_1,
            'env1_inter_impulse_distance': inter_impulse_distance_1,
            'env1_amplitude': amplitude_1,
            'env2_n_impulses': n_impulses_2,
            'env2_placement': placement_2,
            'env2_inter_impulse_distance': inter_impulse_distance_2,
            'env2_amplitude': amplitude_2,
            'split': 'test'  # All real-world examples are test
        }
        
# Add morph filepaths and parameters for each alpha
        for alpha in ALPHA_VALUES:
            envelope_set[f'morph_alpha_{alpha}_filepath'] = morph_filepaths[alpha]
            envelope_set[f'morph_alpha_{alpha}_n_impulses'] = morph_params[alpha]['n_impulses']
            envelope_set[f'morph_alpha_{alpha}_placement'] = morph_params[alpha]['placement']
            envelope_set[f'morph_alpha_{alpha}_inter_impulse_distance'] = morph_params[alpha]['inter_impulse_distance']
            envelope_set[f'morph_alpha_{alpha}_amplitude'] = morph_params[alpha]['amplitude']
        
        return envelope_set
    
    # If we tried max_attempts times and couldn't create a valid set, return None
    return None

def generate_real_world_composite_morph(audio_type, config_idx, max_attempts=50):
    """
    Generate a set of real-world audio envelopes with varying alpha values for the specified composite configuration.
    
    Parameters:
    -----------
    audio_type : str
        Type of audio envelope to use
    config_idx : int
        Index of the composite configuration to use
    max_attempts : int
        Maximum number of attempts to generate a valid set
        
    Returns:
    --------
    dict or None
        Dictionary with envelope data if successful, None if constraints cannot be satisfied.
    """
    config = COMPOSITE_CONFIGURATIONS[config_idx]
    
    # Create a short string representation of the configuration to use in filenames
    config_str = ''.join(['1' if config[param] else '0' for param in ['amplitude', 'placement', 'inter_impulse_distance', 'n_impulses']])
    
    # Count the number of properties being morphed
    num_morphed_props = sum(config.values())
    
    for attempt in range(max_attempts):
        # Initialize base parameters for both envelopes
        # For configurations involving inter_impulse_distance, ensure n_impulses > 1
        if config["inter_impulse_distance"]:
            n_impulses_1 = random.randint(2, 8)  # Start with at least 2 impulses
        else:
            n_impulses_1 = random.randint(1, 8)  # Using 8 as max to allow room for variation
            
        placement_1 = random.uniform(1.0, DURATION - 1.0)
        
        # Choose parameters that are more likely to avoid overlap
        width_estimate = min(0.2, 0.5 / n_impulses_1)
        min_distance = width_estimate * 3  # At least 3 widths separation for audio impulses
        
        inter_impulse_distance_1 = random.uniform(min_distance, min(0.8, min_distance * 2))
        amplitude_1 = random.uniform(0.2, 0.8)
        
        # Set up random differences for morphing
        n_impulses_diff = 0
        placement_diff = 0
        inter_impulse_distance_diff = 0
        amplitude_diff = 0
        
        # Generate parameter differences for morphing based on the configuration
        if config["n_impulses"]:
            max_diff = min(4, 10 - n_impulses_1)  # Ensure we don't exceed 10 impulses
            min_diff = min(2, n_impulses_1 - 1)   # Ensure we don't go below 1 impulse
            
            if max_diff >= 2:  # Can increase
                n_impulses_diff = random.randint(2, max_diff)
            elif min_diff >= 2:  # Can decrease
                n_impulses_diff = -random.randint(2, min_diff)
            else:
                continue  # Skip if we can't make a valid difference
        
        if config["placement"]:
            max_diff = min(placement_1 - 1.0, DURATION - placement_1 - 1.0) * 0.8  # 80% of available space
            if max_diff < 0.2:
                continue  # Not enough room to make a good difference
            
            placement_diff = random.uniform(0.2, max_diff)
            if random.random() < 0.5:
                placement_diff = -placement_diff
        
        if config["inter_impulse_distance"]:
            min_valid_distance = width_estimate * 2  # Minimum to avoid overlap
            max_valid_distance = min(0.8, (DURATION - 1.0) / n_impulses_1)  # Maximum to fit all impulses
            
            if max_valid_distance < min_valid_distance * 1.4:  # Not enough range for a good difference
                continue
            
            inter_impulse_distance_diff = random.uniform(min_valid_distance * 0.3, min_valid_distance * 0.7)
            if random.random() < 0.5:
                inter_impulse_distance_diff = -inter_impulse_distance_diff
            
            # Ensure the result stays within valid bounds
            if (inter_impulse_distance_1 + inter_impulse_distance_diff < min_valid_distance or 
                inter_impulse_distance_1 + inter_impulse_distance_diff > max_valid_distance):
                continue
        
        if config["amplitude"]:
            amplitude_diff = random.uniform(0.1, 0.3)
            if random.random() < 0.5:
                amplitude_diff = -amplitude_diff
            
            # Ensure the result stays within valid bounds
            if (amplitude_1 + amplitude_diff < 0.1 or amplitude_1 + amplitude_diff > 0.9):
                continue
        
        # Calculate parameters for envelope 2
        n_impulses_2 = n_impulses_1 + n_impulses_diff if config["n_impulses"] else n_impulses_1
        placement_2 = placement_1 + placement_diff if config["placement"] else placement_1
        inter_impulse_distance_2 = inter_impulse_distance_1 + inter_impulse_distance_diff if config["inter_impulse_distance"] else inter_impulse_distance_1
        amplitude_2 = amplitude_1 + amplitude_diff if config["amplitude"] else amplitude_1
        
        # Generate the two base envelopes
        env1_result = generate_real_world_envelope(audio_type, n_impulses_1, placement_1, inter_impulse_distance_1, amplitude_1)
        if env1_result[0] is None:
            continue
        env1_filepath, env1 = env1_result
            
        env2_result = generate_real_world_envelope(audio_type, n_impulses_2, placement_2, inter_impulse_distance_2, amplitude_2)
        if env2_result[0] is None:
            continue
        env2_filepath, env2 = env2_result
        
        # Generate morphed envelopes for each alpha value
        morph_filepaths = {}
        morph_params = {}
        valid_set = True
        
        for alpha in ALPHA_VALUES:
            # For alpha=0 and alpha=1, we already have the envelopes
            if alpha == 0:
                morph_filepaths[alpha] = env2_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_2,
                    'placement': placement_2,
                    'inter_impulse_distance': inter_impulse_distance_2,
                    'amplitude': amplitude_2
                }
                continue
            elif alpha == 1:
                morph_filepaths[alpha] = env1_filepath
                morph_params[alpha] = {
                    'n_impulses': n_impulses_1,
                    'placement': placement_1,
                    'inter_impulse_distance': inter_impulse_distance_1,
                    'amplitude': amplitude_1
                }
                continue
            
            # Calculate parameters for this alpha
            # For n_impulses (integer), we need special handling
            if config["n_impulses"]:
                n_impulses_alpha = round(alpha * n_impulses_1 + (1 - alpha) * n_impulses_2)
            else:
                n_impulses_alpha = n_impulses_1  # Keep constant
            
            if config["placement"]:
                placement_alpha = alpha * placement_1 + (1 - alpha) * placement_2
            else:
                placement_alpha = placement_1  # Keep constant
            
            if config["inter_impulse_distance"]:
                inter_impulse_distance_alpha = alpha * inter_impulse_distance_1 + (1 - alpha) * inter_impulse_distance_2
            else:
                inter_impulse_distance_alpha = inter_impulse_distance_1  # Keep constant
            
            if config["amplitude"]:
                # Use power interpolation for amplitude
                amplitude_alpha = power_interpolate_amplitude(amplitude_1, amplitude_2, alpha)
            else:
                amplitude_alpha = amplitude_1  # Keep constant
            
            # Generate the morphed envelope
            morphed_envelope = real_world_env(audio_type, n_impulses_alpha, placement_alpha, inter_impulse_distance_alpha, amplitude_alpha)
            
            if morphed_envelope is None:
                valid_set = False
                break
            
            # Create a descriptive filename
            morph_filename = os.path.join(realworld_dir, f"{audio_type}_composite_{config_str}_a{alpha:.2f}_n{n_impulses_alpha}_p{placement_alpha:.2f}_i{inter_impulse_distance_alpha:.2f}_a{amplitude_alpha:.2f}.wav")
            save_envelope(morphed_envelope, morph_filename)
            
            morph_filepaths[alpha] = morph_filename
            morph_params[alpha] = {
                'n_impulses': n_impulses_alpha,
                'placement': placement_alpha,
                'inter_impulse_distance': inter_impulse_distance_alpha,
                'amplitude': amplitude_alpha
            }
        
        if not valid_set:
            continue  # If any morph couldn't be created, try again
        
        # Create a descriptive morph type string
        morph_type_str = '+'.join([param for param, value in config.items() if value])
        
        # Create the full envelope set data
        envelope_set = {
            'audio_type': audio_type,
            'config_idx': config_idx,
            'config_str': config_str,
            'morph_type': morph_type_str,
            'num_morphed_props': num_morphed_props,
            'morph_amplitude': int(config["amplitude"]),
            'morph_placement': int(config["placement"]),
            'morph_inter_impulse_distance': int(config["inter_impulse_distance"]),
            'morph_n_impulses': int(config["n_impulses"]),
            'env1_filepath': env1_filepath,
            'env2_filepath': env2_filepath,
            'env1_n_impulses': n_impulses_1,
            'env1_placement': placement_1,
            'env1_inter_impulse_distance': inter_impulse_distance_1,
            'env1_amplitude': amplitude_1,
            'env2_n_impulses': n_impulses_2,
            'env2_placement': placement_2,
            'env2_inter_impulse_distance': inter_impulse_distance_2,
            'env2_amplitude': amplitude_2,
            'split': 'test'  # All real-world examples are test
        }
        
        # Add morph filepaths and parameters for each alpha
        for alpha in ALPHA_VALUES:
            envelope_set[f'morph_alpha_{alpha}_filepath'] = morph_filepaths[alpha]
            envelope_set[f'morph_alpha_{alpha}_n_impulses'] = morph_params[alpha]['n_impulses']
            envelope_set[f'morph_alpha_{alpha}_placement'] = morph_params[alpha]['placement']
            envelope_set[f'morph_alpha_{alpha}_inter_impulse_distance'] = morph_params[alpha]['inter_impulse_distance']
            envelope_set[f'morph_alpha_{alpha}_amplitude'] = morph_params[alpha]['amplitude']
        
        return envelope_set
    
    # If we tried max_attempts times and couldn't create a valid set, return None
    return None

def generate_real_world_simple_dataset(num_sets=10000):
    """Generate a dataset of simple morphs with real-world audio envelopes."""
    data = []
    
    # Use all audio types
    audio_types = list(audio_files.keys())
    morph_types = ['n_impulses', 'placement', 'inter_impulse_distance', 'amplitude']
    
    # Try to distribute sets evenly across audio types and morph types
    target_per_combination = num_sets // (len(audio_types) * len(morph_types))
    
    # Keep count of successfully generated sets for each combination
    combination_counts = {(at, mt): 0 for at in audio_types for mt in morph_types}
    
    # Calculate number of plots to create (0.5% of data)
    plot_target = max(1, int(num_sets * 0.005))
    plots_per_combination = max(1, plot_target // (len(audio_types) * len(morph_types)))
    plot_counts = {(at, mt): 0 for at in audio_types for mt in morph_types}
    
    # Pre-cache all audio envelopes
    print("Extracting base audio envelopes...")
    for audio_type in audio_types:
        try:
            extract_audio_envelope(audio_type)
        except Exception as e:
            print(f"Warning: Couldn't extract envelope for {audio_type}: {e}")
    
    # Track progress
    total_generated = 0
    total_plots = 0
    attempts = 0
    max_attempts = num_sets * 10  # Allow up to 10x the requested number as attempts
    
    with tqdm(total=num_sets, desc="Generating real-world simple morphs") as pbar:
        while total_generated < num_sets and attempts < max_attempts:
            # Choose the combinations that need more sets
            combinations_needed = [(at, mt) for at, mt in combination_counts.keys() 
                                  if combination_counts[(at, mt)] < target_per_combination]
            
            # If all combinations have reached their targets, choose randomly
            if not combinations_needed:
                audio_type = random.choice(audio_types)
                morph_type = random.choice(morph_types)
            else:
                audio_type, morph_type = random.choice(combinations_needed)
            
            envelope_set = generate_real_world_simple_morph(audio_type, morph_type)
            attempts += 1
            
            if envelope_set is not None:
                data.append(envelope_set)
                combination_counts[(audio_type, morph_type)] += 1
                total_generated += 1
                pbar.update(1)
                
                # Decide if we should create a plot
                should_plot = (plot_counts[(audio_type, morph_type)] < plots_per_combination)
                
                if should_plot:
                    plot_counts[(audio_type, morph_type)] += 1
                    total_plots += 1
                    
                    # Get filepaths for all alpha values
                    morph_filepaths = [envelope_set[f'morph_alpha_{alpha}_filepath'] for alpha in ALPHA_VALUES]
                    
                    # Create the plot
                    plot_title = f"{audio_type.capitalize()} - {morph_type.capitalize()} Morphing"
                    plot_filename = os.path.join(
                        plot_dirs["test_real_simple"], 
                        f"{audio_type}_{morph_type}_{plot_counts[(audio_type, morph_type)]}.png"
                    )
                    plot_morph_sequence(morph_filepaths, ALPHA_VALUES, plot_title, plot_filename)
                
                # Print progress periodically
                if total_generated % 500 == 0:
                    success_rate = total_generated / attempts * 100
                    print(f"\nSuccess rate: {success_rate:.1f}% ({total_generated}/{attempts} attempts)")
                    print(f"Plots created: {total_plots}/{total_generated}")
    
    # Create DataFrame and simplified dataframe
    df = pd.DataFrame(data)
    test_real_simple_df = create_simplified_dataframe(df)
    
    # Save to CSV
    test_path = os.path.join(output_dir, "test_real_simple.csv")
    test_real_simple_df.to_csv(test_path, index=False)
    
    # Print statistics
    print(f"\nGenerated {len(df)} real-world simple morph sets:")
    print(f"  Test set: {len(df)} sets ({len(test_real_simple_df)} rows)")
    print(f"Audio types distribution:\n{df['audio_type'].value_counts()}")
    print(f"Morph types distribution:\n{df['morph_type'].value_counts()}")
    
    # Print details for each combination
    print("\nCombination distributions:")
    for at in audio_types:
        for mt in morph_types:
            count = len(df[(df['audio_type'] == at) & (df['morph_type'] == mt)])
            print(f"  {at}-{mt}: {count} sets")
    
    return test_real_simple_df

def generate_real_world_composite_dataset(num_sets=10000):
    """Generate a dataset of composite morphs with real-world audio envelopes."""
    data = []
    
    # Use all audio types
    audio_types = list(audio_files.keys())
    
    # Total number of configurations
    num_configs = len(COMPOSITE_CONFIGURATIONS)
    
    # Try to distribute sets evenly across audio types and configurations
    target_per_combination = num_sets // (len(audio_types) * num_configs)
    
    # Keep count of successfully generated sets for each combination
    combination_counts = {(at, ci): 0 for at in audio_types for ci in range(num_configs)}
    
    # Calculate number of plots to create (0.5% of data)
    plot_target = max(1, int(num_sets * 0.005))
    plots_per_combination = max(1, plot_target // (len(audio_types) * num_configs))
    plot_counts = {(at, ci): 0 for at in audio_types for ci in range(num_configs)}
    
    # Track progress
    total_generated = 0
    total_plots = 0
    attempts = 0
    max_attempts = num_sets * 10  # Allow up to 10x the requested number as attempts
    
    with tqdm(total=num_sets, desc="Generating real-world composite morphs") as pbar:
        while total_generated < num_sets and attempts < max_attempts:
            # Choose the combinations that need more sets
            combinations_needed = [(at, ci) for at, ci in combination_counts.keys() 
                                  if combination_counts[(at, ci)] < target_per_combination]
            
            # If all combinations have reached their targets, choose randomly
            if not combinations_needed:
                audio_type = random.choice(audio_types)
                config_idx = random.choice(range(num_configs))
            else:
                audio_type, config_idx = random.choice(combinations_needed)
            
            envelope_set = generate_real_world_composite_morph(audio_type, config_idx)
            attempts += 1
            
            if envelope_set is not None:
                data.append(envelope_set)
                combination_counts[(audio_type, config_idx)] += 1
                total_generated += 1
                pbar.update(1)
                
                # Decide if we should create a plot
                should_plot = (plot_counts[(audio_type, config_idx)] < plots_per_combination)
                
                if should_plot:
                    plot_counts[(audio_type, config_idx)] += 1
                    total_plots += 1
                    
                    # Get filepaths for all alpha values
                    morph_filepaths = [envelope_set[f'morph_alpha_{alpha}_filepath'] for alpha in ALPHA_VALUES]
                    
                    # Get a descriptive string for the config
                    config = COMPOSITE_CONFIGURATIONS[config_idx]
                    config_desc = '+'.join([k for k, v in config.items() if v])
                    
                    # Create the plot
                    plot_title = f"{audio_type.capitalize()} - Composite Morphing: {config_desc}"
                    plot_filename = os.path.join(
                        plot_dirs["test_real_composed"], 
                        f"{audio_type}_composite_{config_idx}_{plot_counts[(audio_type, config_idx)]}.png"
                    )
                    plot_morph_sequence(morph_filepaths, ALPHA_VALUES, plot_title, plot_filename)
                
                # Print progress periodically
                if total_generated % 500 == 0:
                    success_rate = total_generated / attempts * 100
                    print(f"\nSuccess rate: {success_rate:.1f}% ({total_generated}/{attempts} attempts)")
                    print(f"Plots created: {total_plots}/{total_generated}")
    
    # Create DataFrame and simplified dataframe
    df = pd.DataFrame(data)
    test_real_composed_df = create_simplified_dataframe(df)
    
    # Save to CSV
    test_path = os.path.join(output_dir, "test_real_composed.csv")
    test_real_composed_df.to_csv(test_path, index=False)
    
    # Print statistics
    print(f"\nGenerated {len(df)} real-world composite morph sets:")
    print(f"  Test set: {len(df)} sets ({len(test_real_composed_df)} rows)")
    print(f"Audio types distribution:\n{df['audio_type'].value_counts()}")
    
    # Print details for each combination
    print("\nCombination distributions:")
    for at in audio_types:
        for ci in range(num_configs):
            count = len(df[(df['audio_type'] == at) & (df['config_idx'] == ci)])
            config = COMPOSITE_CONFIGURATIONS[ci]
            config_desc = '+'.join([k for k, v in config.items() if v])
            print(f"  {at}-{config_desc}: {count} sets")
    
    return test_real_composed_df

###########################################
# Part 6: Utility Functions
###########################################

def create_simplified_dataframe(df):
    """Create a simplified dataframe from the original dataframe."""
    # Create a new empty dataframe with the desired columns
    if 'audio_type' in df.columns:
        # Real-world audio envelopes
        simplified_df = pd.DataFrame(columns=['audio_type', 'morph_type', 'env1_filepath', 'env2_filepath', 'alpha', 'env3_filepath', 'split'])
    else:
        # Gaussian envelopes
        simplified_df = pd.DataFrame(columns=['morph_type', 'env1_filepath', 'env2_filepath', 'alpha', 'env3_filepath', 'split'])
    
    # Process each row in the original dataframe
    for idx, row in df.iterrows():
        # For each alpha value, create a new row
        alpha_values_to_use = ALPHA_VALUES
        
        # For test sets, only use alpha=0.25, 0.5, 0.75 (not 0 and 1)
        if row['split'] == 'test':
            alpha_values_to_use = [0.25, 0.5, 0.75]
        
        for alpha in alpha_values_to_use:
            # Create new row
            if 'audio_type' in df.columns:
                new_row = {
                    'audio_type': row['audio_type'],
                    'morph_type': row['morph_type'],
                    'env1_filepath': row['env1_filepath'],
                    'env2_filepath': row['env2_filepath'],
                    'alpha': alpha,
                    'env3_filepath': row[f'morph_alpha_{alpha}_filepath'],
                    'split': row['split']
                }
            else:
                new_row = {
                    'morph_type': row['morph_type'],
                    'env1_filepath': row['env1_filepath'],
                    'env2_filepath': row['env2_filepath'],
                    'alpha': alpha,
                    'env3_filepath': row[f'morph_alpha_{alpha}_filepath'],
                    'split': row['split']
                }
            
            # Append new row to simplified dataframe
            simplified_df = pd.concat([simplified_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return simplified_df


def copy_sample_plots_to_summary(output_dir, summary_dir):
    """Copy a sample of plots from each category to a summary directory."""
    os.makedirs(summary_dir, exist_ok=True)
    
    # For each plot directory, copy a few representative samples
    for plot_type, plot_dir in plot_dirs.items():
        # Get all png files in the directory
        png_files = [f for f in os.listdir(plot_dir) if f.endswith('.png')]
        
        # If there are files, select a few samples (up to 3)
        if png_files:
            samples = png_files[:min(3, len(png_files))]
            
            # Copy each sample to the summary directory with a prefix
            for i, sample in enumerate(samples):
                src = os.path.join(plot_dir, sample)
                dst = os.path.join(summary_dir, f"{plot_type}_{i+1}_{sample}")
                shutil.copy(src, dst)

def create_dataset_summary(output_dir):
    """Create a summary of the generated datasets."""
    summary_file = os.path.join(output_dir, "dataset_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write(f"Audio Envelope Morphing Dataset Summary\n")
        f.write(f"Generated on: {timestamp}\n\n")
        
        # List all CSV files
        csv_files = [file for file in os.listdir(output_dir) if file.endswith('.csv')]
        
        f.write(f"Dataset Files:\n")
        for csv_file in csv_files:
            path = os.path.join(output_dir, csv_file)
            df = pd.read_csv(path)
            f.write(f"- {csv_file}: {len(df)} rows\n")
        
        f.write(f"\nDataset Structure:\n")
        f.write(f"1. Train set (90k) of simple Gaussian and Square envelopes (one parameter changes)\n")
        f.write(f"2. Test set (10k) of simple Gaussian and Square envelopes (one parameter changes)\n")
        f.write(f"3. Test set (10k) of composed Gaussian and Square envelopes (2+ parameters change)\n")
        f.write(f"4. Test set (10k) of real-world audio envelopes and square waves (one parameter changes)\n")
        
        f.write(f"\nSample plots are available in the 'plots_summary' directory\n")
        
        f.write(f"\nFile Format:\n")
        f.write(f"Each CSV contains the following columns:\n")
        f.write(f"- morph_type: Type of parameter(s) being morphed\n")
        f.write(f"- env1_filepath: Path to envelope 1 (alpha=1)\n")
        f.write(f"- env2_filepath: Path to envelope 2 (alpha=0)\n")
        f.write(f"- alpha: Morphing parameter (for train: 0, 0.25, 0.5, 0.75, 1; for test: 0.25, 0.5, 0.75)\n")
        f.write(f"- env3_filepath: Path to morphed envelope\n")
        f.write(f"- split: Dataset split ('train' or 'test')\n")
        f.write(f"For real-world datasets, there's an additional 'audio_type' column.\n")

###########################################
# Part 7: Main Execution
###########################################

def main():
    """Main execution function."""
    print(f"Starting audio envelope morphing dataset generation ({timestamp})...")
    print(f"Output directory: {output_dir}")
    
    # 1. Generate train set (90k) of simple Gaussian envelopes
    # 2. Generate test set (10k) of simple Gaussian envelopes
    print("\n=== Generating Gaussian Simple Envelopes (90k train, 10k test) ===")
    train_simple_df, test_simple_df = generate_gaussian_simple_dataset(20000)
    
    # 3. Generate test set (10k) of composed Gaussian envelopes
    print("\n=== Generating Gaussian Composite Envelopes (10k test) ===")
    test_composed_df = generate_gaussian_composite_dataset(2000)
    
    # 4. Generate test set (10k) of real-world audio envelopes
    print("\n=== Generating Real-World Simple Envelopes (10k test) ===")
    test_real_simple_df = generate_real_world_simple_dataset(2000)

    # Create a summary directory with representative plots
    print("\n=== Creating Summary ===")
    summary_plots_dir = os.path.join(output_dir, "plots_summary")
    copy_sample_plots_to_summary(output_dir, summary_plots_dir)

    plot_all_directories()
    
    # Create a summary file
    create_dataset_summary(output_dir)
    
    print(f"\nDataset Generation Complete!")
    print(f"Output directory: {output_dir}")
    print(f"Generated files:")
    print(f"- train_simple_gaussian.csv: {len(train_simple_df)} rows")
    print(f"- test_simple_gaussian.csv: {len(test_simple_df)} rows")
    print(f"- test_composed_gaussian.csv: {len(test_composed_df)} rows")
    print(f"- test_real_simple.csv: {len(test_real_simple_df)} rows")
    # print(f"- test_real_composed.csv: {len(test_real_composed_df)} rows")
    print(f"Sample plots available in: {summary_plots_dir}")
    print(f"Summary info available in: {os.path.join(output_dir, 'dataset_summary.txt')}")

if __name__ == "__main__":
    main()
