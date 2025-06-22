import os
import numpy as np
import soundfile as sf
from scipy.signal import resample

def resample_audio(data, original_rate, target_rate):
    """
    Resample audio data to the target sample rate.
    
    Args:
    - data: Input audio data (numpy array)
    - original_rate: Original sample rate
    - target_rate: Target sample rate
    
    Returns:
    - Resampled audio data
    """
    # Calculate the scaling factor
    scale = target_rate / original_rate
    
    # Calculate new number of samples
    n = int(np.ceil(len(data) * scale))
    
    # Resample the data
    if data.ndim == 1:  # Mono audio
        resampled = resample(data, n)
    else:  # Stereo audio
        resampled = np.zeros((n, data.shape[1]))
        for channel in range(data.shape[1]):
            resampled[:, channel] = resample(data[:, channel], n)
    
    return resampled

def combine_audio(vocals_file, instrumental_file, output_file, vocals_volume=0.7, instrumental_volume=0.5):
    """
    Combine vocals and instrumental audio files, 
    with automatic resampling and volume control.
    
    Args:
    - vocals_file: Path to the vocals audio file
    - instrumental_file: Path to the instrumental audio file
    - output_file: Path for the output combined audio file
    - vocals_volume: Volume multiplier for vocals (default 0.7)
    - instrumental_volume: Volume multiplier for instrumental (default 0.5)
    """
    try:
        # Get the current working directory
        current_dir = os.getcwd()

        # Construct the full file paths for the input files
        vocals_path = os.path.join(current_dir, vocals_file)
        instrumental_path = os.path.join(current_dir, instrumental_file)

        # Read vocal audio
        vocals_data, vocals_sample_rate = sf.read(vocals_path)
        
        # Read instrumental audio
        instrumental_data, instrumental_sample_rate = sf.read(instrumental_path)

        # Determine target sample rate (use the higher of the two)
        target_sample_rate = max(vocals_sample_rate, instrumental_sample_rate)

        # Resample if necessary
        if vocals_sample_rate != target_sample_rate:
            print(f"Resampling vocals from {vocals_sample_rate} to {target_sample_rate} Hz")
            vocals_data = resample_audio(vocals_data, vocals_sample_rate, target_sample_rate)
            vocals_sample_rate = target_sample_rate

        if instrumental_sample_rate != target_sample_rate:
            print(f"Resampling instrumental from {instrumental_sample_rate} to {target_sample_rate} Hz")
            instrumental_data = resample_audio(instrumental_data, instrumental_sample_rate, target_sample_rate)
            instrumental_sample_rate = target_sample_rate

        # Normalize both tracks to stereo if needed
        if vocals_data.ndim == 1:
            vocals_data = np.column_stack((vocals_data, vocals_data))
        
        if instrumental_data.ndim == 1:
            instrumental_data = np.column_stack((instrumental_data, instrumental_data))

        # Pad or trim vocals to match instrumental length
        if len(vocals_data) < len(instrumental_data):
            vocals_data = np.pad(vocals_data, 
                                 ((0, len(instrumental_data) - len(vocals_data)), (0, 0)), 
                                 mode='constant')
        else:
            # If vocals are longer, trim to instrumental length
            vocals_data = vocals_data[:len(instrumental_data)]

        # Mix the audio with custom volume levels
        combined_data = (vocals_volume * vocals_data) + (instrumental_volume * instrumental_data)

        # Normalize to prevent clipping
        max_amplitude = np.max(np.abs(combined_data))
        if max_amplitude > 1:
            combined_data = combined_data / max_amplitude

        # Construct the output file path
        output_path = os.path.join(current_dir, output_file)

        # Ensure the file has a .wav extension
        if not output_path.lower().endswith('.wav'):
            output_path += '.wav'

        # Write the combined audio
        sf.write(output_path, combined_data, target_sample_rate)
        
        print(f"Combined audio saved as {output_path}")
        print(f"Final audio length: {len(combined_data) / target_sample_rate:.2f} seconds")
        print(f"Sample rate: {target_sample_rate} Hz")
        print(f"Vocals volume: {vocals_volume}")
        print(f"Instrumental volume: {instrumental_volume}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
# Example usage
if __name__ == "__main__":
    try:
        vocals_file = input("Enter the name of the vocals audio file: ")
        instrumental_file = input("Enter the name of the instrumental audio file: ")
        output_file = input("Enter the desired name for the output file (default .wav extension): ")
        
        # Use a default name if no input is provided
        if not output_file.strip():
            output_file = "combined_output.wav"
        
        # Optional volume adjustment
        vocals_volume = float(input("Enter vocals volume (default 0.7): ") or 0.7)
        instrumental_volume = float(input("Enter instrumental volume (default 0.5): ") or 0.5)
        
        combine_audio(vocals_file, instrumental_file, output_file, vocals_volume, instrumental_volume)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")