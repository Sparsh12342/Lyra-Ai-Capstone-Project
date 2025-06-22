import numpy as np
import librosa
import soundfile as sf
import os

def extract_vocal_activity(vocals_path, 
                            threshold=0.01,  # Lowered threshold
                            min_duration=0.05,  # Shorter minimum duration
                            gap_tolerance=0.2):  # Allow longer gaps between vocal segments
    """
    Detect vocal activity with improved continuity.
    
    Parameters:
    - vocals_path: Path to the vocals MP3 file
    - threshold: Lower amplitude threshold for detecting vocal presence
    - min_duration: Minimum duration of vocal segments (in seconds)
    - gap_tolerance: Maximum gap between vocal segments to merge
    
    Returns:
    - List of tuples (start_time, end_time) representing vocal segments
    """
    # Load the vocals file
    y, sr = librosa.load(vocals_path, mono=True)
    
    # Calculate the root mean square (RMS) energy
    rms = librosa.feature.rms(y=y)[0]
    
    # Convert RMS to time-based segments
    time_per_frame = len(y) / (sr * len(rms))
    vocal_segments = []
    
    in_vocal_segment = False
    segment_start = 0
    last_vocal_end = 0
    
    for i, energy in enumerate(rms):
        current_time = i * time_per_frame
        
        if energy > threshold and not in_vocal_segment:
            # Check if this segment is close to a previous segment
            if (last_vocal_end > 0 and 
                current_time - last_vocal_end <= gap_tolerance):
                # Continue the previous segment
                in_vocal_segment = True
            else:
                # Start of a new vocal segment
                segment_start = current_time
                in_vocal_segment = True
        
        elif energy <= threshold and in_vocal_segment:
            # End of a vocal segment
            segment_end = current_time
            
            # Check if segment is long enough
            if segment_end - segment_start >= min_duration:
                vocal_segments.append((segment_start, segment_end))
                last_vocal_end = segment_end
            
            in_vocal_segment = False
    
    # Merge very close segments
    merged_segments = []
    if vocal_segments:
        current_start, current_end = vocal_segments[0]
        
        for start, end in vocal_segments[1:]:
            # If gap between segments is small, merge them
            if start - current_end <= gap_tolerance:
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged_segments.append((current_start, current_end))
    
    return merged_segments

def sync_tts_with_vocals(tts_path, vocals_path, output_path):
    """
    Synchronize TTS audio with vocal segments from a song.
    """
    # Load TTS and vocals audio
    tts_y, tts_sr = librosa.load(tts_path, mono=True)
    vocals_y, vocals_sr = librosa.load(vocals_path, mono=True)
    
    # Get vocal activity segments
    vocal_segments = extract_vocal_activity(vocals_path)
    
    # Initialize output audio
    output_y = np.zeros_like(vocals_y)
    
    # Current position in TTS audio
    tts_position = 0
    
    # Fill output audio with TTS during vocal segments
    for start, end in vocal_segments:
        # Convert time to samples
        start_sample = int(start * vocals_sr)
        end_sample = int(end * vocals_sr)
        
        # Calculate how much TTS to use
        segment_length = end_sample - start_sample
        
        # Get TTS segment
        if tts_position + segment_length <= len(tts_y):
            tts_segment = tts_y[tts_position:tts_position + segment_length]
            
            # Adjust volume if needed
            tts_segment *= 0.7  # Reduce volume to blend better
            
            # Place TTS in output at vocal segment
            output_y[start_sample:end_sample] = tts_segment
            
            # Update TTS position
            tts_position += segment_length
    
    # Save the synchronized audio
    sf.write(output_path, output_y, vocals_sr)
    
    print(f"Synchronized audio saved to {output_path}")

def main():
    # Get current working directory
    current_dir = os.getcwd()
    
    # Prompt for input files
    tts_filename = input("Enter the filename of the TTS MP3 file: ")
    tts_path = os.path.join(current_dir, tts_filename)
    
    vocals_filename = input("Enter the filename of the vocals MP3 file: ")
    vocals_path = os.path.join(current_dir, vocals_filename)
    
    # Prompt for output filename
    output_filename = input("Enter the desired output filename (including .wav extension): ")
    output_path = os.path.join(current_dir, output_filename)
    
    # Synchronize audio
    sync_tts_with_vocals(tts_path, vocals_path, output_path)

if __name__ == "__main__":
    main()