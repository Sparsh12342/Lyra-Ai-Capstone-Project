import sys
import os
import numpy as np
import librosa
import soundfile as sf
import torch
from scipy.signal import resample
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
from gtts import gTTS
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

# Part 1: YouTube Transcript Retrieval and Translation

def remove_music(text):
    """Function to replace music indicators with an empty string"""
    words_to_remove = ['[Music]', '♪', '[music]', '[Música]', '[música]']
    for word in words_to_remove:
        text = text.replace(word, '')
    return text

def get_transcript(video_id):
    """Get transcript from YouTube video ID"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        lyrics = " ".join([line['text'] for line in transcript])
        fixed_lyrics = remove_music(lyrics)
        return fixed_lyrics
    except Exception as e:
        print(f"Error: Could not retrieve transcript for video ID: {video_id}. {str(e)}")
        return None

def get_language_choice():
    """Get language choice from user"""
    languages = {
        "Chinese (Simplified)": "zh-CN",
        "Spanish": "es",
        "Arabic": "ar",
        "Hindi": "hi",
        "English": "en",
        "Japanese": "ja",
        "Korean": "ko",
        "French": "fr",
        "German": "de",
        "Italian": "it"
    }
    
    print("Choose a language: ")
    for idx, language in enumerate(languages.keys(), start=1):
        print(f"{idx}. {language}")
    
    while True:
        try:
            language_choice = int(input("Enter the number of your language: "))
            if 1 <= language_choice <= len(languages):
                selected_language = list(languages.values())[language_choice - 1]
                return selected_language
            else:
                print(f"Please enter a number between 1 and {len(languages)}")
        except ValueError:
            print("Please enter a valid number")

def translate_lyrics(lyrics, target_lang):
    """Translate lyrics to target language"""
    try:
        translator = GoogleTranslator(source="auto", target=target_lang)
        return translator.translate(lyrics)
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return None

def text_to_audio(translated_text, target_lang):
    """Convert translated text to audio"""
    title = input("Enter a title for your audio file (no spaces): ").strip()
    
    if not title:
        print("Title cannot be empty. Using default title.")
        title = "translated_audio"
    
    title = title.replace(" ", "_")
    
    audio_file = f"{title}.mp3"
    counter = 1
    while os.path.exists(audio_file):
        audio_file = f"{title}_{counter}.mp3"
        counter += 1

    try:
        tts = gTTS(text=translated_text, lang=target_lang)
        tts.save(audio_file)
        print(f"Audio saved as {audio_file}")
        return audio_file
    except Exception as e:
        print(f"TTS error: {str(e)}")
        return None

# Part 2: Audio Separation using Demucs


def separate_vocals_and_instrumental(input_file, output_dir="separated_output", model_name="htdemucs"):
    """Separate vocals from instrumental using Demucs"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = os.path.splitext(os.path.basename(input_file))[0]
    
    print(f"Loading Demucs model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name)
    model.to(device)
    
    print(f"Loading audio file: {input_file}")
    wav = AudioFile(input_file).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    
    wav = wav.unsqueeze(0)
    
    wav = wav.to(device)
    
    print("Separating audio sources...")
    with torch.no_grad():
        sources = apply_model(model, wav)
    
    sources = sources.cpu()
    
    if "vocals" in model.sources:
        vocals_idx = model.sources.index("vocals")
        
        vocals_path = os.path.join(output_dir, f"{filename}_vocals.wav")
        vocals_audio = sources[0, vocals_idx]
        print(f"Saving vocals to: {vocals_path}")
        save_audio(vocals_audio, vocals_path, model.samplerate)
        
        instrumental_path = os.path.join(output_dir, f"{filename}_instrumental.wav")
        instrumental_audio = torch.zeros_like(sources[0, 0])
        for i, source_name in enumerate(model.sources):
            if i != vocals_idx:
                instrumental_audio += sources[0, i]
        
        print(f"Saving instrumental to: {instrumental_path}")
        save_audio(instrumental_audio, instrumental_path, model.samplerate)
        
        print("Separation complete!")
        print(f"Files saved to:")
        print(f"  Vocals: {vocals_path}")
        print(f"  Instrumental: {instrumental_path}")
        
        return vocals_path, instrumental_path
    else:
        print("Error: This model doesn't have a 'vocals' source.")
        return None, None


# Part 3: Vocal Activity Detection and TTS Synchronization


def extract_vocal_activity(vocals_path, 
                           threshold=0.01, 
                           min_duration=0.05,  
                           gap_tolerance=0.2):  
    """
    Detect vocal activity with improved continuity.
    """
    y, sr = librosa.load(vocals_path, mono=True)
    
    rms = librosa.feature.rms(y=y)[0]
    
    time_per_frame = len(y) / (sr * len(rms))
    vocal_segments = []
    
    in_vocal_segment = False
    segment_start = 0
    last_vocal_end = 0
    
    for i, energy in enumerate(rms):
        current_time = i * time_per_frame
        
        if energy > threshold and not in_vocal_segment:
            if (last_vocal_end > 0 and 
                current_time - last_vocal_end <= gap_tolerance):
                in_vocal_segment = True
            else:
                segment_start = current_time
                in_vocal_segment = True
        
        elif energy <= threshold and in_vocal_segment:
            segment_end = current_time
            
            if segment_end - segment_start >= min_duration:
                vocal_segments.append((segment_start, segment_end))
                last_vocal_end = segment_end
            
            in_vocal_segment = False
    
    if in_vocal_segment:
        segment_end = len(rms) * time_per_frame
        if segment_end - segment_start >= min_duration:
            vocal_segments.append((segment_start, segment_end))
    
    merged_segments = []
    if vocal_segments:
        current_start, current_end = vocal_segments[0]
        
        for start, end in vocal_segments[1:]:
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
    tts_y, tts_sr = librosa.load(tts_path, mono=True)
    vocals_y, vocals_sr = librosa.load(vocals_path, mono=True)
    
    vocal_segments = extract_vocal_activity(vocals_path)
    
    output_y = np.zeros_like(vocals_y)
    
    tts_position = 0
    
    for start, end in vocal_segments:
        start_sample = int(start * vocals_sr)
        end_sample = int(end * vocals_sr)
        
        segment_length = end_sample - start_sample
        
        if tts_position + segment_length <= len(tts_y):
            tts_segment = tts_y[tts_position:tts_position + segment_length]
            
            tts_segment *= 0.7  
            
            output_y[start_sample:end_sample] = tts_segment
            
            tts_position += segment_length
    
    sf.write(output_path, output_y, vocals_sr)
    
    print(f"Synchronized audio saved to {output_path}")
    return output_path


# Part 4: Audio Mixing


def resample_audio(data, original_rate, target_rate):
    """Resample audio data to the target sample rate."""
    scale = target_rate / original_rate
    
    n = int(np.ceil(len(data) * scale))
    
    if data.ndim == 1:  
        resampled = resample(data, n)
    else:  
        resampled = np.zeros((n, data.shape[1]))
        for channel in range(data.shape[1]):
            resampled[:, channel] = resample(data[:, channel], n)
    
    return resampled

def combine_audio(vocals_file, instrumental_file, output_file, vocals_volume=0.7, instrumental_volume=0.5):
    """
    Combine vocals and instrumental audio files with automatic resampling and volume control.
    """
    try:
        current_dir = os.getcwd()

        vocals_data, vocals_sample_rate = sf.read(vocals_file)
        
        instrumental_data, instrumental_sample_rate = sf.read(instrumental_file)

        target_sample_rate = max(vocals_sample_rate, instrumental_sample_rate)

        if vocals_sample_rate != target_sample_rate:
            print(f"Resampling vocals from {vocals_sample_rate} to {target_sample_rate} Hz")
            vocals_data = resample_audio(vocals_data, vocals_sample_rate, target_sample_rate)
            vocals_sample_rate = target_sample_rate

        if instrumental_sample_rate != target_sample_rate:
            print(f"Resampling instrumental from {instrumental_sample_rate} to {target_sample_rate} Hz")
            instrumental_data = resample_audio(instrumental_data, instrumental_sample_rate, target_sample_rate)
            instrumental_sample_rate = target_sample_rate

        if vocals_data.ndim == 1:
            vocals_data = np.column_stack((vocals_data, vocals_data))
        
        if instrumental_data.ndim == 1:
            instrumental_data = np.column_stack((instrumental_data, instrumental_data))

        if len(vocals_data) < len(instrumental_data):
            vocals_data = np.pad(vocals_data, 
                                 ((0, len(instrumental_data) - len(vocals_data)), (0, 0)), 
                                 mode='constant')
        else:
            vocals_data = vocals_data[:len(instrumental_data)]

        combined_data = (vocals_volume * vocals_data) + (instrumental_volume * instrumental_data)

        max_amplitude = np.max(np.abs(combined_data))
        if max_amplitude > 1:
            combined_data = combined_data / max_amplitude

        if not output_file.lower().endswith('.wav'):
            output_file += '.wav'

        sf.write(output_file, combined_data, target_sample_rate)
        
        print(f"Combined audio saved as {output_file}")
        print(f"Final audio length: {len(combined_data) / target_sample_rate:.2f} seconds")
        print(f"Sample rate: {target_sample_rate} Hz")
        print(f"Vocals volume: {vocals_volume}")
        print(f"Instrumental volume: {instrumental_volume}")
        
        return output_file
        
    except Exception as e:
        print(f"An error occurred while combining audio: {e}")
        return None


# Main Program Workflow


def main():
    print("=" * 7)
    print("LyraAI")
    print("=" * 7)
    
    # Step 1: Get YouTube video ID and transcript
    video_id = input("Enter YouTube video ID: ")
    transcript = get_transcript(video_id)
    
    if not transcript:
        print("Failed to retrieve transcript. Exiting.")
        return
    
    # Step 2: Get target language and translate
    target_lang = get_language_choice()
    translated_text = translate_lyrics(transcript, target_lang)
    
    if not translated_text:
        print("Translation failed. Exiting.")
        return
    
    # Display original and translated text
    print("\nOriginal Text:")
    print(transcript[:200] + "..." if len(transcript) > 200 else transcript)
    print("\nTranslated Text:")
    print(translated_text[:200] + "..." if len(translated_text) > 200 else translated_text)
    
    # Step 3: Generate TTS audio
    tts_file = text_to_audio(translated_text, target_lang)
    
    if not tts_file:
        print("TTS generation failed. Exiting.")
        return
    
    # Step 4: Get original song to separate
    print("\nNow we need the original song to separate into vocals and instrumental.")
    song_file = input("Enter the path to the original song file (.mp3 or .wav): ")
    
    if not os.path.exists(song_file):
        print(f"File {song_file} does not exist. Exiting.")
        return
    
    # Step 5: Separate song into vocals and instrumental
    vocals_file, instrumental_file = separate_vocals_and_instrumental(song_file)
    
    if not vocals_file or not instrumental_file:
        print("Audio separation failed. Exiting.")
        return
    
    # Step 6: Synchronize TTS with vocals
    print("\nSynchronizing TTS with vocal segments...")
    sync_output = os.path.join("separated_output", "synced_tts.wav")
    synced_file = sync_tts_with_vocals(tts_file, vocals_file, sync_output)
    
    if not synced_file:
        print("Audio synchronization failed. Exiting.")
        return
    
    # Step 7: Combine synced TTS with instrumental
    print("\nCombining synced TTS with instrumental...")
    
    # Ask for volume levels
    vocals_volume = float(input("Enter TTS volume (default 0.7): ") or 0.7)
    instrumental_volume = float(input("Enter instrumental volume (default 0.5): ") or 0.5)
    
    # Generate final output filename
    output_name = input("Enter the final output filename (default: final_translated_song.wav): ")
    if not output_name:
        output_name = "final_translated_song.wav"
    
    final_file = combine_audio(synced_file, instrumental_file, output_name, vocals_volume, instrumental_volume)
    
    if final_file:
        print("\n" + "=" * 70)
        print(f"Process complete! Your translated song is ready: {final_file}")
        print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")