import os
import argparse
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

def separate_vocals_and_instrumental(input_file, output_dir, model_name="htdemucs"):
    """
    Separate vocals from instrumental using Demucs, outputting only vocals and combined instrumental files
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory to save output files
        model_name (str): Name of the Demucs model to use
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename without extension
    filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Load the model
    print(f"Loading Demucs model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name)
    model.to(device)
    
    # Load the audio file
    print(f"Loading audio file: {input_file}")
    wav = AudioFile(input_file).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    
    # Add a batch dimension
    wav = wav.unsqueeze(0)
    
    # Move to device
    wav = wav.to(device)
    
    # Separate sources
    print("Separating audio sources...")
    with torch.no_grad():
        sources = apply_model(model, wav)
    
    # Move back to CPU
    sources = sources.cpu()
    
    # Check if vocals source exists in the model
    if "vocals" in model.sources:
        vocals_idx = model.sources.index("vocals")
        
        # Save the vocals file
        vocals_path = os.path.join(output_dir, f"{filename}_vocals.wav")
        vocals_audio = sources[0, vocals_idx]
        print(f"Saving vocals to: {vocals_path}")
        save_audio(vocals_audio, vocals_path, model.samplerate)
        
        # Create and save the instrumental file (all non-vocal sources combined)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Separate vocals and instrumental from an audio file using Demucs.")
    parser.add_argument("input_file", help="Path to the input audio file.")
    parser.add_argument("--output_dir", default="separated_output", help="Directory to save output files.")
    parser.add_argument("--model", default="htdemucs", help="Demucs model to use (htdemucs, mdx_extra, etc.)")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: File {args.input_file} does not exist.")
        exit(1)
    
    separate_vocals_and_instrumental(args.input_file, args.output_dir, args.model)