import sys
from youtube_transcript_api import YouTubeTranscriptApi # Beginning of transcript retrieval code

def remove_music(text):   # Function to replace '[Music]' with an empty string

    for word in words_to_remove:
        text = text.replace(word, '')
    return text

words_to_remove = ['[Music]', '♪'] # Add any keywords you'd like to remove to this variable

def get_id(): # Function to get video ID from user
    video_id_input = str(input("Enter your video ID: "))
    return video_id_input

video_id = get_id()

try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
except Exception as e:
    print(f"Error: Could not retrieve transcript for video ID: {video_id}. The video may not exist or has no available transcript.")
    sys.exit()

transcript = YouTubeTranscriptApi.get_transcript(video_id)

# Combine transcript text
lyrics = " ".join([line['text'] for line in transcript])
fixed_lyrics = remove_music(lyrics)

from deep_translator import GoogleTranslator # Beginning of translation code

def translate_lyrics(fixed_lyrics, target_lang):  # Function to translate lyrics
    translator = GoogleTranslator(source="auto", target=target_lang)
    return translator.translate(fixed_lyrics)

def get_language_choice():
    languages = {
        "Chinese (Simplified)": "zh-CN",
        "Spanish": "es",
        "Arabic": "ar",
        "Hindi": "hi",
        "English": "en"
    }
    
    print("Choose a language: ")
    for idx, language in enumerate(languages.keys(), start=1):
        print(f"{idx}. {language}")
    
    language_choice = int(input("Enter the number of your language: "))
    
    # Ensure the input is valid
    if language_choice in range(1, len(languages) + 1):
        selected_language = list(languages.values())[language_choice - 1]
        return selected_language
    else:
        print("Invalid, please try again with a valid number choice.")
        sys.exit()
        return get_language_choice()

target_lang = get_language_choice() # Get the user's choice

# Translate the lyrics
translated_lyrics = translate_lyrics(fixed_lyrics, target_lang)

# Display the translation
print(f"Orginal Language: {fixed_lyrics}")
print(f"Translated: {translated_lyrics}")

from gtts import gTTS # Beginning of translated text to audio code
import os

def text_to_audio(translated_text, target_lang):
        # Prompt user for a title for their audio file
    title = input("Enter a title for your audio file (no spaces): ").strip()
    
    # Ensure the title is not empty
    if not title:
        print("Title cannot be empty. Using default title.")
        title = "translated_audio"
    
    # Check if the file already exists, and append a number if it does
    audio_file = f"{title}.mp3"
    counter = 1
    while os.path.exists(audio_file):  # Check if the file exists
        audio_file = f"{title}_{counter}.mp3"
        counter += 1


    tts = gTTS(text=translated_text, lang=target_lang)
    tts.save(audio_file)
    print(f"Audio saved as {audio_file}")
    return audio_file

audio_file = text_to_audio(translated_lyrics, target_lang)