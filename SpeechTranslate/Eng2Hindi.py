from pydub import AudioSegment
import speech_recognition as sr
from translate import Translator
from gtts import gTTS

def convert_mp3_to_wav(input_mp3, output_wav):
    audio = AudioSegment.from_mp3(input_mp3)
    audio.export(output_wav, format="wav")


def recognize_speech(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
    try:
        recognized_text = recognizer.recognize_google(audio_data)
        return recognized_text
    except sr.UnknownValueError:
        return "Speech recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google's Speech Recognition API; {e}"
    


def translate_text(text, target_language):
    if text is not None:
        translator = Translator(to_lang=target_language)
        translation = translator.translate(text)
        return translation
    else:
        return "No text to translate"
    

def convert_text_to_speech(text, lang_code, output_path):
    if text != "No text to translate":
        tts = gTTS(text=text, lang=lang_code)
        tts.save(output_path)


def speech_to_speech_pipeline(input_mp3, output_mp3, target_language='hi'):
    # Step 1: Convert MP3 to WAV
    wav_file = "temp_speech.wav"
    convert_mp3_to_wav(input_mp3, wav_file)
 
    # Step 2: Recognize Speech
    recognized_text = recognize_speech(wav_file)
    print("Recognized Speech:")
    print(recognized_text)
 
    # Step 3: Translate Recognized Text
    translated_text = translate_text(recognized_text, target_language)
    print("Translated Text:")
    print(translated_text)
 
    # Step 4: Convert Translated Text to Speech
    convert_text_to_speech(translated_text, target_language, output_mp3)
    audio = AudioSegment.from_mp3(output_mp3)
    return audio



if __name__=="__main__":
    
    # Pipeline for speech translation
    audioFile = "audio.mp3"


    input_audio_file = audioFile# replace it with your input file
    output_audio_file = "translated_"+audioFile#translated_speech.mp3"
    speech_to_speech_pipeline(input_audio_file, output_audio_file, target_language='hi')
    