import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")  # Default to 'base' if not set

_model = None

def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print(f"Whisper model '{WHISPER_MODEL}' loaded successfully.")
    return _model

def transcribe_chunk(chunk_path: str, translate: bool = False) -> str:
    # Transcribe a single audio chunk using the Whisper model.

    # Args:
    #     chunk_path (str): Path to the audio chunk file.
    #     translate (bool): Whether to translate the transcription to English.
    #     False -> Transcribe in the original language, True -> Translate to English.

    # Returns:
    #     str: The transcribed text.

    model = load_model()
    
    task = "translate" if translate else "transcribe"
    print(f"Transcribing chunk: {chunk_path} with task: {task}")
    result = model.transcribe(chunk_path, task=task)

    # there are multiple keys in the result, but we only need the transcribed text for now
    return result["text"] 

def transcribe_all(chunks: list, translate: bool = False) -> str:
    # Transcribe a list of audio chunks and concatenate the results.

    full_transcription = ""
    for chunk_path in chunks:
        transcription = transcribe_chunk(chunk_path, translate=translate)
        full_transcription += transcription + " "  # Add a space between chunks

    return full_transcription.strip()  # Remove any trailing whitespace

