from utils.audio_processor import split_audio_into_chunks
from core.transcriber import transcribe_all

src = "https://youtu.be/Ty8gcCKuwNI?si=Fp4y_5MpPlhGg6G0"

chunks = split_audio_into_chunks(src)
transcription = transcribe_all(chunks) 

print(transcription)