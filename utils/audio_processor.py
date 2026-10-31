import yt_dlp
from pydub import AudioSegment
import os

DOWNLOAD_DIR = 'downloads'

os.makedirs(DOWNLOAD_DIR, exist_ok = True)

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

# audio into chunks in 10mins audio chunks
def chunk_audio(wav_path:str, chunk_minutes:int = 10) -> list:
    """Chunk a WAV audio file into smaller segments of specified duration in minutes."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_length_ms = chunk_minutes * 60 * 1000  # Convert minutes to milliseconds
    chunks = []
    
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i + chunk_length_ms]
        chunk_filename = f"{os.path.splitext(wav_path)[0]}_chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_filename, format="wav")
        chunks.append(chunk_filename)
    
    return chunks

# pathV = download_youtube_audio("https://www.youtube.com/watch?v=Ty8gcCKuwNI")
# pathW = convert_to_wav(pathV)
# chunks = chunk_audio(pathW, chunk_minutes=10)

# print(f"Downloaded and processed audio from YouTube. Chunks created: {chunks}")

def split_audio_into_chunks(source:str) -> list:
    """Download, convert, and chunk YouTube audio/any local file."""
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected youtube URL Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to wav...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path, chunk_minutes=10)
    print(f"Audio processing complete. {len(chunks)} chunks created.")
    return chunks

