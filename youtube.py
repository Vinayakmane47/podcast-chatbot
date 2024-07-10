import os
import shutil
import librosa
import openai
import soundfile as sf
import yt_dlp as youtube_dl  # Use yt_dlp instead of youtube_dl
from yt_dlp.utils import DownloadError
from dotenv import load_dotenv
load_dotenv()
def find_audio_files(path, extension=".mp3"):
    """Recursively find all files with extension in path."""
    audio_files = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(extension):
                audio_files.append(os.path.join(root, f))

    return audio_files

def youtube_to_mp3(youtube_url: str, output_dir: str) -> str:
    """Download the audio from a youtube video, save it to output_dir as an .mp3 file.

    Returns the filename of the savied video.
    """

    # config
    ydl_config = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "verbose": True,
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Downloading video from {youtube_url}")

    try:
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])
    except DownloadError:
        # weird bug where youtube-dl fails on the first download, but then works on second try... hacky ugly way around it.
        with youtube_dl.YoutubeDL(ydl_config) as ydl:
            ydl.download([youtube_url])

    audio_filename = find_audio_files(output_dir)[0]
    return audio_filename



youtube_url = "https://www.youtube.com/watch?v=g1pb2aK2we4"
youtube_url_mohak = "https://www.youtube.com/watch?v=apxOPVWYSDg"
youtube_url_15 = "https://youtu.be/DK0dxlttFM4?si=dI0X-tTti2Rn1J3Q"
youtube_url_dhruv = "https://www.youtube.com/watch?v=BFU9eSKO_t4"
outputs_dir = "youtube2/"

#youtube_to_mp3(youtube_url_dhruv,outputs_dir)
