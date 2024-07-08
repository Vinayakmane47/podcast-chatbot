from subprocess import CalledProcessError, run
from pyannote.audio import Pipeline
import os
import time
import torch
from openai import OpenAI
from dotenv import load_dotenv 
import time 
import os
from pydub import AudioSegment
load_dotenv()

client = OpenAI()
# Define your transcribe_audio function
def transcribe_audio(file_path): 
    audio_file = open(file_path, "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        response_format="text",  # Default output format is json; if you want in json format, just comment out response format
        file=audio_file 
    )
    return transcript




def combine_audio_files(output_dir): 
    # Counter to keep track of the combined files
    count = 1 

    # Loop through each speaker folder in the output directory
    for speaker_folder in os.listdir(output_dir):
        speaker_path = os.path.join(output_dir, speaker_folder)
        
        if os.path.isdir(speaker_path):
            # List all WAV files in the speaker folder
            wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
            
            if wav_files:
                # Initialize an empty AudioSegment
                combined_audio = AudioSegment.empty()
                
                # Load and concatenate all WAV files
                for wav_file in wav_files:
                    wav_path = os.path.join(speaker_path, wav_file)
                    audio = AudioSegment.from_wav(wav_path)
                    combined_audio += audio
                
                # Export the combined audio to a new file
                output_path = os.path.join(speaker_path, f"combined_speaker_{count}.wav")
                combined_audio.export(output_path, format="wav")
                
                print(f"Combined audio saved to: {output_path}")
                count += 1 




# Define the split_audio function
def split_audio(input_file, output_file, start, end):
    length = end - start
    cmd = ["ffmpeg", "-ss", str(start), "-i", input_file, "-t", str(length), "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", output_file]
    try:
        run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"FFMPEG error {str(e)}")

# Define the assign_speaker function
def assign_speaker(output_dir):
    transcripts = []
    
    # Loop through each speaker folder in the output directory
    for speaker_folder in os.listdir(output_dir):
        speaker_path = os.path.join(output_dir, speaker_folder)
        
        if os.path.isdir(speaker_path):
            # Look for the combined_speaker file
            combined_files = [f for f in os.listdir(speaker_path) if f.startswith('combined_speaker') and f.endswith('.wav')]
            
            for combined_file in combined_files:
                combined_file_path = os.path.join(speaker_path, combined_file)
                
                # Transcribe the audio
                transcript = transcribe_audio(combined_file_path)
                
                # Extract the speaker folder name from the file path
                speaker_name = speaker_folder.replace("speaker_", "")
                
                # Format the transcript with the speaker name
                formatted_transcript = f"{speaker_name}: {transcript}"
                
                # Add to the list of transcripts
                transcripts.append(formatted_transcript)
    
    return transcripts

# Main function to run the entire process
def main(input_wav, output_dir):
    start_time = time.time()
    
    hf_token = os.environ["HUGGINGFACE_TOKEN"]
    
    # Load the pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )

    # Send pipeline to GPU (when available)
    pipeline.to(torch.device("mps"))

    # Perform diarization
    diarization = pipeline(input_wav)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    count = 10001

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start}s stop={turn.end}s speaker_{speaker}")

        speaker_dir = f"{output_dir}/speaker_{speaker}/"
        if not os.path.isdir(speaker_dir):
            os.mkdir(speaker_dir)

        filename = os.path.join(speaker_dir, f"convo-{count}.wav")
        split_audio(input_wav, filename, turn.start, turn.end)
        count += 1

    # Combine and transcribe audio
    combine_audio_files(output_dir)
    transcripts = assign_speaker(output_dir)
    for transcript in transcripts:
        print(transcript)

    end_time = time.time()
    print(f"Time taken to run this algorithm is {end_time - start_time} seconds")

# Example usage
input_wav = "/Users/vinayak/AI/projects/voice-assistant-bot/environment_debate.wav"
output_dir = "output2"
main(input_wav, output_dir)
