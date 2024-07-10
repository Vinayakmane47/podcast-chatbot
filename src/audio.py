from subprocess import CalledProcessError, run
from pyannote.audio import Pipeline
import os
import time
import torch
from openai import OpenAI
from dotenv import load_dotenv 
from pydub import AudioSegment
import math

class AudioTranscription:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI()
        
    def transcribe_audio(self, file_path): 
        audio_file = open(file_path, "rb")
        transcript = self.client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file 
        )
        return transcript
    
    def save_list_to_txt(self, file_path, text_list):
        with open(file_path, 'w') as file:
            for item in text_list:
                file.write(f"{item}\n")
    
    def split_large_audio(self, file_path, max_size_mb=25):
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            audio = AudioSegment.from_wav(file_path)
            duration_ms = len(audio)
            num_chunks = math.ceil(file_size_mb / max_size_mb)
            chunk_duration_ms = duration_ms / num_chunks
            
            chunks = []
            for i in range(num_chunks):
                start_ms = i * chunk_duration_ms
                end_ms = start_ms + chunk_duration_ms
                chunk = audio[start_ms:end_ms]
                chunk_file_path = f"{file_path}_part_{i}.wav"
                chunk.export(chunk_file_path, format="wav")
                chunks.append(chunk_file_path)
            
            return chunks
        else:
            return [file_path]
    
    def combine_audio_files(self, output_dir): 
        count = 1 
        for speaker_folder in os.listdir(output_dir):
            speaker_path = os.path.join(output_dir, speaker_folder)
            
            if os.path.isdir(speaker_path):
                wav_files = [f for f in os.listdir(speaker_path) if f.endswith('.wav')]
                
                if wav_files:
                    combined_audio = AudioSegment.empty()
                    
                    for wav_file in wav_files:
                        wav_path = os.path.join(speaker_path, wav_file)
                        audio = AudioSegment.from_wav(wav_path)
                        combined_audio += audio
                    
                    output_path = os.path.join(speaker_path, f"combined_speaker_{count}.wav")
                    combined_audio.export(output_path, format="wav")
                    
                    print(f"Combined audio saved to: {output_path}")
                    count += 1 
    
    def split_audio(self, input_file, output_file, start, end):
        length = end - start
        cmd = ["ffmpeg", "-ss", str(start), "-i", input_file, "-t", str(length), "-vn", "-acodec", "pcm_s16le", "-ar", "48000", "-ac", "1", output_file]
        try:
            run(cmd, capture_output=True, check=True).stdout
        except CalledProcessError as e:
            raise RuntimeError(f"FFMPEG error {str(e)}")
    
    def assign_speaker(self, output_dir):
        transcripts = []
        
        for speaker_folder in os.listdir(output_dir):
            speaker_path = os.path.join(output_dir, speaker_folder)
            
            if os.path.isdir(speaker_path):
                combined_files = [f for f in os.listdir(speaker_path) if f.startswith('combined_speaker') and f.endswith('.wav')]
                
                for combined_file in combined_files:
                    combined_file_path = os.path.join(speaker_path, combined_file)
                    audio_chunks = self.split_large_audio(combined_file_path)
                    
                    combined_transcript = ""
                    for chunk in audio_chunks:
                        transcript = self.transcribe_audio(chunk)
                        combined_transcript += transcript
                    
                    speaker_name = speaker_folder.replace("speaker_", "")
                    formatted_transcript = f"{speaker_name}: {combined_transcript}"
                    transcripts.append(formatted_transcript)
        
        return transcripts
    
    def main(self, input_wav, output_dir):
        start_time = time.time()
        
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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
            self.split_audio(input_wav, filename, turn.start, turn.end)
            count += 1
        
        self.combine_audio_files(output_dir)
        transcripts = self.assign_speaker(output_dir)
        self.save_list_to_txt(file_path="data.txt", text_list=transcripts)
        for transcript in transcripts:
            print(transcript)
        
        end_time = time.time()
        print(f"Time taken to run this algorithm is {end_time - start_time} seconds")
        
        return transcripts
# Example usage
if __name__ == "__main__":
    audio_processor = AudioTranscription()
    input_wav = "/Users/vinayak/AI/projects/voice-assistant-bot/youtube2/neet_dhruv.mp3"
    output_dir = "output"
    audio_processor.main(input_wav, output_dir)
