import sounddevice as sd
import numpy as np
import wavio
from faster_whisper import WhisperModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tempfile import NamedTemporaryFile
import torch
import os 
import logging

# Setup logging
from dotenv import load_dotenv
load_dotenv()

from logging import Logger 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure logging to file
file_handler = logging.FileHandler('main.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



# Transcription and meeting minutes pipeline
class AudioTranscription:
    def __init__(self, model_size="large-v3", device="auto", compute_type="auto"):
        if device == "auto":
            device = "cpu"  # Default to CPU
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                print("MPS is available, but not supported by faster_whisper. Using CPU instead.")
                
        
        # Ensure device is either "cpu" or "cuda"
        device = "cpu" if device not in ["cpu", "cuda"] else device
        #device = torch.device("mps")
        # Set compute_type based on device
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Using device: {device}, compute_type: {compute_type}")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        #self.model = whisper.load_model("medium" ,device=torch.device("mps"))
    
    def transcribe_audio_file(self, file_path, beam_size=5):
        try:
            logger.info(f"Transcribing audio file: {file_path}")
            segments, info = self.model.transcribe(file_path, beam_size=beam_size)
            transcription = " ".join([segment.text for segment in segments])
            logger.info("Transcription completed successfully.")
            return transcription
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return None
    
    def record_audio(self, duration=10, sample_rate=16000):
        try:
            logger.info("Recording audio...")
            audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype=np.int16)
            sd.wait()  # Wait until recording is finished
            with NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                wavio.write(audio_file.name, audio_data, sample_rate, sampwidth=2)
                logger.info(f"Audio recorded successfully: {audio_file.name}")
                return audio_file.name
        except Exception as e:
            logger.error(f"Error during audio recording: {e}")
            return None

class MeetingMinutes:
    def __init__(self, api_key=None, model_type="openai"):
        self.model_type = model_type
        if model_type == "openai":
            if not api_key:
                raise ValueError("API key is required for OpenAI model")
            os.environ["OPENAI_API_KEY"] = api_key
            self.model = ChatOpenAI(model="gpt-3.5-turbo-0125")
        elif model_type == "ollama":
            self.model = ChatOllama(model="llama3")
        else:
            raise ValueError("Invalid model type. Choose 'openai' or 'ollama'")
        
        self.output_parser = StrOutputParser()

    def convert_to_minutes(self, transcription):
        try:
            logger.info("Converting transcription to meeting minutes...")
            prompt = ChatPromptTemplate.from_template(
                """Convert the following transcription into meaningful podcast Summary, 
                Identify each speakers and what is their summary like this Speaker_1 : summary , 
                if speaker name is mentioned then replace speaker_1 with their name

                {transcription}"""
            )
            chain = prompt | self.model | self.output_parser
            response = chain.invoke({"transcription": transcription})
            logger.info("Conversion to minutes completed successfully.")
            return response
        except Exception as e:
            logger.error(f"Error during conversion to minutes: {e}")
            return None

class TranscriptionPipeline:
    def __init__(self, api_key=None, model_type="openai", model_size="large-v3", device="auto", compute_type="auto"):
        self.transcriber = AudioTranscription(model_size, device, compute_type)
        #self.minutes_converter = MeetingMinutes(api_key, model_type)
        
        if model_type == "openai" and api_key is None:
            raise ValueError("API key is required for OpenAI model")
        
        self.minutes_converter = MeetingMinutes(api_key, model_type)

    def process_audio_file(self, file_path):
        transcription = self.transcriber.transcribe_audio_file(file_path)
        with open("test.txt","w") as txt_file : 
                txt_file.write(transcription)
        if transcription:
            minutes = self.minutes_converter.convert_to_minutes(transcription)
            logger.info(f"Podcast Summary : {minutes}")
            return minutes
        return None

    def process_live_recording(self, duration=10):
        audio_file = self.transcriber.record_audio(duration)
        if audio_file:
            return self.process_audio_file(audio_file)
        return None
