import os
import logging
import asyncio
import streamlit as st
from faster_whisper import WhisperModel
import whisper
import sounddevice as sd
import numpy as np
import wavio
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from tempfile import NamedTemporaryFile
import torch
#from test3 import prepare_transcript
#from chains import invoke_chain

# Setup logging
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("Audio Transcription and Meeting Minutes Generator")

st.sidebar.title("Options")
upload_option = st.sidebar.radio("Choose an option", ["Upload Audio File", "Record Audio"])

if upload_option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        with st.spinner("Transcribing audio..."):
            minutes = ""
            #transcript = prepare_transcript(input_wav=temp_file_path , 
                                            #output_dir="output")
            #minutes = invoke_chain(text=transcript , question="summarize this text")
        os.remove(temp_file_path)
        
        if minutes:
            st.subheader("Meeting Minutes")
            st.text(minutes)
        else:
            st.error("Transcription failed")

elif upload_option == "Record Audio":
    duration = st.slider("Select recording duration (seconds)", min_value=1, max_value=60, value=10)
    if st.button("Record and Transcribe"):
        with st.spinner("Recording audio..."):
            minutes = pipeline.process_live_recording(duration)
        
        if minutes:
            st.subheader("Meeting Minutes")
            st.text(minutes)
        else:
            st.error("Transcription failed")
