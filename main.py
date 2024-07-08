import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import time, os
from logging import Logger
from pipeline import TranscriptionPipeline
from tempfile import NamedTemporaryFile

# Initialize session state for chat history and database
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage("Hello! I'm a podcast assistant. Ask me anything about your podcast Videos."),
        HumanMessage("hi")
    ]

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to invoke LangChain or any other processing logic based on user input
def invoke_chain(prompt, messages):
    # Your logic to process user input and generate a response goes here
    pass

# Function to simulate recording
def simulate_recording():
    # Simulate a recording process (replace with actual logic to record audio)
    recording_time = 5  # Simulating a 5-second recording
    for _ in range(recording_time):
        time.sleep(1)  # Simulate each second of recording

# Streamlit app code starts here
st.title("Podcast Chatbot")

# Display chat messages from history on app rerun
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = invoke_chain(prompt, st.session_state.messages)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar options
st.sidebar.title("Options")

# Radio button for choosing upload or record audio
upload_option = st.sidebar.radio("Choose an option", ["Upload Audio File", "Record Audio"])

# Environment variables
API_KEY = os.getenv("OPENAI_API_KEY")
#if not API_KEY:
    #logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

pipeline = TranscriptionPipeline(API_KEY, compute_type="int8")

# Conditional rendering based on radio button selection
if upload_option == "Upload Audio File":
    uploaded_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        with st.spinner("Transcribing audio..."):
            minutes = pipeline.process_audio_file(temp_file_path)

            
            
        
        os.remove(temp_file_path)
        
        if minutes:
            st.subheader("Meeting Minutes")
            st.text(minutes)
        else:
            st.error("Transcription failed")

elif upload_option == "Record Audio":
    duration = st.sidebar.slider("Select recording duration (seconds)", min_value=1, max_value=60, value=10)
    if st.sidebar.button("Record and Transcribe"):
        with st.spinner("Recording audio..."):
            minutes = pipeline.process_live_recording(duration)
        
        if minutes:
            st.subheader("Meeting Minutes")
            st.text(minutes)
        else:
            st.error("Transcription failed")

# End of Streamlit app code
