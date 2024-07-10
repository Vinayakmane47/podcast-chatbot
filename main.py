import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
import time, os
from logging import Logger
from pipeline import TranscriptionPipeline
from tempfile import NamedTemporaryFile
from audio import AudioTranscription 
from chains import invoke_chain
import os
import shutil
from youtube import youtube_to_mp3

directory = 'output'

# Check if the directory exists
if os.path.exists(directory):
    # Remove the directory and its contents
    shutil.rmtree(directory)

question_prompt = """
This following content contains some important information. I want to get that information out from this 
context but I don't know which exact question to ask you so that I can get the information.
 Provide me all possible questions to ask so that I can get meaningful information from this context. 
 Only generate questions and no other information. but limit questions to 10 
"""

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
def generate_response(transcript, prompt):
    response = invoke_chain(transcript, prompt)
    return response

# Function to simulate recording
def simulate_recording():
    # Simulate a recording process (replace with actual logic to record audio)
    recording_time = 5  # Simulating a 5-second recording
    for _ in range(recording_time):
        time.sleep(1)  # Simulate each second of recording

# Function to convert YouTube video to MP3


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

    # Get transcript if available
    transcript = st.session_state.get("transcript", "")

    # Display assistant response in chat message container
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = generate_response(transcript, prompt)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar options
st.sidebar.title("Options")

# Radio button for choosing upload or record audio
upload_option = st.sidebar.radio("Choose an option", ["Upload Audio File", "Record Audio", "YouTube URL"])

# Environment variables
API_KEY = os.getenv("OPENAI_API_KEY")

# Conditional rendering based on radio button selection
if upload_option == "Upload Audio File":
    uploaded_file = st.sidebar.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file is not None and "transcript" not in st.session_state:
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        with st.spinner("Transcribing audio..."):
            transcript = AudioTranscription().main(input_wav=temp_file_path, output_dir="output")
        
        os.remove(temp_file_path)
        
        if transcript and len(transcript) > 0:
            st.session_state.transcript = transcript  # Store transcript in session state
            sample_questions = invoke_chain(transcript, question_prompt)
            
            # Split the string into a list of questions
            questions_list = sample_questions.split('\n')
            
            # Remove any empty strings from the list
            questions_list = [question for question in questions_list if question]
            
            st.session_state.questions_list = questions_list
            st.sidebar.success("Done processing")
        else:
            st.error("Transcription failed")

elif upload_option == "Record Audio":
    duration = st.sidebar.slider("Select recording duration (seconds)", min_value=1, max_value=60, value=10)
    if st.sidebar.button("Record and Transcribe") and "transcript" not in st.session_state:
        with st.spinner("Recording audio..."):
            minutes = pipeline.process_live_recording(duration)
        
        if minutes:
            st.session_state.transcript = minutes  # Store transcript in session state
            sample_questions = invoke_chain(minutes, question_prompt)
            
            # Split the string into a list of questions
            questions_list = sample_questions.split('\n')
            
            # Remove any empty strings from the list
            questions_list = [question for question in questions_list if question]
            
            st.session_state.questions_list = questions_list
            st.sidebar.success("Done processing")
        else:
            st.error("Transcription failed")

elif upload_option == "YouTube URL":
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    if st.sidebar.button("Process"):
        if youtube_url:
            with st.spinner("Converting YouTube video to MP3..."):
                mp3_file_path = youtube_to_mp3(youtube_url, "youtube")  # Replace "output" with your desired output directory
                
            if mp3_file_path:
                with st.spinner("Transcribing audio..."):
                    transcript = AudioTranscription().main(input_wav=mp3_file_path, output_dir="output")
                
                if transcript and len(transcript) > 0:
                    st.session_state.transcript = transcript  # Store transcript in session state
                    sample_questions = invoke_chain(transcript, question_prompt)
                    
                    # Split the string into a list of questions
                    questions_list = sample_questions.split('\n')
                    
                    # Remove any empty strings from the list
                    questions_list = [question for question in questions_list if question]
                    
                    st.session_state.questions_list = questions_list
                    st.sidebar.success("Done processing")
                else:
                    st.error("Transcription failed")
            else:
                st.error("Conversion to MP3 failed")

if "questions_list" in st.session_state:
    # Slider to select the number of questions to display
    num_questions = st.sidebar.slider("Number of questions to display", min_value=1, max_value=len(st.session_state.questions_list), value=3)
    
    # Display the selected number of questions
    st.sidebar.markdown("### Frequently Asked Questions")
    for i in range(num_questions):
        st.sidebar.markdown(f"- {st.session_state.questions_list[i]}")
