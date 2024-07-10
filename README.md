# PODCAST CHATBOT
A Podcast Chatbot built using LangChain, OpenAI  LLM, OpenAI  Whisper , pynnote audio model , Huggingface  and Streamlit. This project converts YouTube videos to MP3, diarizes the audio, identifies speakers in the podcast, and provides sample questions to extract information from the podcast. You can interact with the podcast content through a chat interface.


### Features
1. **YouTube to MP3 Conversion:** Converts YouTube videos to MP3 format.
2. **Diarization:** Separates and identifies speakers in the podcast.
3. **Speaker Identification:** Recognizes different speakers in the podcast.
4. **Sample Questions Generation:** Provides sample questions to query the podcast content.
5. **Chat Interface:** Allows users to interact with the podcast content through a chat interface.

### Step-by-Step Project Flow: 

#### Audio Input Options
1. YouTube URL: User provides a YouTube video link.
2. Upload Audio File: User uploads an audio file.
3. Record Audio: User records audio directly.
   
***YouTube to Audio Conversion:*** 
- If the user provides a YouTube URL, the video is converted to an audio file and saved locally.
Audio Processing

***Diarization:*** 
- The uploaded or recorded audio file undergoes diarization using the Hugging Face library to identify and separate different speakers.

***Speaker-Specific Audio Segmentation:***
- Create separate folders for each identified speaker.
- Split the audio file into segments corresponding to each speaker and store these segments in the respective speaker folders.

***Combining Speaker Audio:*** 
- Combine all audio segments for each speaker into a single audio file for that speaker.

***Handling Large Files:*** 
- If the combined audio file size for any speaker exceeds 25 MB, split the file into smaller chunks to ensure compatibility with the Whisper model.

***Transcription with Whisper Model:***
- Each combined audio file (or chunk) is processed through the Whisper model to generate transcriptions.
- Join the transcriptions of all chunks to form a complete transcription for each speaker.

***Generating FAQs:*** 
- Feed the final transcription into LangChain with a prompt to generate a list of Frequently Asked Questions (FAQs) related to the content of the transcription.
- This helps the user to quickly understand what questions to ask to get the most information.

***Chatbot Interaction:***
- The user can interact with the chatbot and ask questions related to the content of the video.
- The chatbot uses the generated FAQs and the underlying LangChain model to provide relevant answers based on the transcription content.


### Installation 

1. Clone the repository :
   `git clone <repo name>`
   `cd <repo>`

2. Create Vistual environment
   `conda create -p venv python==3.10 -y`
   `conda activate venv`

3. Install required packages :
   `pip install -r requirements.txt`

4. Run the Application :
   `streamlit run main.py`

### Project Structure : 
```
PODCAST-CHATBOT
├── src
│   ├── __pycache__
│   ├── __init__.py
│   ├── audio.py
│   ├── chains.py
│   ├── yt_downloader.py
├── .env
├── .gitignore
├── LICENSE
├── main.log
├── main.py
├── pipeline.py
├── README.md
├── requirements.txt
├── test.ipynb
├── test.py

```

### Environment variables : 

- Create .env file and add your env variables in it.

```

OPENAI_API_KEY = "sk-proj........"
HUGGINGFACE_TOKEN = "hf_8........"
```


