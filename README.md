# PODCAST CHATBOT
A Podcast Chatbot built using LangChain, OpenAI  LLM, OpenAI  Whisper , pynnote audio model , Huggingface  and Streamlit. This project converts YouTube videos to MP3, diarizes the audio, identifies speakers in the podcast, and provides sample questions to extract information from the podcast. You can interact with the podcast content through a chat interface.


### Features
1. **YouTube to MP3 Conversion:** Converts YouTube videos to MP3 format.
2. **Diarization:** Separates and identifies speakers in the podcast.
3. **Speaker Identification:** Recognizes different speakers in the podcast.
4. **Sample Questions Generation:** Provides sample questions to query the podcast content.
5. **Chat Interface:** Allows users to interact with the podcast content through a chat interface.


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


