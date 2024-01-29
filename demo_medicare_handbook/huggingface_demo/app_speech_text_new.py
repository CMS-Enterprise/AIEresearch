import gradio as gr
import tempfile
import openai
#from openai import OpenAI
from openai import AsyncOpenAI
from transformers import pipeline, BarkModel, AutoProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
#import subprocess
import numpy as np
#import pyttsx3
import configparser
import os
import time
#from gtts import gTTS
import scipy.io.wavfile
#import nltk
import pyttsx3

# for text to speech
import torch
import soundfile as sf
from datasets import load_dataset
from scipy.io import wavfile

config = configparser.ConfigParser()
config.read("/mnt/efs/data/AIEresearch/config.ini")
# Set the OpenAI authorization token 
openai_key = config['openai']['api_key']
os.environ['OPENAI_API_KEY'] = openai_key

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    return transcriber({"sampling_rate": sr, "raw": y})["text"]


# Below works, but the voice is espeak

def text_to_speech(text):
    engine = pyttsx3.init()
    #voices = engine.getProperty('voices')
    #engine.setProperty('rate', 160)
    #engine.setProperty("voice", voices[11].id)
    #engine.setProperty('voice', 'com.apple.speech.synthesis.voice.samantha')
    temp_file_path = "/mnt/efs/data/AIEresearch/demo_medicare_handbook/huggingface_demo/temp_audio.wav"
    engine.save_to_file(text, temp_file_path)
    engine.runAndWait()
    
    # Wait for a short period to ensure file is written
    time.sleep(1)
    
    # Check if file exists and is not empty
    if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
        try:
            sample_rate, audio_data = wavfile.read(temp_file_path)
            os.remove(temp_file_path)
            return sample_rate, audio_data
        except Exception as e:
            return None, None  # or handle the exception as needed
    else:
        return None, None


# LLM response with chat history
chat_history = []
client = AsyncOpenAI()
async def llm_response(text):
    global chat_history
    # Update chat history with the user's message
    chat_history.append({"role": "user", "content": text})

    try:
        response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=chat_history)
        # Update chat history with the model's response
        model_response_content = response.choices[0].message.content
        chat_history.append({"role": "system", "content": model_response_content})
        return model_response_content
    except Exception as e:
        return str(e)

def reset_chat():
    global chat_history
    chat_history = []
    return "", "", None, ""

def format_chat_history():
    return "\n".join(f"{message['role'].title()}: {message['content']}" for message in chat_history)

# Gradio app with out of the users mic, chat history, and context retention
with gr.Blocks() as app:
    with gr.Row():
        with gr.Column():
            speech_input = gr.Audio(label="Your Speech", sources=["microphone"], type="numpy")
            submit_button = gr.Button("Submit")
            reset_button = gr.Button("Reset Chat")
            transcribed_text_output = gr.Textbox(label="Transcribed Text")
        with gr.Column():
            text_output = gr.Textbox(label="LLM Response")
            speech_output = gr.Audio(label="Response in Speech", type="filepath")
            #speech_output = gr.Audio(label="Response in Speech")

    with gr.Row():
        chat_history_output = gr.Textbox(label="Chat History", lines=10, interactive=False)

    async def process_audio(audio):
        transcribed_text = transcribe(audio)
        llm_resp = await llm_response(transcribed_text)
        tts_sample_rate, tts_audio_data = text_to_speech(llm_resp)
        #tts_output = text_to_speech(llm_resp)
        #print(tts_output)
        chat_hist = format_chat_history()
        #return transcribed_text, llm_resp, tts_output, chat_hist
        return transcribed_text, llm_resp, (tts_sample_rate, tts_audio_data), chat_hist

    submit_button.click(
        fn=process_audio,
        inputs=speech_input,
        outputs=[transcribed_text_output, text_output, speech_output, chat_history_output]
    )

    reset_button.click(
        fn=reset_chat,
        inputs=[],
        outputs=[transcribed_text_output, text_output, speech_output, chat_history_output]
    )

app.launch()