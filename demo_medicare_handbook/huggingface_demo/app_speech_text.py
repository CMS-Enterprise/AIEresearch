import gradio as gr
import tempfile
#import openai
#from openai import OpenAI
from openai import AsyncOpenAI
from transformers import pipeline, BarkModel, AutoProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
#import subprocess
import numpy as np
#import pyttsx3
import configparser
import os
#import time
#from gtts import gTTS
import scipy.io.wavfile
#import nltk
import pyttsx3

# for text to speech
import torch
import soundfile as sf
from datasets import load_dataset

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

# Function to process text input (from LLM) and convert to speech
#processor = AutoProcessor.from_pretrained("suno/bark-small")
#model = BarkModel.from_pretrained("suno/bark-small", local_files_only=True)
#model = model.to_bettertransformer()
#nltk.download('punkt')

# Function to process text input (from LLM) and convert to speech - Does not use API
#def text_to_speech(text):
    #engine = pyttsx3.init()
    # Use a .wav suffix as it's more universally supported by pyttsx3
    #with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as fp:
        #engine.save_to_file(text, fp.name)
        #temp_filename = fp.name
    #engine.runAndWait()
    #return temp_filename

def divide_text(text, max_length):
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
        # Find the nearest space to the max_length
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:  # No space found, forced to split at max_length
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()  # Remove leading spaces from the next chunk
    return chunks

#def text_to_speech(text):
    #processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    #model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    #vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    #inputs = processor(text=text, return_tensors="pt", normalize=True)

    # load xvector containing speaker's voice characteristics from a dataset
    #embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    #speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    #speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    #sf.write("speech.wav", speech.numpy(), samplerate=16000)
    #return "speech.wav"
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

def text_to_speech(text):
    text_chunks = divide_text(text, 600)
    audio_data = []
    for chunk in text_chunks:
        inputs = processor(text=chunk, return_tensors="pt", normalize=True)
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        audio_data.append(speech)
    combined_audio = np.concatenate(audio_data)

    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode='w+b') as temp_audio_file:
        #scipy.io.wavfile.write(temp_audio_file, rate=16000, data=combined_audio.numpy())
        scipy.io.wavfile.write(temp_audio_file, rate=16000, data=combined_audio)
        return temp_audio_file.name

#below works
#def text_to_speech(text):
    #inputs = processor(text=text, voice_preset="v2/en_speaker_6")
    #audio_array = model.generate(**inputs, pad_token_id=processor.tokenizer.pad_token_id)
    #audio_array = audio_array.cpu().numpy().squeeze()
    #sample_rate = model.generation_config.sample_rate

    # Create a temporary file to store the audio
    #with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode='w+b') as temp_audio_file:
        #scipy.io.wavfile.write(temp_audio_file, rate=sample_rate, data=audio_array)
        #return temp_audio_file.name


# LLM response with chat history
chat_history = [{"role": "system", "content": "You are a helpful assistant. Please keep your answers succinct."}]
client = AsyncOpenAI()
async def llm_response(text):
    global chat_history
    # Update chat history with the user's message
    chat_history.append({"role": "user", "content": text})

    try:
        response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=chat_history)
        #response = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo",
            #messages=chat_history
        #)
        # Update chat history with the model's response
        model_response_content = response.choices[0].message.content
        chat_history.append({"role": "system", "content": model_response_content})
        return model_response_content
        #chat_history.append({"role": "system", "content": response.choices[0].message['content']})
        #return response.choices[0].message['content']
    except Exception as e:
        return str(e)

def reset_chat():
    global chat_history
    chat_history = [{"role": "system", "content": "You are a helpful assistant. Please keep your answers succinct."}]
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

    with gr.Row():
        chat_history_output = gr.Textbox(label="Chat History", lines=10, interactive=False)

    async def process_audio(audio):
        transcribed_text = transcribe(audio)
        llm_resp = await llm_response(transcribed_text)
        tts_output = text_to_speech(llm_resp)
        chat_hist = format_chat_history()
        return transcribed_text, llm_resp, tts_output, chat_hist

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