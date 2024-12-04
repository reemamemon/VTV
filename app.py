import streamlit as st
import sounddevice as sd
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pyttsx3
import io

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Set up Streamlit interface
st.title("Real-Time Voice-to-Voice Processing")
st.write("Click the record button and speak. The system will transcribe your speech, generate a response, and play it back.")

# Record audio function
def record_audio():
    st.write("Recording...")
    samplerate = 16000  # Sample rate for audio recording
    duration = 5  # seconds to record
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    st.write("Recording finished.")
    return audio_data

# Process the recorded audio and generate a response using GPT-2
def generate_response(audio_data):
    # Convert audio to text (you can integrate a speech-to-text model here, e.g., Whisper or other models)
    # For simplicity, using a placeholder text since no STT is implemented
    transcribed_text = "Hello, how are you?"

    # Tokenize and generate a response with GPT-2
    inputs = tokenizer.encode(transcribed_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Generate speech response
    engine.save_to_file(response_text, 'response.mp3')
    engine.runAndWait()

    return response_text

# Main logic for Streamlit app
if st.button("Record"):
    audio_data = record_audio()
    response_text = generate_response(audio_data)
    
    st.write(f"Response: {response_text}")
    st.audio('response.mp3', format='audio/mp3')
