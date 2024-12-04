import streamlit as st
import numpy as np
import pyttsx3
import pyaudio
import wave
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import io

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Set up Streamlit interface
st.title("Real-Time Voice-to-Voice Processing")
st.write("Click the record button and speak. The system will transcribe your speech, generate a response, and play it back.")

# Record audio function using PyAudio
def record_audio():
    st.write("Recording...")
    samplerate = 16000  # Sample rate for audio recording
    duration = 5  # seconds to record
    
    # Set up PyAudio
    p = pyaudio.PyAudio()
    
    # Open stream for recording
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=1024)
    
    frames = []
    
    # Record for 'duration' seconds
    for _ in range(0, int(samplerate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded audio to a wave file
    with wave.open('recorded_audio.wav', 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(samplerate)
        wf.writeframes(b''.join(frames))

    st.write("Recording finished.")
    return 'recorded_audio.wav'

# Process the recorded audio and generate a response using GPT-2
def generate_response(audio_file):
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
    audio_file = record_audio()
    response_text = generate_response(audio_file)
    
    st.write(f"Response: {response_text}")
    st.audio('response.mp3', format='audio/mp3')
