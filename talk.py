from openai import OpenAI
import time
import pyaudio
import numpy as np
import os
import re
import mlx_whisper
from pynput import keyboard
import subprocess

with open("prompt.txt", 'r', encoding='utf-8') as file:
    SYSTEM_MESSAGE = file.read().strip()

llm_client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

def merge_short_sentences(sens):
    sens_out = []
    for s in sens:
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

def split_sentences(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[""]', '"', text)
    text = re.sub('['']', "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    text = re.sub('[\n\t ]+', ' ', text)
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent.split(" "))
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences(new_sentences)

def play_audio(text):
    texts = split_sentences(text)
    for t in texts:
        t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
        subprocess.call(['say', '-v', 'Karen', t])

def record_audio():
    recording = False
    def on_press(key):
        nonlocal recording
        if key == keyboard.Key.shift:
            recording = True

    def on_release(key):
        nonlocal recording
        if key == keyboard.Key.shift:
            recording = False
            return False

    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    print('Press shift to record...')
    while not recording:
        time.sleep(0.1)
    print('Start recording...')

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=1024, input=True)
    frames = []
    while recording:
        data = stream.read(1024, exception_on_overflow = False)
        frames.append(np.frombuffer(data, dtype=np.int16))
    print('Finished recording')

    audio_data = np.hstack(frames)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return audio_data

def transcribe_audio(audio_data):
    # Save audio data to a temporary file
    temp_file = "temp_audio.wav"
    with open(temp_file, "wb") as f:
        f.write(audio_data.tobytes())
    
    # Transcribe using mlx_whisper
    result = mlx_whisper.transcribe(temp_file)
    
    # Remove temporary file
    os.remove(temp_file)
    
    return result["text"]

def conversation():
    conversation_history = [{'role': 'system', 'content': SYSTEM_MESSAGE}]
    while True:
        audio_data = record_audio()
        user_input = transcribe_audio(audio_data)
        conversation_history.append({'role': 'user', 'content': user_input})

        response = llm_client.chat.completions.create(model="mistral", messages=conversation_history)
        chatbot_response = response.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': chatbot_response})
        print(conversation_history)
        play_audio(chatbot_response)

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

if __name__=='__main__':
    conversation()
