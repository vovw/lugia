# Lugia

## Overview
Lugia is a high-performance local model inference system designed for macOS, built on top of MLX (Machine Learning X). It aims to provide a seamless experience for deploying and running machine learning models locally with minimal resource consumption.

## Features
- **Fast Inference**: Optimized for local inference with MLX, ensuring low latency and high throughput.
- **Customizable Sampling**: Advanced sampling techniques including adaptive sampling to generate diverse outputs from language models.
- **Modular Tooling**: Built in a modular way to allow easy extension and integration of additional functionalities.
- **User Interface**: Text-based user interface (TUI) for easy interaction.
- **Text-to-Speech (TTS)**: Integration with TTS systems for voice-based interactions.
- **Speech-to-Text (STT)**: Integration with STT systems for voice recognition.
- **Sentiment Analysis (SST)**: Basic sentiment analysis capabilities.
- **Conversational assistant**: Mimics a conversational assistant using language models.

## Goals
- **Jarvis**: Emulate a personal assistant akin to Siri or Alexa but running entirely locally.

## Anti-Goals
- **Avoid Bloat**: Minimize dependencies and keep the implementation lightweight.
- **Efficiency**: Avoid excessive computational overhead.
- **API-based**: Focus on command-line and TUI interactions rather than serving APIs.

## Installation
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/lugia.git
   cd lugia
   ```

## Usage
### Running the Talk Interface
```bash
python talk.py
```
This script records audio input, transcribes it, sends the text to a local language model, and then plays the response as audio.

### Running Custom Tools
You can use `tool.py` to perform various tasks such as transcribing audio, sampling from models, etc.

```bash
python tool.py --task transcribe --audio_path path_to_audio.wav
```
