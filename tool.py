import argparse
import replicate
from pydantic import create_model
import inspect
import json
import re
from inspect import Parameter
from fastcore.foundation import L
from functools import partial
from pprint import pprint
import numpy as np
import os
import pyaudio
import mlx_whisper

# Helper functions for the Replicate inference API and parsing the response

def parse(text):
    """Use regular expressions to find content within the tags."""
    function_call_search = re.search(r"<function-call>\s*(.*?)\s*</function-call>", text, re.DOTALL)
    answer_search = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    function_call = function_call_search.group(1).strip() if function_call_search else None
    answer = answer_search.group(1).strip() if answer_search else None

    if function_call and function_call != "None":
        return {"type": "function-call", "content": function_call}
    elif answer:
        return {"type": "text", "content": answer}
    else:
        return {"type": "text", "content": text}

def run(prompt: str, tools: str = None):
    inp = {"prompt": f"{prompt}", "temperature": 0}
    if tools:
        inp['tools'] = tools
    output = replicate.run(
        "hamelsmu/llama-3-70b-instruct-awq-with-tools:b6042c085a7927a3d87e065a9f51fb7238ef6870c7a2ab7b03caa3c0e9413e19",
        input=inp
    )
    txt = ''.join(output)
    return parse(txt)

# Helper to turn functions into a schema from fastai/llm-hackers

def schema(f):
    kw = {n: (o.annotation, ... if o.default == Parameter.empty else o.default)
          for n, o in inspect.signature(f).parameters.items()}
    s = create_model(f'Input for `{f.__name__}`', **kw).schema()
    return dict(name=f.__name__, description=f.__doc__, parameters=s)

# Define tools

def get_exchange_rate(base_currency: str, target_currency: str):
    """
    Get the exchange rate between two currencies.

    Parameters:
    - base_currency (str): The currency to convert from.
    - target_currency (str): The currency to convert to.

    Returns:
    float: The exchange rate from base_currency to target_currency.
    """
    # Placeholder implementation
    exchange_rates = {
        ("USD", "JPY"): 147.5,  # Example rate
    }
    return exchange_rates.get((base_currency, target_currency), None)

def create_contact(name: str, email: str):
    """
    Create a new contact.

    Parameters:
    - name (str): The name of the contact.
    - email (str): The email address of the contact.

    Returns:
    dict: Confirmation of the created contact.
    """
    # Placeholder implementation
    return {"name": name, "email": email}

tools = json.dumps(list(L([get_exchange_rate, create_contact]).map(schema)))

# Mapping of function names to actual functions
FUNCTION_MAP = {
    "get_exchange_rate": get_exchange_rate,
    "create_contact": create_contact
}

# Helper functions for audio handling

def record_audio(output_path="temp_audio.wav", duration=5):
    """
    Records audio from the microphone and saves it to the specified output path.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, frames_per_buffer=1024, input=True)
    frames = []
    print("Recording...")
    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    with open(output_path, "wb") as f:
        f.write(b'\x00' * 44)  # Placeholder for header (44 bytes for WAV header)
        f.write(b''.join(frames))
        # Reopen to add WAV header
        with open(output_path, "r+b") as f:
            riff_header = b'RIFF' + (len(f.read(44)) + 36).to_bytes(4, 'little') + b'WAVE'
            f.seek(0)
            f.write(riff_header)

def transcribe_audio(audio_path):
    """
    Transcribes audio from the specified path using mlx_whisper.
    """
    result = mlx_whisper.transcribe(audio_path)
    os.remove(audio_path)  # Remove temporary file
    return result["text"]

def execute_function_call(function_call: str):
    """
    Execute the function call based on the parsed response.
    """
    try:
        # Extract function name and arguments
        function_call = function_call.strip()
        func_name, args_str = function_call.rsplit('(', maxsplit=1)
        args_str = args_str.rstrip(')')
        args = json.loads(f'{{{", ".join(arg.split("=") for arg in args_str.split(", "))}}}')

        # Find the function and execute it
        func = FUNCTION_MAP.get(func_name)
        if func:
            result = func(**args)
            return result
        else:
            return f"Function {func_name} not found."
    except Exception as e:
        return f"Error executing function call: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Lugia Tool for various tasks.")
    parser.add_argument("--task", required=True, choices=["transcribe", "sample", "converse"], help="Task to perform")
    parser.add_argument("--prompt", help="Prompt for the model")
    parser.add_argument("--audio_path", help="Path to the audio file (required for transcribe)")
    parser.add_argument("--logits", type=float, nargs='+', help="Logits from the model (required for sample)")
    parser.add_argument("--output_path", default="temp_audio.wav", help="Output audio file path (default: temp_audio.wav)")
    parser.add_argument("--duration", type=int, default=5, help="Duration of recording in seconds (default: 5)")
    args = parser.parse_args()

    if args.task == "transcribe":
        if not args.audio_path:
            parser.error("--audio_path is required for the transcribe task.")
        print("Transcribing...")
        text = transcribe_audio(args.audio_path)
        print(f"Transcribed text: {text}")
    elif args.task == "sample":
        if not args.logits:
            parser.error("--logits is required for the sample task.")
        print("Sampling from model...")
        sampled_output = sample_from_model(np.array(args.logits))
        print(f"Sampled output: {sampled_output}")
    elif args.task == "converse":
        if not args.prompt:
            parser.error("--prompt is required for the converse task.")
        print("Conversing...")
        response = run(prompt=args.prompt, tools=tools)
        pprint(response)
        if response["type"] == "function-call":
            function_response = execute_function_call(response["content"])
            pprint(function_response)

if __name__ == "__main__":
    main()
