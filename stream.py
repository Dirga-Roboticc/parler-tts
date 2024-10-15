import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread
import sounddevice as sd
import queue

torch_device = "cuda:0"  # Use "mps" for Mac 
torch_dtype = torch.bfloat16
model_name = "parler-tts/parler-tts-mini-v1"

# Set max length for padding
max_length = 50

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = ParlerTTSForConditionalGeneration.from_pretrained(
    model_name,
).to(torch_device, dtype=torch_dtype)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

# Queue to buffer audio chunks
audio_queue = queue.Queue()

def audio_player():
    """Thread to play audio from the queue"""
    while True:
        audio_chunk = audio_queue.get()
        if audio_chunk is None:
            break
        sd.play(audio_chunk, samplerate=sampling_rate)
        sd.wait()  # Wait until the sound has finished playing

def generate(text, description, play_steps_in_s=0.5):
    play_steps = int(frame_rate * play_steps_in_s)
    streamer = ParlerTTSStreamer(model, device=torch_device, play_steps=play_steps)
    
    # Tokenization
    inputs = tokenizer(description, return_tensors="pt").to(torch_device)
    prompt = tokenizer(text, return_tensors="pt").to(torch_device)
    
    # Generation kwargs
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )
    
    # Initialize Thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Iterate over chunks of audio and put them in the queue
    for new_audio in streamer:
        if new_audio.shape[0] == 0:
            break
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 4)} seconds")
        audio_queue.put(new_audio)  # Add audio chunk to queue

# Now you can do
text = "This is a test of the streamer class"
description = "Jon's talking really fast."
chunk_size_in_s = 0.5

# Start the audio player thread
player_thread = Thread(target=audio_player, daemon=True)
player_thread.start()

# Generate audio and fill the queue
generate(text, description, chunk_size_in_s)

# Wait until all audio chunks are played
audio_queue.put(None)  # Signal end of audio to the player thread
player_thread.join()
