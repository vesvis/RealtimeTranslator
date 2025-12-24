"""
Simple Deepgram + OpenAI Translation Test Script
Tests the Deepgram connection by listening to the default microphone,
transcribing in Turkish, and translating to English using OpenAI.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for pyaudio
try:
    import pyaudio
except ImportError:
    print("PyAudio not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "pyaudio"])
    import pyaudio

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from openai import AsyncOpenAI

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096
FORMAT = pyaudio.paInt16

# Translation settings
SOURCE_LANG = "Turkish"
TARGET_LANG = "English"

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def translate_text(text: str) -> str:
    """Translate text using OpenAI"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a translator. Translate the following {SOURCE_LANG} text to {TARGET_LANG}. Only output the translation, nothing else."
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Translation error: {e}]"

async def main():
    # Get API key
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key or api_key == "dummy_key":
        print("ERROR: DEEPGRAM_API_KEY not set in .env file")
        return
    
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "dummy_key":
        print("ERROR: OPENAI_API_KEY not set in .env file")
        return
    
    print(f"Deepgram API key: {api_key[:8]}...")
    print(f"OpenAI API key: {openai_key[:8]}...")
    
    # Initialize Deepgram
    deepgram = DeepgramClient(api_key)
    
    # Create connection
    dg_connection = deepgram.listen.live.v("1")
    
    # Get event loop for async translation
    loop = asyncio.get_running_loop()
    
    # Sentence buffer - accumulate until we have a complete sentence
    sentence_buffer = []
    
    # Event handlers
    def on_message(self, result, **kwargs):
        nonlocal sentence_buffer
        transcript = result.channel.alternatives[0].transcript
        is_final = result.is_final
        if transcript:
            if is_final:
                # Add to sentence buffer
                sentence_buffer.append(transcript)
                full_text = " ".join(sentence_buffer)
                
                # Check if we have a complete sentence (ends with . ! ?)
                if full_text.strip() and full_text.strip()[-1] in '.!?':
                    # Translate complete sentence
                    text_to_translate = full_text.strip()
                    sentence_buffer = []  # Reset buffer
                    
                    async def do_translate():
                        translation = await translate_text(text_to_translate)
                        print(f"\n[TR] {text_to_translate}")
                        print(f"[EN] {translation}")
                    
                    asyncio.run_coroutine_threadsafe(do_translate(), loop)
                else:
                    # Show accumulated text so far (not yet a complete sentence)
                    print(f"  (buffering: {full_text})", end="\r")
            else:
                # Show interim results (original only)
                current = " ".join(sentence_buffer) + " " + transcript if sentence_buffer else transcript
                print(f"  ... {current}", end="\r")
    
    def on_error(self, error, **kwargs):
        print(f"ERROR: {error}")
    
    def on_close(self, close, **kwargs):
        print(f"Connection closed: {close}")
    
    def on_open(self, open, **kwargs):
        print("Deepgram connection opened!")
    
    # Register handlers
    dg_connection.on(LiveTranscriptionEvents.Transcript, on_message)
    dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    
    # Configure options
    options = LiveOptions(
        model="nova-2",
        language="tr",
        smart_format=True,
        encoding="linear16",
        channels=CHANNELS,
        sample_rate=SAMPLE_RATE,
        interim_results=True,
        punctuate=True,
    )
    
    print("Starting Deepgram connection...")
    result = dg_connection.start(options)
    print(f"Connection start result: {result}")
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Open microphone stream
    print(f"Opening default microphone (sample rate: {SAMPLE_RATE}Hz)...")
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("\n" + "="*50)
    print(f"LISTENING... Speak {SOURCE_LANG} into your microphone!")
    print(f"Translations will appear in {TARGET_LANG}")
    print("Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    try:
        while True:
            # Read audio data
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            # Send to Deepgram
            dg_connection.send(data)
            # Small delay to prevent CPU hogging
            await asyncio.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\nStopping...")
    
    finally:
        # Cleanup
        stream.stop_stream()
        stream.close()
        p.terminate()
        try:
            dg_connection.finish()
        except Exception as e:
            print(f"Warning closing connection: {e}")
        print("Done!")

if __name__ == "__main__":
    asyncio.run(main())

