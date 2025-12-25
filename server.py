"""
Integrated Realtime Translator Server

Single server that:
1. Captures microphone audio via PyAudio
2. Transcribes via Deepgram
3. Translates via OpenAI
4. Generates TTS via ElevenLabs
5. Broadcasts to web listeners via WebSocket
"""

import os
import sys
import json
import asyncio
import time
import base64
import uuid
import threading
from datetime import datetime
from typing import Dict, Optional, Set
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from dotenv import load_dotenv
import numpy as np

# Check for pyaudio
try:
    import pyaudio
except ImportError:
    print("PyAudio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
    import pyaudio

from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
from openai import AsyncOpenAI
from elevenlabs.client import ElevenLabs

# Load environment variables
load_dotenv(override=True)

# ============== Configuration ==============

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 4096
FORMAT = pyaudio.paInt16

# Available languages (code, name, deepgram_code)
LANGUAGES = [
    ("en", "English", "en"),
    ("es", "Spanish", "es"),
    ("fr", "French", "fr"),
    ("de", "German", "de"),
    ("it", "Italian", "it"),
    ("pt", "Portuguese", "pt"),
    ("ru", "Russian", "ru"),
    ("zh", "Chinese", "zh"),
    ("ja", "Japanese", "ja"),
    ("ko", "Korean", "ko"),
    ("ar", "Arabic", "ar"),
    ("tr", "Turkish", "tr"),
    ("nl", "Dutch", "nl"),
    ("pl", "Polish", "pl"),
    ("hi", "Hindi", "hi"),
]

# Available ElevenLabs voices (id, name, description)
VOICES = [
    ("29vD33N1CtxCmqQRPOHJ", "Drew", "Male - Main Speaker"),
    ("bIHbv24MWmeRgasZH58o", "Will", "Male - Relaxed, Casual"),
    ("iP95p4xoKVk53GoZ742B", "Chris", "Male - Friendly, Down-to-Earth"),
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger", "Male - Laid-Back"),
    ("UgBBYS2sOqTuMpoF3BR0", "Mark", "Male - Natural"),
    ("nPczCjzI2devNBz1zQrb", "Brian", "Male - Deep, Comforting"),
    ("onwK4e9ZLuTAKqWW03F9", "Daniel", "Male - British, Steady"),
    ("JBFqnCBsd6RMkjVDRZzb", "George", "Male - British, Warm"),
]

# Speaker voice mapping for diarization (speaker_id -> voice_id)
# Names are set dynamically via get_speaker_name()
SPEAKER_VOICES = {
    0: "29vD33N1CtxCmqQRPOHJ",  # Main speaker - Drew voice
    1: "bIHbv24MWmeRgasZH58o",  # Contributor 1 - Will voice
    2: "iP95p4xoKVk53GoZ742B",  # Contributor 2 - Chris voice
}

# Risale-i Nur domain-specific keywords for Deepgram (no boost, just recognition hints)
RISALE_KEYWORDS = [
    # Proper nouns & titles
    "Bediüzzaman", "Said Nursi", "Risale-i Nur", 
    "Barla", "Kastamonu", "Emirdağ", "Isparta",

    # Book titles
    "Sözler", "Mektubat", "Lemalar", "Şualar",
    "Mesnevi-i Nuriye", "İşaratü'l-İ'caz", "Asa-yı Musa",

    # Multi-word theological phrases
    "Mana-yı Harfi", "Mana-yı İsmi",
    "Vacib-ül Vücud", 
    "Cenab-ı Hak",
    "Kadir-i Mutlak", "Alim-i Mutlak",
    "Esma-i Hüsna",
    "Alemi Şehadet", "Alemi Gayb", "Alemi Berzah"
]

# Specialized translation system prompt for Risale-i Nur lectures (Paragraph Mode)
RISALE_TRANSLATION_PROMPT = """You are an expert simultaneous interpreter for the "Risale-i Nur". You are receiving a complete, buffered block of spoken text.

**INPUT DATA:**
- [CONTEXT]: The previous few sentences (read-only, for continuity).
- [TARGET_BATCH]: A complete block of recent speech to translate.

**CORE INSTRUCTIONS:**
1. **Wait for the Verb:** You have the full Turkish sentence. Locate the main verb (usually at the end) and construct a grammatically perfect English sentence (SVO structure).
2. **Smoothing:** The input is spoken text. Remove stuttering, false starts, or self-corrections. Make the English output fluent and clean.
3. **Tone Strategy:**
   - **Default:** Formal, dignified, academic (Theology).
   - **EXCEPTION:** If the text is a direct command or warning (imperative), switch to **Punchy, Idiomatic English** (e.g., use "Come to your senses!" instead of "Take your head among your hands").

**CRITICAL: PHONETIC & TERMINOLOGY REPAIR:**
   - The input contains ASR (Speech-to-Text) errors. You MUST infer the intended word from context.
   - **Common Phonetic Corrections:**
     - "Liman" (port) → Correct to **"Lem'a"** (The Flash) when citing books.
     - "live cil/cillah" → Correct to **"li-vechillâh"** (for the sake of God).
     - "laan turcusu" → Correct to idiom **"lahana turşusu"** (contradiction).
     - "Bicrim" → Correct to **"Cirim"** (particle/body).
   - **Standard Terms:**
     - Üstad → Üstad
     - Cenab-ı Hak → Almighty God
     - Ene → The Ego
     - Tevhid → Divine Unity
     - Haşir → Resurrection
     - Esma-i Hüsna → The Divine Names
     - Abi / Kardeş → Brother

**OUTPUT FORMAT FOR NATURAL SPEECH:**
   - Output each distinct sentence on a NEW LINE for natural pauses.
   - When there is a MAJOR shift (e.g., speaker says "let's continue reading" or transitions from explanation to quotation), insert "||" on its own line to indicate a longer pause.
   - Example:
     ```
     This is the first point.
     And this explains it further.
     ||
     Now let us read from the text.
     "The truth manifests itself clearly."
     ```

**Output:** Provide ONLY the English translation with line breaks as described."""

# Runtime config (set by interactive menu)
config = {
    "source_lang": "tr",
    "source_lang_name": "Turkish",
    "target_lang": "en",
    "target_lang_name": "English",
    "voice_id": "29vD33N1CtxCmqQRPOHJ",
    "voice_name": "Drew",
    "mic_device_index": None,  # None = default device
    "mic_device_name": "Default",
    "use_risale_context": True,  # Enable Risale-i Nur specific context
    "session_id": "LECTURE",  # Custom session ID for persistent listener links
    "main_speaker_name": "Speaker",  # Name displayed for main speaker
}

def get_speaker_name(speaker_id: int) -> str:
    """Get display name for a speaker based on ID."""
    if speaker_id == 0:
        return config.get("main_speaker_name", "Speaker")
    else:
        return f"Contributor {speaker_id}"

def get_speaker_voice(speaker_id: int) -> str:
    """Get voice ID for a speaker."""
    return SPEAKER_VOICES.get(speaker_id, SPEAKER_VOICES[0])

# ============== API Clients ==============

deepgram_key = os.getenv("DEEPGRAM_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")

# Validate API keys
def check_api_keys():
    missing = []
    if not deepgram_key or deepgram_key == "dummy_key":
        missing.append("DEEPGRAM_API_KEY")
    if not openai_key or openai_key == "dummy_key":
        missing.append("OPENAI_API_KEY")
    if not elevenlabs_key or elevenlabs_key == "dummy_key":
        missing.append("ELEVENLABS_API_KEY")
    
    if missing:
        print(f"ERROR: Missing API keys: {', '.join(missing)}")
        print("Please set them in your .env file")
        sys.exit(1)

deepgram_client = DeepgramClient(deepgram_key or "dummy_key")
openai_client = AsyncOpenAI(api_key=openai_key or "dummy_key")
elevenlabs_client = ElevenLabs(api_key=elevenlabs_key or "dummy_key")

# ============== FastAPI App ==============

app = FastAPI(title="Realtime Translator")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============== Session Management ==============

@dataclass
class Session:
    """Represents a translation session"""
    id: str
    source_lang: str = "tr"
    target_lang: str = "en"
    source_lang_name: str = "Turkish"
    target_lang_name: str = "English"
    voice_id: str = "29vD33N1CtxCmqQRPOHJ"
    is_live: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    listener_ws_set: Set[WebSocket] = field(default_factory=set)
    
    def to_dict(self):
        return {
            "id": self.id,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "source_lang_name": self.source_lang_name,
            "target_lang_name": self.target_lang_name,
            "is_live": self.is_live,
            "listener_count": len(self.listener_ws_set),
            "created_at": self.created_at.isoformat()
        }

# Global session (created on startup)
active_session: Optional[Session] = None

# ============== Producer State ==============

class ProducerState:
    """Manages the audio capture and transcription state"""
    def __init__(self):
        self.is_running = False
        self.dg_connection = None
        self.audio_thread: Optional[threading.Thread] = None
        self.pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.current_speaker = 0  # Track current speaker for diarization
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Reconnection state
        self.is_reconnecting = False
        self.connection_healthy = False
        # Buffer timeout tracking
        self.buffer_timeout_task: Optional[asyncio.Task] = None
        # Translation buffer for time-based sentence accumulation
        self.translation_buffer: Optional['TranslationBuffer'] = None

producer = ProducerState()


class TranslationBuffer:
    """Smart buffer for accumulated sentence-boundary translation.
    
    Accumulates text until detecting a complete thought based on:
    - Semantic: Ends with strong punctuation AND >= min_words
    - Timeout: Buffer held > timeout_seconds (prevents stalling)
    """
    
    def __init__(self, timeout_seconds: float = 15.0, min_words: int = 15):
        self.buffer = ""
        self.last_flush_time = time.time()
        self.history = []  # Last 3 sentences for context (original Turkish)
        self.timeout_seconds = timeout_seconds
        self.min_words = min_words
        self.current_speaker = 0  # Track speaker for the buffered content
    
    def add_segment(self, text: str, speaker_id: int = 0):
        """Append incoming ASR text to buffer."""
        self.buffer = (self.buffer + " " + text).strip()
        self.current_speaker = speaker_id
    
    def should_flush(self) -> bool:
        """Check if buffer should be flushed for translation."""
        if not self.buffer.strip():
            return False
        
        # Condition A: Semantic - strong punctuation AND long enough
        has_punctuation = self.buffer.strip().endswith(('.', '?', '!'))
        word_count = len(self.buffer.split())
        is_long_enough = word_count >= self.min_words
        
        # Condition B: Timeout - prevents stalling
        time_elapsed = time.time() - self.last_flush_time
        is_timeout = time_elapsed > self.timeout_seconds
        
        return (has_punctuation and is_long_enough) or is_timeout
    
    def get_buffer_content(self) -> str:
        """Get current buffer content without flushing."""
        return self.buffer.strip()
    
    def get_word_count(self) -> int:
        """Get current word count in buffer."""
        return len(self.buffer.split()) if self.buffer else 0
    
    def get_time_elapsed(self) -> float:
        """Get seconds since last flush."""
        return time.time() - self.last_flush_time
    
    def flush(self) -> tuple:
        """Get buffer content and history for translation.
        
        Returns:
            tuple: (content, speaker_id, history)
        """
        content = self.buffer.strip()
        history = self.history.copy()
        speaker_id = self.current_speaker
        
        # Move original text to history for context
        if content:
            self.history.append(content)
            if len(self.history) > 3:  # Keep last 3 sentences
                self.history.pop(0)
        
        self.buffer = ""
        self.last_flush_time = time.time()
        return content, speaker_id, history
    
    def has_pending(self) -> bool:
        """Check if there is any pending text in buffer."""
        return bool(self.buffer.strip())
    
    def pending_count(self) -> int:
        """Get approximate 'count' - here just 1 if has content, 0 otherwise."""
        return 1 if self.buffer.strip() else 0

# ============== Recording Manager ==============

class RecordingManager:
    """Manages local recording of translations"""
    def __init__(self):
        self.is_recording = False
        self.session_dir: Optional[str] = None
        self.segment_count = 0
        self.transcript_file = None
        self.audio_segments: list = []  # List of (path, speaker_name) for merging
    
    def start_recording(self, session_id: str):
        """Start a new recording session"""
        # Create recordings directory if it doesn't exist
        recordings_base = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(recordings_base, exist_ok=True)
        
        # Create session-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(recordings_base, f"session_{session_id}_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)
        
        # Initialize transcript file
        transcript_path = os.path.join(self.session_dir, "transcript.txt")
        self.transcript_file = open(transcript_path, "w", encoding="utf-8")
        self.transcript_file.write(f"# Translation Session: {session_id}\n")
        self.transcript_file.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.transcript_file.write(f"# Source: {config['source_lang_name']} → Target: {config['target_lang_name']}\n")
        self.transcript_file.write("=" * 60 + "\n\n")
        self.transcript_file.flush()
        
        self.segment_count = 0
        self.audio_segments = []
        self.is_recording = True
        
        print(f"📹 Recording started: {self.session_dir}")
    
    def save_segment(self, audio_data: bytes, original_text: str, translation: str, speaker_name: str):
        """Save an audio segment and log to transcript"""
        if not self.is_recording or not self.session_dir:
            return
        
        self.segment_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Save audio segment
        audio_filename = f"segment_{self.segment_count:04d}_{speaker_name.lower()}.mp3"
        audio_path = os.path.join(self.session_dir, audio_filename)
        
        try:
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            self.audio_segments.append((audio_path, speaker_name))
        except Exception as e:
            print(f"Warning: Could not save audio segment: {e}")
        
        # Log to transcript
        if self.transcript_file:
            try:
                self.transcript_file.write(f"[{timestamp}] [{speaker_name}]\n")
                self.transcript_file.write(f"  Original: {original_text}\n")
                self.transcript_file.write(f"  Translation: {translation}\n\n")
                self.transcript_file.flush()
            except Exception as e:
                print(f"Warning: Could not write to transcript: {e}")
    def merge_segments(self) -> Optional[str]:
        """Merge all audio segments into a single continuous file"""
        if not self.audio_segments or not self.session_dir:
            return None
        
        try:
            from pydub import AudioSegment
        except ImportError:
            print("⚠️  pydub not installed. Run: pip install pydub")
            print("   Skipping audio merge. Individual segments are still available.")
            return None
        
        try:
            print("🔄 Merging audio segments...")
            
            # Combine all segments
            combined = AudioSegment.empty()
            for audio_path, speaker_name in self.audio_segments:
                if os.path.exists(audio_path):
                    segment = AudioSegment.from_mp3(audio_path)
                    combined += segment
            
            # Export merged file
            merged_path = os.path.join(self.session_dir, "full_recording.mp3")
            combined.export(merged_path, format="mp3")
            
            duration_sec = len(combined) / 1000
            duration_min = duration_sec / 60
            print(f"✅ Merged recording saved: full_recording.mp3 ({duration_min:.1f} min)")
            
            # Clean up individual segment files after successful merge
            deleted_count = 0
            for audio_path, _ in self.audio_segments:
                try:
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                        deleted_count += 1
                except Exception as del_err:
                    print(f"   Warning: Could not delete segment {audio_path}: {del_err}")
            
            if deleted_count > 0:
                print(f"🗑️  Cleaned up {deleted_count} segment files")
            
            return merged_path
            
        except Exception as e:
            print(f"⚠️  Could not merge audio segments: {e}")
            return None
    
    def stop_recording(self):
        """Stop recording and finalize files"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Merge all segments into one file
        session_dir_backup = self.session_dir  # Save before clearing
        segments_backup = self.audio_segments.copy()
        
        if self.audio_segments:
            self.merge_segments()
        
        # Write summary to transcript
        if self.transcript_file:
            try:
                self.transcript_file.write("\n" + "=" * 60 + "\n")
                self.transcript_file.write(f"# Recording ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.transcript_file.write(f"# Total segments: {self.segment_count}\n")
                self.transcript_file.close()
            except:
                pass
            self.transcript_file = None
        
        if session_dir_backup:
            print(f"📹 Recording stopped: {self.segment_count} segments saved to {session_dir_backup}")
        
        self.session_dir = None
        self.segment_count = 0
        self.audio_segments = []

# Global recording manager
recording_manager = RecordingManager()

# ============== Translation Pipeline ==============

# Context buffer for better translation (stores last N sentences)
# Can be set via CONTEXT_BUFFER_SIZE env var, or changed at runtime via API
CONTEXT_BUFFER_SIZE = int(os.getenv("CONTEXT_BUFFER_SIZE", "5"))
translation_context = []  # List of (original, translation) tuples


def construct_payload(history: list, current: str) -> str:
    """Construct a structured payload with context for translation.
    
    Args:
        history: List of previous original Turkish sentences (up to 3)
        current: The current batch of text to translate
    
    Returns:
        Formatted string with XML-style delimiters
    """
    payload_parts = []
    
    # Context (read-only for model reference)
    if history:
        payload_parts.append("[CONTEXT]")
        for segment in history[-3:]:  # Last 3 sentences
            payload_parts.append(segment)
        payload_parts.append("[/CONTEXT]")
        payload_parts.append("")
    
    # Target batch (what to translate)
    payload_parts.append("[TARGET_BATCH]")
    payload_parts.append(current)
    payload_parts.append("[/TARGET_BATCH]")
    
    return "\n".join(payload_parts)

async def translate_text(text: str, history: list = None) -> str:
    """Translate text using OpenAI with context-aware prompting.
    
    Args:
        text: The text to translate
        history: Optional list of previous original texts for context
    """
    global translation_context
    
    try:
        # Use specialized Risale-i Nur prompt if context is enabled
        if config.get("use_risale_context", False):
            system_prompt = RISALE_TRANSLATION_PROMPT
            # Use structured payload with XML-style context delimiters
            user_content = construct_payload(
                history=history or [],
                current=text
            )
        else:
            system_prompt = f"You are a translator. Translate the following {active_session.source_lang_name} text to {active_session.target_lang_name}. Only output the translation, nothing else."
            user_content = text
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        translation = response.choices[0].message.content.strip()
        
        # Add to context buffer (limit size) for backward compatibility
        translation_context.append((text, translation))
        if len(translation_context) > CONTEXT_BUFFER_SIZE:
            translation_context.pop(0)
        
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return "[Translation error]"

def format_text_for_tts(text: str) -> str:
    """Format translation text for natural TTS with SSML breaks.
    
    Converts:
    - '||' markers → 1.5 second pause (major section break)
    - Line breaks → 0.5 second pause (sentence break)
    """
    import re
    
    # Replace || with long pause
    text = re.sub(r'\s*\|\|\s*', ' <break time="1.5s"/> ', text)
    
    # Replace line breaks with short pause (but not multiple)
    text = re.sub(r'\n+', ' <break time="0.5s"/> ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


async def generate_speech(text: str, speaker_id: int = 0) -> Optional[bytes]:
    """Generate speech using ElevenLabs with speaker-specific voice"""
    try:
        # Get voice based on speaker ID (for diarization)
        voice_id = get_speaker_voice(speaker_id)
        
        # Format text with SSML breaks for natural pauses
        formatted_text = format_text_for_tts(text)
        
        audio = elevenlabs_client.generate(
            text=formatted_text,
            voice=voice_id,
            model="eleven_turbo_v2"
        )
        
        # Collect audio chunks
        audio_bytes = b""
        for chunk in audio:
            audio_bytes += chunk
        
        return audio_bytes
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

async def broadcast_to_listeners(message: dict):
    """Broadcast a message to all connected listeners"""
    if not active_session:
        return
    
    disconnected = set()
    for ws in active_session.listener_ws_set:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    
    # Remove disconnected listeners
    active_session.listener_ws_set -= disconnected

# Minimum words required to process (filters out breathing, "um", etc.)
MIN_WORDS_TO_PROCESS = 3

async def process_complete_sentence(text: str, speaker_id: int = 0,
                                      history: list = None):
    """Process a complete sentence through the translation + TTS pipeline.
    
    Args:
        text: The text to translate
        speaker_id: ID of the speaker for voice selection
        history: Optional list of previous texts for context
    """
    if not active_session or not active_session.is_live:
        return
    
    # Filter out very short fragments (breathing, filler sounds, etc.)
    words = text.strip().split()
    if len(words) < MIN_WORDS_TO_PROCESS:
        print(f"  (skipped short fragment: '{text}')")
        return
    
    # Get speaker name for display
    speaker_name = get_speaker_name(speaker_id)
    
    # Clear the buffering line and show the translation
    print(" " * 120, end="\r")  # Clear buffering line
    print(f"\n{'─'*60}")
    print(f"🎤 [{speaker_name}] [{active_session.source_lang_name}]")
    print(f"   {text}")
    
    # Translate with context
    translation = await translate_text(text, history=history)
    print(f"🔊 [{speaker_name}] [{active_session.target_lang_name}]")
    print(f"   {translation}")
    print(f"{'─'*60}")
    
    # Generate TTS with speaker-specific voice
    audio_data = await generate_speech(translation, speaker_id)
    
    if audio_data:
        # Save to local recording
        recording_manager.save_segment(audio_data, text, translation, speaker_name)
        
        # Broadcast to listeners with speaker info
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        await broadcast_to_listeners({
            "type": "audio",
            "data": audio_base64,
            "text": translation,
            "original": text,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name
        })
        
        # Add a small pause after each segment for natural pacing
        await asyncio.sleep(0.5)

# ============== Deepgram Handlers ==============

def on_transcript(self, result, **kwargs):
    """Handle incoming transcription from Deepgram with buffered sentence-boundary translation."""
    transcript = result.channel.alternatives[0].transcript
    is_final = result.is_final
    words = result.channel.alternatives[0].words
    
    if not transcript:
        return
    
    # Extract speaker ID from diarization (if available)
    speaker_id = producer.current_speaker  # Default to current speaker
    if words and len(words) > 0:
        first_word = words[0]
        if hasattr(first_word, 'speaker') and first_word.speaker is not None:
            speaker_id = min(first_word.speaker, 2)  # Cap at 2 voices
    
    if is_final:
        # Check if speaker changed - if so, flush the buffer first
        if speaker_id != producer.current_speaker and producer.translation_buffer:
            if producer.translation_buffer.has_pending():
                content, old_speaker, history = producer.translation_buffer.flush()
                if content and producer.loop:
                    asyncio.run_coroutine_threadsafe(
                        process_complete_sentence(content, old_speaker, history=history),
                        producer.loop
                    )
        
        # Update current speaker
        producer.current_speaker = speaker_id
        
        # Add transcript to buffer
        if producer.translation_buffer:
            producer.translation_buffer.add_segment(transcript, speaker_id)
            
            # Check if we should flush (semantic + length OR timeout)
            if producer.translation_buffer.should_flush():
                content, flush_speaker, history = producer.translation_buffer.flush()
                if content and producer.loop:
                    asyncio.run_coroutine_threadsafe(
                        process_complete_sentence(content, flush_speaker, history=history),
                        producer.loop
                    )
            else:
                # Show buffering status
                speaker_name = get_speaker_name(speaker_id)
                word_count = producer.translation_buffer.get_word_count()
                time_elapsed = producer.translation_buffer.get_time_elapsed()
                buffer_preview = producer.translation_buffer.get_buffer_content()[:60]
                if len(producer.translation_buffer.get_buffer_content()) > 60:
                    buffer_preview += "..."
                print(f"  [{speaker_name}] ({word_count}w, {time_elapsed:.0f}s/{producer.translation_buffer.timeout_seconds:.0f}s) {buffer_preview}", end="\r")
    else:
        # Show interim results
        buffer_content = producer.translation_buffer.get_buffer_content() if producer.translation_buffer else ""
        current = buffer_content + " " + transcript if buffer_content else transcript
        speaker_name = get_speaker_name(speaker_id)
        print(f"  [{speaker_name}] ... {current[:100]}", end="\r")

def on_error(self, error, **kwargs):
    print(f"\n⚠️  Deepgram ERROR: {error}")
    print("   This may be a network issue. Attempting to reconnect...")
    producer.connection_healthy = False
    # Trigger reconnection
    if producer.loop and producer.is_running and not producer.is_reconnecting:
        asyncio.run_coroutine_threadsafe(reconnect_deepgram(), producer.loop)

def on_close(self, close, **kwargs):
    print(f"\n⚠️  Deepgram connection closed: {close}")
    producer.connection_healthy = False
    if producer.is_running and not producer.is_reconnecting:
        print("   Attempting to reconnect...")
        if producer.loop:
            asyncio.run_coroutine_threadsafe(reconnect_deepgram(), producer.loop)

def on_open(self, open, **kwargs):
    print("✓ Deepgram connection opened!")
    producer.connection_healthy = True
    producer.is_reconnecting = False

# ============== Auto-Reconnect & Buffer Timeout ==============

# Reconnection settings
MAX_RECONNECT_DELAY = 30  # Maximum seconds between reconnection attempts
INITIAL_RECONNECT_DELAY = 2  # Start with 2 second delay

async def reconnect_deepgram(retry_count: int = 0):
    """Reconnect to Deepgram after a disconnection with exponential backoff"""
    if producer.is_reconnecting or not producer.is_running:
        return
    
    producer.is_reconnecting = True
    
    # Calculate delay with exponential backoff (2, 4, 8, 16, 30, 30, ...)
    delay = min(INITIAL_RECONNECT_DELAY * (2 ** retry_count), MAX_RECONNECT_DELAY)
    print(f"\n🔄 Reconnecting to Deepgram in {delay} seconds... (attempt {retry_count + 1})")
    
    await asyncio.sleep(delay)
    
    if not producer.is_running:
        producer.is_reconnecting = False
        return
    
    try:
        # Close existing connection if any
        if producer.dg_connection:
            try:
                producer.dg_connection.finish()
            except:
                pass
        
        # Create new connection
        producer.dg_connection = deepgram_client.listen.live.v("1")
        producer.dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        producer.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
        producer.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
        producer.dg_connection.on(LiveTranscriptionEvents.Open, on_open)
        
        # Configure Deepgram options
        options = LiveOptions(
            model="nova-3",
            language=active_session.source_lang,
            smart_format=True,
            encoding="linear16",
            channels=CHANNELS,
            sample_rate=SAMPLE_RATE,
            interim_results=True,
            punctuate=True,
            diarize=True,
        )
        
        result = producer.dg_connection.start(options)
        if result is False:
            print("WARNING: Deepgram start() returned False, but proceeding...")
        
        print("✓ Reconnection successful!")
        # Reset reconnecting flag - on_open will also set connection_healthy = True
        
    except Exception as e:
        print(f"❌ Reconnection failed: {e}")
        producer.is_reconnecting = False
        
        # Keep retrying with exponential backoff until connection succeeds
        if producer.is_running:
            print(f"   Will retry in {min(INITIAL_RECONNECT_DELAY * (2 ** (retry_count + 1)), MAX_RECONNECT_DELAY)} seconds...")
            asyncio.create_task(reconnect_deepgram(retry_count + 1))

# Buffer timeout is now handled by TranslationBuffer.should_flush(), but we still
# need a background checker to trigger flushes when speaker is idle

async def buffer_timeout_checker():
    """Background task that checks for translation buffer timeout."""
    while producer.is_running:
        await asyncio.sleep(2)  # Check every 2 seconds
        
        # Check if translation buffer should be flushed (timeout condition)
        if producer.translation_buffer and producer.translation_buffer.has_pending():
            if producer.translation_buffer.should_flush():
                print(f"\n⏱️  Buffer timeout ({producer.translation_buffer.timeout_seconds}s) - processing accumulated text...")
                content, speaker_id, history = producer.translation_buffer.flush()
                if content:
                    await process_complete_sentence(content, speaker_id, history=history)

# ============== Audio Capture ==============

def audio_capture_thread():
    """Background thread that captures audio from microphone"""
    try:
        producer.pyaudio_instance = pyaudio.PyAudio()
        
        # Open microphone stream with selected device
        device_index = config.get("mic_device_index")  # None = default
        producer.stream = producer.pyaudio_instance.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        
        print(f"\n{'='*50}")
        print(f"MICROPHONE: {config['mic_device_name']}")
        print(f"LISTENING... Speak {active_session.source_lang_name}!")
        print(f"Translations will appear in {active_session.target_lang_name}")
        print("Press Ctrl+C to stop")
        print(f"{'='*50}\n")
        
        while producer.is_running:
            try:
                data = producer.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                # Only send if connection exists and is healthy
                if producer.dg_connection and producer.connection_healthy:
                    try:
                        producer.dg_connection.send(data)
                    except Exception:
                        # Connection lost - don't spam logs, reconnect handler will fire
                        producer.connection_healthy = False
            except Exception as e:
                if producer.is_running:
                    print(f"Audio capture error: {e}")
                break
    
    except Exception as e:
        print(f"Audio thread error: {e}")
    
    finally:
        cleanup_audio()

def cleanup_audio():
    """Clean up audio resources"""
    if producer.stream:
        try:
            producer.stream.stop_stream()
            producer.stream.close()
        except:
            pass
        producer.stream = None
    
    if producer.pyaudio_instance:
        try:
            producer.pyaudio_instance.terminate()
        except:
            pass
        producer.pyaudio_instance = None

def start_producer():
    """Start the audio capture and Deepgram connection"""
    global active_session
    
    if producer.is_running:
        return
    
    producer.is_running = True
    producer.sentence_buffer = []
    producer.current_speaker = 0  # Reset speaker tracking
    producer.translation_buffer = TranslationBuffer()  # Initialize look-ahead buffer
    
    # Create Deepgram connection
    producer.dg_connection = deepgram_client.listen.live.v("1")
    producer.dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
    producer.dg_connection.on(LiveTranscriptionEvents.Error, on_error)
    producer.dg_connection.on(LiveTranscriptionEvents.Close, on_close)
    producer.dg_connection.on(LiveTranscriptionEvents.Open, on_open)
    
    # Configure Deepgram options with diarization enabled
    # Note: SDK v3.9 uses 'keywords' but Nova-3 requires 'keyterms' (not yet in this SDK version)
    options = LiveOptions(
        model="nova-3",
        language=active_session.source_lang,
        smart_format=True,
        encoding="linear16",
        channels=CHANNELS,
        sample_rate=SAMPLE_RATE,
        interim_results=True,
        punctuate=True,
        diarize=True,  # Enable speaker diarization for multi-speaker support
    )

    
    print("Starting Deepgram connection...")
    result = producer.dg_connection.start(options)
    if result is False:
        print("WARNING: Deepgram start() returned False, but proceeding...")
    
    # Mark session as live
    active_session.is_live = True
    
    # Auto-start recording
    recording_manager.start_recording(active_session.id)
    
    # Start audio capture thread
    producer.audio_thread = threading.Thread(target=audio_capture_thread, daemon=True)
    producer.audio_thread.start()

def stop_producer():
    """Stop the audio capture and Deepgram connection"""
    global active_session
    
    producer.is_running = False
    
    if active_session:
        active_session.is_live = False
    
    # Close Deepgram
    if producer.dg_connection:
        try:
            producer.dg_connection.finish()
        except Exception as e:
            print(f"Warning closing Deepgram: {e}")
        producer.dg_connection = None
    
    # Wait for audio thread
    if producer.audio_thread and producer.audio_thread.is_alive():
        producer.audio_thread.join(timeout=2.0)
    
    cleanup_audio()
    
    # Stop recording
    recording_manager.stop_recording()
    
    print("\nProducer stopped.")

# ============== Static File Routes ==============

@app.get("/", response_class=HTMLResponse)
async def serve_landing():
    return FileResponse("static/index.html")

@app.get("/listen/{session_code}", response_class=HTMLResponse)
async def serve_listener(session_code: str):
    return FileResponse("static/listener.html")

# ============== API Routes ==============

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/api/context")
async def get_context():
    """Get current context buffer info"""
    global CONTEXT_BUFFER_SIZE, translation_context
    return {
        "buffer_size": CONTEXT_BUFFER_SIZE,
        "current_count": len(translation_context),
        "context": [{"original": o, "translation": t} for o, t in translation_context]
    }

@app.post("/api/context/size/{size}")
async def set_context_size(size: int):
    """Set context buffer size (0-20)"""
    global CONTEXT_BUFFER_SIZE, translation_context
    if size < 0 or size > 20:
        raise HTTPException(status_code=400, detail="Size must be between 0 and 20")
    
    old_size = CONTEXT_BUFFER_SIZE
    CONTEXT_BUFFER_SIZE = size
    
    # Trim buffer if new size is smaller
    if len(translation_context) > size:
        translation_context = translation_context[-size:] if size > 0 else []
    
    print(f"📝 Context buffer size changed: {old_size} → {size}")
    return {"message": f"Context size set to {size}", "previous_size": old_size}

@app.post("/api/context/clear")
async def clear_context():
    """Clear the context buffer"""
    global translation_context
    old_count = len(translation_context)
    translation_context = []
    print(f"📝 Context buffer cleared ({old_count} items)")
    return {"message": f"Cleared {old_count} items from context"}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if not active_session or active_session.id != session_id.upper():
        raise HTTPException(status_code=404, detail="Session not found")
    return active_session.to_dict()

# ============== WebSocket Routes ==============

@app.websocket("/ws/listener/{session_id}")
async def listener_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for listeners"""
    # Debug logging
    print(f"DEBUG: Listener connecting with session_id='{session_id}'")
    print(f"DEBUG: Active session id='{active_session.id if active_session else 'None'}'")
    
    if not active_session or active_session.id != session_id.upper():
        print(f"DEBUG: Session mismatch! Rejecting connection.")
        await websocket.close(code=4004, reason="Session not found")
        return
    
    await websocket.accept()
    active_session.listener_ws_set.add(websocket)
    
    print(f"Listener connected ({len(active_session.listener_ws_set)} total)")
    
    # Send current session status
    await websocket.send_json({
        "type": "session_info",
        "session": active_session.to_dict()
    })
    
    try:
        while True:
            # Wait for message with timeout for keepalive
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=15.0)
                if message["type"] == "websocket.disconnect":
                    break
                # Handle ping from client
                if message.get("type") == "websocket.receive":
                    data = message.get("text", "")
                    if data == "ping":
                        await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send keepalive ping to client
                try:
                    await websocket.send_json({"type": "ping"})
                except:
                    break  # Connection lost
    
    except WebSocketDisconnect:
        pass
    
    finally:
        active_session.listener_ws_set.discard(websocket)
        print(f"Listener disconnected ({len(active_session.listener_ws_set)} remaining)")

# ============== Startup / Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    global active_session
    
    check_api_keys()
    
    # Use custom session ID from config
    session_id = config["session_id"].upper()
    active_session = Session(
        id=session_id,
        source_lang=config["source_lang"],
        target_lang=config["target_lang"],
        source_lang_name=config["source_lang_name"],
        target_lang_name=config["target_lang_name"],
        voice_id=config["voice_id"]
    )
    
    # Store event loop for background thread
    producer.loop = asyncio.get_running_loop()
    
    # Start producer
    start_producer()
    
    # Start buffer timeout checker background task
    producer.buffer_timeout_task = asyncio.create_task(buffer_timeout_checker())
    
    # Print listener link
    print(f"\n{'='*50}")
    print(f"Listener link: http://localhost:8000/listen/{session_id}")
    print("Share this link with your audience!")
    print(f"{'='*50}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    # Cancel buffer timeout task
    if producer.buffer_timeout_task:
        producer.buffer_timeout_task.cancel()
        try:
            await producer.buffer_timeout_task
        except asyncio.CancelledError:
            pass
    
    stop_producer()
    
    # Notify all listeners
    if active_session:
        await broadcast_to_listeners({
            "type": "status",
            "is_live": False,
            "message": "Server shutting down"
        })

# ============== Interactive Menu ==============

def print_menu(title: str, options: list, show_index: bool = True) -> None:
    """Print a numbered menu"""
    print(f"\n{title}")
    print("-" * len(title))
    for i, opt in enumerate(options, 1):
        if show_index:
            print(f"  {i}. {opt}")
        else:
            print(f"  {opt}")

def get_choice(prompt: str, max_val: int, default: int = 1) -> int:
    """Get user's numeric choice"""
    while True:
        try:
            raw = input(f"{prompt} [default: {default}]: ").strip()
            if not raw:
                return default
            choice = int(raw)
            if 1 <= choice <= max_val:
                return choice
            print(f"  Please enter a number between 1 and {max_val}")
        except ValueError:
            print("  Please enter a valid number")

def get_input_devices() -> list:
    """Get list of available audio input devices"""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        # Only include input devices
        if info.get('maxInputChannels', 0) > 0:
            name = info.get('name', f'Device {i}')
            devices.append((i, name))
    
    p.terminate()
    return devices

def interactive_setup() -> None:
    """Interactive configuration menu"""
    global config
    
    print("\n" + "=" * 50)
    print("   REALTIME TRANSLATOR - SETUP")
    print("=" * 50)
    
    # Source language
    lang_options = [f"{name} ({code})" for code, name, _ in LANGUAGES]
    print_menu("Select SOURCE language (what you will speak):", lang_options)
    
    # Default to Turkish (index 12)
    default_source = next((i+1 for i, (c,_,_) in enumerate(LANGUAGES) if c == "tr"), 1)
    source_choice = get_choice("Enter number", len(LANGUAGES), default_source)
    source_code, source_name, _ = LANGUAGES[source_choice - 1]
    config["source_lang"] = source_code
    config["source_lang_name"] = source_name
    print(f"  ✓ Source: {source_name}")
    
    # Target language
    print_menu("\nSelect TARGET language (translation output):", lang_options)
    
    # Default to English (index 1)
    default_target = 1
    target_choice = get_choice("Enter number", len(LANGUAGES), default_target)
    target_code, target_name, _ = LANGUAGES[target_choice - 1]
    config["target_lang"] = target_code
    config["target_lang_name"] = target_name
    print(f"  ✓ Target: {target_name}")
    
    # Voice selection
    voice_options = [f"{name} ({gender})" for _, name, gender in VOICES]
    print_menu("\nSelect VOICE for TTS output:", voice_options)
    
    # Default to Drew (index 2)
    default_voice = 2
    voice_choice = get_choice("Enter number", len(VOICES), default_voice)
    voice_id, voice_name, _ = VOICES[voice_choice - 1]
    config["voice_id"] = voice_id
    config["voice_name"] = voice_name
    print(f"  ✓ Voice: {voice_name}")
    
    # Microphone selection
    input_devices = get_input_devices()
    if input_devices:
        mic_options = [name for _, name in input_devices]
        print_menu("\nSelect MICROPHONE:", mic_options)
        
        mic_choice = get_choice("Enter number", len(input_devices), 1)
        mic_index, mic_name = input_devices[mic_choice - 1]
        config["mic_device_index"] = mic_index
        config["mic_device_name"] = mic_name
        print(f"  ✓ Microphone: {mic_name}")
    else:
        print("\n  No microphones found, using system default.")
    
    # Session ID (for persistent listener links)
    print("\n" + "-" * 50)
    print("Session ID (used in listener URL - same ID = same link after restart)")
    default_session = config.get("session_id", "LECTURE")
    session_input = input(f"  Enter session name [default: {default_session}]: ").strip()
    if session_input:
        # Sanitize: uppercase, alphanumeric only
        config["session_id"] = ''.join(c for c in session_input.upper() if c.isalnum())[:20]
    else:
        config["session_id"] = default_session
    print(f"  ✓ Session ID: {config['session_id']}")
    
    # Main speaker name
    print("\n" + "-" * 50)
    print("Main Speaker Name (displayed to listeners)")
    default_speaker = config.get("main_speaker_name", "Speaker")
    speaker_input = input(f"  Enter main speaker name [default: {default_speaker}]: ").strip()
    if speaker_input:
        config["main_speaker_name"] = speaker_input[:30]  # Limit length
    else:
        config["main_speaker_name"] = default_speaker
    print(f"  ✓ Main Speaker: {config['main_speaker_name']}")
    print("     (Other speakers will be shown as 'Contributor 1', 'Contributor 2')")
    
    # Summary
    print("\n" + "=" * 50)
    print("   CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"  Source:       {config['source_lang_name']} ({config['source_lang']})")
    print(f"  Target:       {config['target_lang_name']} ({config['target_lang']})")
    print(f"  Voice:        {config['voice_name']}")
    print(f"  Microphone:   {config['mic_device_name']}")
    print(f"  Session ID:   {config['session_id']}")
    print(f"  Main Speaker: {config['main_speaker_name']}")
    print(f"  Listener URL: http://localhost:8000/listen/{config['session_id']}")
    print("=" * 50)
    
    input("\nPress Enter to start the server...")

# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    
    print("\n🎙️  REALTIME TRANSLATOR")
    print("   Speak → Transcribe → Translate → TTS → Broadcast")
    
    # Run interactive setup
    interactive_setup()
    
    print("\nStarting server...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=30,
        ws_ping_timeout=60
    )
