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

# Speaker voice mapping for diarization (speaker_id -> voice_id, voice_name)
# Speaker 0 = Main speaker, Speaker 1 & 2 = Contributors
SPEAKER_VOICES = {
    0: ("29vD33N1CtxCmqQRPOHJ", "Drew"),      # Main speaker
    1: ("bIHbv24MWmeRgasZH58o", "Will"),       # Contributor 1 - Relaxed
    2: ("iP95p4xoKVk53GoZ742B", "Chris"),      # Contributor 2 - Friendly
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

# Specialized translation system prompt for Risale-i Nur lectures
RISALE_TRANSLATION_PROMPT = """You are an expert simultaneous interpreter translating "Risale-i Nur" theological lectures from Turkish to English.

**INPUT DATA:**
- [PREVIOUS_CONTEXT]: Recently spoken sentences (Read-only context).
- [TARGET_SEGMENT]: The text you must translate now.
- [NEXT_SEGMENT_PREVIEW]: The upcoming sentence (use this to resolve ambiguous cut-offs or unfinished thoughts).

**CRITICAL INSTRUCTIONS:**

1. **ASR ERROR CORRECTION (Phonetic Repair):**
   The input text originates from real-time speech recognition and may contain phonetic errors. Infer the intended word based on theological context.
   - Common Fixes:
     - "Liman" (port) → Correct to "Lem'a" (The Flash) when citing books.
     - "Şua" (light/beam) → Correct to "The Ray" (Book title: Şualar).
     - "Laan turcusu" → Correct to idiom "lahana turşusu" (contradiction).
     - "Bicrim" → Correct to "Cirim" (particle/body).

2. **THEOLOGICAL TERMINOLOGY MAPPING:**
   - Üstad → Keep as "Üstad" (do not translate as Master/Teacher).
   - Cenab-ı Hak → "Almighty God"
   - Tevhid → "Divine Unity"
   - Ene → "The Ego" or "The Self"
   - Haşir → "Resurrection"
   - Esma-i Hüsna → "The Divine Names"
   - Abi / Kardeş → "Brother" (Community context).

3. **TONE & DELIVERY (Text-to-Speech Optimization):**
   - Output ONLY the English translation.
   - The output must be natural, spoken English. Avoid robotic or academic phrasing.
   - If the speaker reads from a book: Use formal, elevated language.
   - If the speaker is explaining/joking: Use conversational, relaxed language.

4. **CONTEXTUAL CONTINUITY:**
   - If [TARGET_SEGMENT] is a sentence fragment that completes in [NEXT_SEGMENT_PREVIEW], translate it as a cohesive partial thought, using ellipses (...) if necessary.

**Output:** Provide only the English translation of the [TARGET_SEGMENT]."""

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
}

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
        self.sentence_buffer = []
        self.current_speaker = 0  # Track current speaker for diarization
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        # Reconnection state
        self.is_reconnecting = False
        self.connection_healthy = False
        # Buffer timeout tracking
        self.last_buffer_update = 0.0
        self.buffer_timeout_task: Optional[asyncio.Task] = None
        # Translation buffer for look-ahead (1-segment lag)
        self.translation_buffer: Optional['TranslationBuffer'] = None

producer = ProducerState()


class TranslationBuffer:
    """Buffer queue for look-ahead translation with 1-segment lag.
    
    Implements the buffering logic that allows the translation engine to
    'see ahead' by one segment, improving sentence continuity and context.
    """
    
    def __init__(self):
        self.queue = []  # List of (text, speaker_id) tuples
        self.history = []  # List of original texts for context (last 5)
    
    def add_segment(self, text: str, speaker_id: int):
        """Add a new segment to the buffer queue."""
        self.queue.append((text, speaker_id))
    
    def can_translate(self) -> bool:
        """Check if we have enough segments to translate (need >= 2)."""
        return len(self.queue) >= 2
    
    def get_translation_batch(self):
        """Get target segment and next preview when ready.
        
        Returns:
            tuple: (target_text, speaker_id, next_preview, history) or None
        """
        if not self.can_translate():
            return None
        
        # Target is the oldest segment (first in queue)
        target_text, speaker_id = self.queue[0]
        # Next preview is the second segment
        next_preview = self.queue[1][0] if len(self.queue) > 1 else None
        
        return (target_text, speaker_id, next_preview, self.history.copy())
    
    def commit_translation(self, text: str):
        """After translation, move target to history and pop from queue."""
        if self.queue:
            self.queue.pop(0)
            self.history.append(text)
            # Keep history limited to last 5
            if len(self.history) > 5:
                self.history.pop(0)
    
    def flush(self):
        """Force-process remaining segments (e.g., on speaker change or timeout).
        
        Returns:
            List of (text, speaker_id) tuples to process immediately
        """
        remaining = self.queue.copy()
        self.queue = []
        return remaining
    
    def has_pending(self) -> bool:
        """Check if there are any pending segments."""
        return len(self.queue) > 0
    
    def pending_count(self) -> int:
        """Get the number of pending segments."""
        return len(self.queue)

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


def construct_payload(history: list, current: str, next_segment: str = None) -> str:
    """Construct a structured payload with context for translation.
    
    Args:
        history: List of previous translated segments (up to 5)
        current: The current segment to translate
        next_segment: Optional preview of the next incoming segment
    
    Returns:
        Formatted string with XML-style delimiters
    """
    payload_parts = []
    
    # Previous context (read-only for model reference)
    if history:
        payload_parts.append("[PREVIOUS_CONTEXT]")
        for segment in history[-5:]:  # Last 5 segments
            payload_parts.append(segment)
        payload_parts.append("[/PREVIOUS_CONTEXT]")
        payload_parts.append("")
    
    # Target segment (what to translate)
    payload_parts.append("[TARGET_SEGMENT]")
    payload_parts.append(current)
    payload_parts.append("[/TARGET_SEGMENT]")
    
    # Next segment preview (for sentence completion)
    if next_segment:
        payload_parts.append("")
        payload_parts.append("[NEXT_SEGMENT_PREVIEW]")
        payload_parts.append(next_segment)
        payload_parts.append("[/NEXT_SEGMENT_PREVIEW]")
    
    return "\n".join(payload_parts)

async def translate_text(text: str, history: list = None, next_segment: str = None) -> str:
    """Translate text using OpenAI with context-aware prompting.
    
    Args:
        text: The text to translate
        history: Optional list of previous original texts for context
        next_segment: Optional preview of the next incoming segment
    """
    global translation_context
    
    try:
        # Use specialized Risale-i Nur prompt if context is enabled
        if config.get("use_risale_context", False):
            system_prompt = RISALE_TRANSLATION_PROMPT
            # Use structured payload with XML-style context delimiters
            user_content = construct_payload(
                history=history or [],
                current=text,
                next_segment=next_segment
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


async def generate_speech(text: str, speaker_id: int = 0) -> Optional[bytes]:
    """Generate speech using ElevenLabs with speaker-specific voice"""
    try:
        # Get voice based on speaker ID (for diarization)
        if speaker_id in SPEAKER_VOICES:
            voice_id, voice_name = SPEAKER_VOICES[speaker_id]
        else:
            # Fallback to main speaker voice for unknown speaker IDs
            voice_id, voice_name = SPEAKER_VOICES[0]
        
        audio = elevenlabs_client.generate(
            text=text,
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
                                      history: list = None, next_segment: str = None):
    """Process a complete sentence through the translation + TTS pipeline.
    
    Args:
        text: The text to translate
        speaker_id: ID of the speaker for voice selection
        history: Optional list of previous texts for context
        next_segment: Optional preview of next segment
    """
    if not active_session or not active_session.is_live:
        return
    
    # Filter out very short fragments (breathing, filler sounds, etc.)
    words = text.strip().split()
    if len(words) < MIN_WORDS_TO_PROCESS:
        print(f"  (skipped short fragment: '{text}')")
        return
    
    # Get speaker name for display
    if speaker_id in SPEAKER_VOICES:
        _, speaker_name = SPEAKER_VOICES[speaker_id]
    else:
        speaker_name = f"Speaker {speaker_id}"
    
    # Clear the buffering line and show the translation
    print(" " * 80, end="\r")  # Clear buffering line
    print(f"\n{'─'*50}")
    print(f"🎤 [{speaker_name}] [{active_session.source_lang_name}] {text}")
    
    # Translate with context
    translation = await translate_text(text, history=history, next_segment=next_segment)
    print(f"🔊 [{speaker_name}] [{active_session.target_lang_name}] {translation}")
    print(f"{'─'*50}")
    
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
    """Handle incoming transcription from Deepgram with speaker diarization and look-ahead buffering."""
    transcript = result.channel.alternatives[0].transcript
    is_final = result.is_final
    words = result.channel.alternatives[0].words
    
    if not transcript:
        return
    
    # Extract speaker ID from diarization (if available)
    speaker_id = producer.current_speaker  # Default to current speaker
    if words and len(words) > 0:
        # Get speaker from first word of this transcript
        first_word = words[0]
        if hasattr(first_word, 'speaker') and first_word.speaker is not None:
            # Cap speaker ID at 2 (we only have 3 voices: 0, 1, 2)
            speaker_id = min(first_word.speaker, 2)
    
    if is_final:
        # Check if speaker changed - if so, flush the buffers first
        if speaker_id != producer.current_speaker:
            # Flush sentence buffer
            if producer.sentence_buffer:
                partial_text = " ".join(producer.sentence_buffer).strip()
                if partial_text and producer.translation_buffer:
                    producer.translation_buffer.add_segment(partial_text, producer.current_speaker)
                producer.sentence_buffer = []
            
            # Flush translation buffer - process all pending without look-ahead
            if producer.translation_buffer and producer.translation_buffer.has_pending():
                remaining = producer.translation_buffer.flush()
                for seg_text, seg_speaker in remaining:
                    if producer.loop:
                        asyncio.run_coroutine_threadsafe(
                            process_complete_sentence(seg_text, seg_speaker),
                            producer.loop
                        )
        
        # Update current speaker
        producer.current_speaker = speaker_id
        
        # Add to sentence buffer and track update time
        producer.sentence_buffer.append(transcript)
        producer.last_buffer_update = time.time()
        full_text = " ".join(producer.sentence_buffer)
        
        # Check if we have a complete sentence (ends with . ! ?)
        if full_text.strip() and full_text.strip()[-1] in '.!?':
            text_to_translate = full_text.strip()
            producer.sentence_buffer = []  # Reset buffer
            producer.last_buffer_update = 0.0  # Reset timeout
            
            # Add to translation buffer for look-ahead processing
            if producer.translation_buffer:
                producer.translation_buffer.add_segment(text_to_translate, speaker_id)
                
                # Process if we have enough segments (>= 2 for look-ahead)
                while producer.translation_buffer.can_translate():
                    batch = producer.translation_buffer.get_translation_batch()
                    if batch and producer.loop:
                        target_text, target_speaker, next_preview, history = batch
                        asyncio.run_coroutine_threadsafe(
                            process_complete_sentence(
                                target_text, target_speaker,
                                history=history, next_segment=next_preview
                            ),
                            producer.loop
                        )
                        producer.translation_buffer.commit_translation(target_text)
            else:
                # Fallback: direct processing if buffer not initialized
                if producer.loop:
                    asyncio.run_coroutine_threadsafe(
                        process_complete_sentence(text_to_translate, speaker_id),
                        producer.loop
                    )
        else:
            # Show buffering status with speaker
            speaker_name = SPEAKER_VOICES.get(speaker_id, (None, f"Speaker {speaker_id}"))[1]
            pending = producer.translation_buffer.pending_count() if producer.translation_buffer else 0
            print(f"  [{speaker_name}] (buffering: {full_text}) [queue: {pending}]", end="\r")
    else:
        # Show interim results with speaker
        current = " ".join(producer.sentence_buffer) + " " + transcript if producer.sentence_buffer else transcript
        speaker_name = SPEAKER_VOICES.get(speaker_id, (None, f"Speaker {speaker_id}"))[1]
        print(f"  [{speaker_name}] ... {current}", end="\r")

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

BUFFER_TIMEOUT_SECONDS = 3.0  # Force-process buffer after 3 seconds

async def buffer_timeout_checker():
    """Background task that force-processes buffered sentences after timeout"""
    while producer.is_running:
        await asyncio.sleep(1)  # Check every second
        
        # Check if we have buffered text that's been waiting too long
        if (producer.sentence_buffer and 
            producer.last_buffer_update > 0 and
            time.time() - producer.last_buffer_update >= BUFFER_TIMEOUT_SECONDS):
            
            # Force-process the buffered text
            full_text = " ".join(producer.sentence_buffer).strip()
            if full_text:
                speaker_id = producer.current_speaker
                print(f"\n⏱️  Buffer timeout - processing incomplete sentence...")
                producer.sentence_buffer = []
                producer.last_buffer_update = 0.0
                
                # Add to translation buffer instead of direct processing
                if producer.translation_buffer:
                    producer.translation_buffer.add_segment(full_text, speaker_id)
        
        # Also check translation buffer - flush if oldest item has been waiting
        if producer.translation_buffer and producer.translation_buffer.has_pending():
            # If we have pending items but no new segments coming in, flush after timeout
            if (producer.last_buffer_update == 0.0 and 
                not producer.sentence_buffer):
                print(f"\n⏱️  Translation queue timeout - flushing pending segments...")
                remaining = producer.translation_buffer.flush()
                for seg_text, seg_speaker in remaining:
                    await process_complete_sentence(seg_text, seg_speaker)

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
    
    # Summary
    print("\n" + "=" * 50)
    print("   CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"  Source:     {config['source_lang_name']} ({config['source_lang']})")
    print(f"  Target:     {config['target_lang_name']} ({config['target_lang']})")
    print(f"  Voice:      {config['voice_name']}")
    print(f"  Microphone: {config['mic_device_name']}")
    print(f"  Session ID: {config['session_id']}")
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
