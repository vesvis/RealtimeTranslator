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
from fish_audio_sdk import Session as FishAudioSession, TTSRequest

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
BUILTIN_VOICES = [
    ("29vD33N1CtxCmqQRPOHJ", "Drew", "Male - Main Speaker"),
    ("bIHbv24MWmeRgasZH58o", "Will", "Male - Relaxed, Casual"),
    ("iP95p4xoKVk53GoZ742B", "Chris", "Male - Friendly, Down-to-Earth"),
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger", "Male - Laid-Back"),
    ("UgBBYS2sOqTuMpoF3BR0", "Mark", "Male - Natural"),
    ("nPczCjzI2devNBz1zQrb", "Brian", "Male - Deep, Comforting"),
    ("onwK4e9ZLuTAKqWW03F9", "Daniel", "Male - British, Steady"),
    ("JBFqnCBsd6RMkjVDRZzb", "George", "Male - British, Warm"),
]

# Available OpenAI voices (male first, female at end)
OPENAI_VOICES = [
    ("onyx", "Onyx", "Male - Deep, Authoritative"),
    ("echo", "Echo", "Male - Warm, Soft"),
    ("fable", "Fable", "Male - British, Expressive"),
    ("alloy", "Alloy", "Neutral, Versatile"),
    ("nova", "Nova", "Female - Energetic"),
    ("shimmer", "Shimmer", "Female - Clear"),
]

# Available Fish Audio voices (Reference IDs)
# Users can add more via custom_voices.json
FISH_AUDIO_VOICES = [
    ("bf322df2096a46f18c579d0baa36f41d", "Adrian", "A steady narrator"), 
    ("79d0bd3e4e5444b18f7b6d89b5927bf1", "Jordan", "A motivational speaker"),
    ("536d3a5e000945adb7038665781a4aca", "Ethan", "A curious explorer"),
]
def load_custom_voices():
    """Load custom voices from custom_voices.json if it exists."""
    custom_file = os.path.join(os.path.dirname(__file__), "custom_voices.json")
    try:
        with open(custom_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            custom = data.get("custom_voices", [])
            return [(v["id"], v["name"], v.get("description", "Custom")) for v in custom]
    except FileNotFoundError:
        return []
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Error loading custom_voices.json: {e}")
        return []

# Merge custom voices (at top) with built-in voices
VOICES = load_custom_voices() + BUILTIN_VOICES

# Speaker voice mapping for diarization (speaker_id -> voice_id)
# Names are set dynamically via get_speaker_name()

SPEAKER_VOICES_ELEVENLABS = {
    0: "29vD33N1CtxCmqQRPOHJ",  # Main speaker - Drew voice
    1: "bIHbv24MWmeRgasZH58o",  # Contributor 1 - Will voice
    2: "iP95p4xoKVk53GoZ742B",  # Contributor 2 - Chris voice
}

SPEAKER_VOICES_OPENAI = {
    0: "onyx",   # Main
    1: "fable",  # Contributor 1
    2: "echo",   # Contributor 2
}

SPEAKER_VOICES_FISH_AUDIO = {
    0: "bf322df2096a46f18c579d0baa36f41d",  # Adrian (Main default)
    1: "79d0bd3e4e5444b18f7b6d89b5927bf1",  # Jordan
    2: "536d3a5e000945adb7038665781a4aca",  # Ethan
}

# Available Deepgram TTS voices (Aura-2 models, male first)
DEEPGRAM_VOICES = [
    ("aura-2-orion-en", "Orion", "Male - Strong, Confident"),
    ("aura-2-arcas-en", "Arcas", "Male - Warm, Friendly"),
    ("aura-2-apollo-en", "Apollo", "Male - Professional"),
    ("aura-2-zeus-en", "Zeus", "Male - Authoritative"),
    ("aura-2-orpheus-en", "Orpheus", "Male - Expressive"),
    ("aura-2-atlas-en", "Atlas", "Male - Deep"),
    ("aura-2-thalia-en", "Thalia", "Female - Vibrant"),
    ("aura-2-andromeda-en", "Andromeda", "Female - Warm"),
    ("aura-2-helena-en", "Helena", "Female - Clear"),
    ("aura-2-luna-en", "Luna", "Female - Calm"),
]

SPEAKER_VOICES_DEEPGRAM = {
    0: "aura-2-orion-en",   # Main - Orion (confident male)
    1: "aura-2-arcas-en",   # Contributor 1 - Arcas (warm male)
    2: "aura-2-apollo-en",  # Contributor 2 - Apollo (professional male)
}

# Current active map (defaults to ElevenLabs)
SPEAKER_VOICES = SPEAKER_VOICES_ELEVENLABS

# ============== Translation Prompts ==============

# Default prompt (always available, embedded in code)
DEFAULT_PROMPT = {
    "name": "Default Translator",
    "description": "General-purpose translation with natural speech formatting",
    "prompt": """You are an expert simultaneous interpreter. Your goal is to produce natural, fluent speech while preserving the speaker's meaning accurately.

**INPUT FORMAT:**
You will receive:
- [CONTEXT]: Previous translated sentences (for continuity - read only)
- [TARGET_BATCH]: The current text to translate
- Additional user-provided context may be included below with domain-specific terminology or style guidance.

**TRANSLATION APPROACH:**
- Preserve the speaker's meaning and intent accurately.
- Favor idiomatic expressions in the target language when they fit naturally.
- Restructure sentences when needed for clarity, but stay close to the original when it works well.
- If a phrase sounds awkward literally, find a natural equivalent that keeps the meaning.

**INSTRUCTIONS:**
1. Translate accurately but naturally - balance faithfulness with fluency.
2. Remove stuttering, false starts, and filler words.
3. Output each sentence on a NEW LINE for natural pauses.
4. Use "||" on its own line for major topic shifts or speaker transitions.

**Output:** Provide ONLY the translation with line breaks."""
}

def load_custom_prompts():
    """Load custom translation prompts from prompts.json if it exists."""
    prompts_file = os.path.join(os.path.dirname(__file__), "prompts.json")
    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing prompts.json: {e}")
        return {}

# Build prompts dict: default (always available) + custom prompts from file
TRANSLATION_PROMPTS = {"default": DEFAULT_PROMPT}
TRANSLATION_PROMPTS.update(load_custom_prompts())

def get_prompt(prompt_key: str) -> str:
    """Get a translation prompt by key."""
    if prompt_key in TRANSLATION_PROMPTS:
        return TRANSLATION_PROMPTS[prompt_key]["prompt"]
    return TRANSLATION_PROMPTS.get("default", {}).get("prompt", DEFAULT_TRANSLATION_PROMPT)

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
    "prompt_key": "risale_i_nur",  # Key for selected translation prompt
    "session_id": "LECTURE",  # Custom session ID for persistent listener links
    "main_speaker_name": "Speaker",  # Name displayed for main speaker
    "enable_review_pass": False,  # Enable second-pass translation review
    "translation_model": "gpt-4o-mini",  # Model for translation (gpt-4o or gpt-4o-mini)
    "tts_service": "elevenlabs",  # Service: elevenlabs, openai, fish_audio, deepgram
    "tts_speed": 0.8,  # Speech speed (0.7-1.2 for ElevenLabs, 0.25-4.0 for OpenAI)
}

def get_speaker_name(speaker_id: int) -> str:
    """Get display name for a speaker based on ID."""
    if speaker_id == 0:
        return config.get("main_speaker_name", "Speaker")
    else:
        return f"Contributor {speaker_id}"

def get_speaker_voice(speaker_id: int) -> str:
    """Get voice ID for a speaker based on current service configuration.
    
    For main speaker (0), uses the voice selected in GUI (ElevenLabs only) or default for other services.
    For contributors, uses the predefined per-service mapping.
    """
    service = config.get("tts_service", "elevenlabs")
    
    if service == "openai":
        return SPEAKER_VOICES_OPENAI.get(speaker_id, SPEAKER_VOICES_OPENAI[0])
        
    elif service == "fish_audio":
        return SPEAKER_VOICES_FISH_AUDIO.get(speaker_id, SPEAKER_VOICES_FISH_AUDIO[0])
    
    elif service == "deepgram":
        return SPEAKER_VOICES_DEEPGRAM.get(speaker_id, SPEAKER_VOICES_DEEPGRAM[0])
    
    # ElevenLabs - use config voice for main speaker, predefined for others
    if speaker_id == 0:
        return config.get("voice_id", SPEAKER_VOICES_ELEVENLABS[0])
    return SPEAKER_VOICES_ELEVENLABS.get(speaker_id, SPEAKER_VOICES_ELEVENLABS[0])

# ============== API Clients ==============

# Default keys from .env (used as fallback)
_env_deepgram_key = os.getenv("DEEPGRAM_API_KEY")
_env_openai_key = os.getenv("OPENAI_API_KEY")
_env_elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
_env_fish_audio_key = os.getenv("FISH_AUDIO_API_KEY")

# Active API keys (can be overwritten by config)
deepgram_key = _env_deepgram_key
openai_key = _env_openai_key
elevenlabs_key = _env_elevenlabs_key
fish_audio_key = _env_fish_audio_key

# API Clients (initialized with defaults, can be re-initialized)
deepgram_client = DeepgramClient(deepgram_key or "dummy_key")
openai_client = AsyncOpenAI(api_key=openai_key or "dummy_key")
elevenlabs_client = ElevenLabs(api_key=elevenlabs_key or "dummy_key")
# Fish Audio session is created per-request to handle async context properly


def init_api_clients(user_keys: dict = None):
    """Initialize or re-initialize API clients with user-provided keys.
    
    Args:
        user_keys: Dict with keys like {"deepgram": "key", "openai": "key", ...}
                   If a key is empty/None, falls back to .env value.
    """
    global deepgram_key, openai_key, elevenlabs_key, fish_audio_key
    global deepgram_client, openai_client, elevenlabs_client
    
    if user_keys:
        # Use user-provided keys if present, otherwise fall back to .env
        deepgram_key = user_keys.get("deepgram") or _env_deepgram_key
        openai_key = user_keys.get("openai") or _env_openai_key
        elevenlabs_key = user_keys.get("elevenlabs") or _env_elevenlabs_key
        fish_audio_key = user_keys.get("fish_audio") or _env_fish_audio_key
    else:
        # Use .env keys
        deepgram_key = _env_deepgram_key
        openai_key = _env_openai_key
        elevenlabs_key = _env_elevenlabs_key
        fish_audio_key = _env_fish_audio_key
    
    # Re-initialize clients with new keys
    deepgram_client = DeepgramClient(deepgram_key or "dummy_key")
    openai_client = AsyncOpenAI(api_key=openai_key or "dummy_key")
    elevenlabs_client = ElevenLabs(api_key=elevenlabs_key or "dummy_key")
    
    print(f"🔑 API clients initialized (Deepgram: {'✓' if deepgram_key else '✗'}, "
          f"OpenAI: {'✓' if openai_key else '✗'}, "
          f"ElevenLabs: {'✓' if elevenlabs_key else '✗'}, "
          f"Fish Audio: {'✓' if fish_audio_key else '✗'})")


def check_api_keys(user_keys: dict = None):
    """Validate that required API keys are present.
    
    Args:
        user_keys: Optional dict of user-provided keys to validate
    """
    dg = (user_keys or {}).get("deepgram") or deepgram_key
    oai = (user_keys or {}).get("openai") or openai_key
    el = (user_keys or {}).get("elevenlabs") or elevenlabs_key
    
    missing = []
    if not dg or dg == "dummy_key":
        missing.append("DEEPGRAM_API_KEY")
    if not oai or oai == "dummy_key":
        missing.append("OPENAI_API_KEY")
    
    # ElevenLabs is optional if using other TTS
    tts_service = config.get("tts_service", "elevenlabs")
    if tts_service == "elevenlabs" and (not el or el == "dummy_key"):
        missing.append("ELEVENLABS_API_KEY")
    
    if missing:
        print(f"⚠️  Missing API keys: {', '.join(missing)}")
        print("   Please provide them in the GUI or .env file")
        return False
    return True

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
    """Sentence-based translation buffer.
    
    Collects complete sentences (ending with . ? !) and batches them for translation.
    Short sentences (< batch_threshold words) are batched together.
    Uses timeout as fallback to prevent stalling.
    """
    
    def __init__(self, timeout_seconds: float = 8.0, batch_threshold: int = 10):
        self.pending_text = ""  # Incomplete text waiting for sentence end
        self.sentence_queue = []  # List of complete sentences ready for batching
        self.last_flush_time = time.time()
        self.history = []  # Last 5 segments for context
        self.timeout_seconds = timeout_seconds
        self.batch_threshold = batch_threshold  # Min words to translate alone
        self.current_speaker = 0
    
    def add_segment(self, text: str, speaker_id: int = 0):
        """Add incoming ASR text, extracting complete sentences."""
        self.pending_text = (self.pending_text + " " + text).strip()
        self.current_speaker = speaker_id
        
        # Extract complete sentences from pending text
        self._extract_sentences()
    
    def _extract_sentences(self):
        """Extract complete sentences from pending_text into sentence_queue."""
        import re
        
        # Pattern: Match sentences ending with . ? ! followed by space or end
        # Keep the punctuation with the sentence
        pattern = r'([^.!?]*[.!?])(?:\s+|$)'
        
        while True:
            match = re.match(pattern, self.pending_text)
            if not match:
                break
            
            sentence = match.group(1).strip()
            if sentence:
                self.sentence_queue.append(sentence)
            
            # Remove matched sentence from pending
            self.pending_text = self.pending_text[match.end():].strip()
    
    def should_flush(self) -> bool:
        """Check if we should flush for translation."""
        if not self.sentence_queue and not self.pending_text.strip():
            return False
        
        # Check if we have enough content in the queue
        if self.sentence_queue:
            total_words = sum(len(s.split()) for s in self.sentence_queue)
            
            # If we have enough words, flush
            if total_words >= self.batch_threshold:
                return True
            
            # If we have multiple sentences, batch them
            if len(self.sentence_queue) >= 2:
                return True
        
        # Timeout fallback - flush whatever we have
        time_elapsed = time.time() - self.last_flush_time
        if time_elapsed > self.timeout_seconds:
            # Either sentences in queue or pending text
            return bool(self.sentence_queue) or bool(self.pending_text.strip())
        
        return False
    
    def get_buffer_content(self) -> str:
        """Get current buffer content (for display)."""
        queued = " ".join(self.sentence_queue)
        if self.pending_text:
            return f"{queued} {self.pending_text}".strip()
        return queued
    
    def get_word_count(self) -> int:
        """Get total word count in buffer."""
        content = self.get_buffer_content()
        return len(content.split()) if content else 0
    
    def get_time_elapsed(self) -> float:
        """Get seconds since last flush."""
        return time.time() - self.last_flush_time
    
    def get_sentence_count(self) -> int:
        """Get number of complete sentences in queue."""
        return len(self.sentence_queue)
    
    def flush(self) -> tuple:
        """Flush sentences for translation.
        
        Returns:
            tuple: (content, speaker_id, history)
        """
        # Combine queued sentences
        if self.sentence_queue:
            content = " ".join(self.sentence_queue)
            self.sentence_queue = []
        elif self.pending_text.strip():
            # Timeout case: flush incomplete text
            content = self.pending_text.strip()
            self.pending_text = ""
        else:
            content = ""
        
        history = self.history.copy()
        speaker_id = self.current_speaker
        
        # Add to history for context
        if content:
            self.history.append(content)
            if len(self.history) > 5:
                self.history.pop(0)
        
        self.last_flush_time = time.time()
        return content, speaker_id, history
    
    def has_pending(self) -> bool:
        """Check if there is any pending content."""
        return bool(self.sentence_queue) or bool(self.pending_text.strip())
    
    def pending_count(self) -> int:
        """Get count of pending sentences."""
        count = len(self.sentence_queue)
        if self.pending_text.strip():
            count += 1
        return count

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
            except Exception:
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


# Common words for language detection (echo prevention)
LANG_INDICATORS = {
    "en": {"the", "is", "are", "was", "were", "have", "has", "been", "being", "this", "that", "with", "from", "they", "will", "would", "could", "should"},
    "tr": {"bir", "bu", "ve", "için", "ile", "olan", "var", "yok", "gibi", "daha", "çok", "ama", "ancak", "için", "olarak"},
    "ar": {"في", "من", "إلى", "على", "هذا", "هذه", "الذي", "التي", "كان", "يكون"},
    "de": {"der", "die", "das", "und", "ist", "sind", "war", "haben", "werden", "nicht"},
    "fr": {"le", "la", "les", "est", "sont", "était", "ont", "avec", "pour", "dans"},
    "es": {"el", "la", "los", "las", "es", "son", "está", "están", "con", "para"},
}

async def is_likely_target_language(text: str, target_lang: str) -> bool:
    """
    Detect if text is likely already in the target language.
    Used to prevent echo/feedback when TTS output is picked up by microphone.
    
    Returns True if we should SKIP this text (it's already in target language).
    """
    if not text or len(text.strip()) < 10:
        return False
    
    words = set(text.lower().split())
    
    # Get indicators for target language
    target_indicators = LANG_INDICATORS.get(target_lang, set())
    if not target_indicators:
        return False  # Can't detect, allow translation
    
    # Count how many target language indicator words are present
    matches = len(words & target_indicators)
    word_count = len(words)
    
    # If more than 30% of words are target language indicators, likely already translated
    if word_count > 0 and matches / word_count > 0.3:
        return True
    
    # If at least 3 common target language words found in short text
    if word_count < 15 and matches >= 3:
        return True
    
    return False


async def translate_text(text: str, history: list = None) -> str:
    """Translate text using OpenAI with context-aware prompting.
    
    Args:
        text: The text to translate
        history: Optional list of previous original texts for context
    """
    global translation_context
    
    try:
        # Get the selected translation prompt
        prompt_key = config.get("prompt_key", "default")
        base_prompt = get_prompt(prompt_key)
        
        # CRITICAL: Inject source and target language into the prompt
        source_lang = config.get("source_lang_name", "English")
        target_lang = config.get("target_lang_name", "Turkish")
        
        # Prepend AND append language direction to ensure it overrides any hardcoded language in custom prompts
        language_prefix = f"""**TRANSLATION DIRECTION:** {source_lang} → {target_lang}
You MUST translate from {source_lang} into {target_lang}. IGNORE any conflicting language instructions below.

"""
        # User-provided context comes AFTER the base prompt (domain-specific additions)
        user_context = config.get("user_context", "")
        context_section = ""
        if user_context:
            # Debug: Show that user context is being used
            preview = user_context[:100].replace('\n', ' ')
            print(f"  📝 Using user context: \"{preview}{'...' if len(user_context) > 100 else ''}\"")
            
            context_section = f"""

**ADDITIONAL USER CONTEXT:**
{user_context}
"""
        
        language_suffix = f"""

**CRITICAL OVERRIDE:** Output ONLY in {target_lang}. Do NOT output in English or any other language unless {target_lang} IS English.
"""
        # Structure: language_prefix + base_prompt + user_context + language_suffix
        system_prompt = language_prefix + base_prompt + context_section + language_suffix
        
        # Use structured payload with context delimiters
        user_content = construct_payload(
            history=history or [],
            current=text
        )
        
        response = await openai_client.chat.completions.create(
            model=config.get("translation_model", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        translation = response.choices[0].message.content.strip()
        
        # Optional: Run translation review/polish pass with context history
        if config.get("enable_review_pass", False):
            translation = await polish_translation(text, translation, translation_context)
        
        # Add to context buffer (limit size) for backward compatibility
        translation_context.append((text, translation))
        if len(translation_context) > CONTEXT_BUFFER_SIZE:
            translation_context.pop(0)
        
        return translation
    except Exception as e:
        print(f"Translation error: {e}")
        return "[Translation error]"


# Translation review/polish prompt - includes theological context for Risale-i Nur translations
def get_review_prompt(source_lang: str, target_lang: str) -> str:
    """Build the review prompt with dynamic source/target languages."""
    return f"""You are a translation quality reviewer specializing in Islamic theological texts, particularly Risale-i Nur (Said Nursi's works).

**TRANSLATION DIRECTION:** {source_lang} → {target_lang}

**DOMAIN CONTEXT:**
This is a live interpretation of a lecture on Risale-i Nur. Key theological terms that MUST be preserved or correctly rendered:
- "Risale-i Nur" / "Risale" → Keep as-is or "Epistles of Light"
- "Üstad" / "Bediüzzaman" → "Master Nursi" or "Bediuzzaman Said Nursi"
- "iman" → "faith" or "belief" (never "religion")
- "hakikat" → "reality" or "truth" (spiritual sense)
- "marifetullah" → "knowledge of God" or "divine knowledge"
- "tevhid" → "divine unity" or "oneness of God"
- "tefekkür" → "contemplation" or "reflection"
- "tafsilat" → "detailed exposition" or "elaboration"
- "mana-i harfi" → "indicative meaning" (pointing to Creator)
- "mana-i ismi" → "nominative meaning" (thing in itself)
- "nur" → "light" (divine/spiritual light)
- "zulmet" → "darkness" (spiritual darkness)

**RULES:**
1. If the translation is already natural and accurate, output it UNCHANGED.
2. If the translation sounds awkward, too literal, or has grammar issues, IMPROVE it.
3. PRESERVE all line breaks exactly as they are (they indicate pauses).
4. PRESERVE all "||" markers exactly as they are (they indicate longer pauses).
5. Ensure theological terms are rendered correctly per the domain context above.
6. Keep the meaning faithful to the source - only improve {target_lang} expression.
7. Use the previous translations for context continuity.
8. Do NOT add explanations - output ONLY the translation in {target_lang}.
{{history_section}}
**Original ({source_lang} source):**
{{original}}

**Translation to review:**
{{translation}}

**Output:** The polished translation in {target_lang} (improved if needed, unchanged if already good). Preserve all line breaks and || markers."""


def construct_history_section(history: list, source_lang: str, target_lang: str) -> str:
    """Build the history section for the review prompt."""
    if not history:
        return ""
    
    lines = ["\n**PREVIOUS TRANSLATIONS (for context continuity):**"]
    for orig, trans in history[-3:]:  # Last 3 for context
        lines.append(f"- {source_lang}: {orig[:100]}{'...' if len(orig) > 100 else ''}")
        lines.append(f"  {target_lang}: {trans[:100]}{'...' if len(trans) > 100 else ''}")
    lines.append("")
    return "\n".join(lines)


async def polish_translation(original: str, translation: str, history: list = None) -> str:
    """Optional second pass to review and polish the translation with theological context.
    
    Args:
        original: The original source text
        translation: The first-pass translation
        history: Optional list of (original, translation) tuples for context
    """
    try:
        # Get languages from config
        source_lang = config.get("source_lang_name", "English")
        target_lang = config.get("target_lang_name", "Turkish")
        
        history_section = construct_history_section(history or [], source_lang, target_lang)
        
        # Build the review prompt with dynamic languages
        review_prompt = get_review_prompt(source_lang, target_lang)
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": review_prompt.format(
                    original=original,
                    translation=translation,
                    history_section=history_section
                )}
            ],
            temperature=0.2,
            max_tokens=500
        )
        
        polished = response.choices[0].message.content.strip()
        
        # If the model returned something, use it; otherwise keep original
        if polished and len(polished) > 5:
            return polished
        return translation
        
    except Exception as e:
        print(f"Review pass error (using original): {e}")
        return translation

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



async def generate_openai_audio(text: str, voice_id: str) -> Optional[bytes]:
    """Generate speech using OpenAI TTS."""
    try:
        # Simple cleanup for OpenAI
        clean_text = text.replace("||", "... ")
        
        response = await openai_client.audio.speech.create(
            model="tts-1",
            voice=voice_id,
            input=clean_text
        )
        return response.content
    except Exception as e:
        print(f"OpenAI TTS Error: {e}")
        return None

async def generate_fish_audio(text: str, voice_id: str) -> Optional[bytes]:
    """Generate speech using Fish Audio."""
    try:
        from fishaudio import FishAudio
        
        key = os.getenv("FISH_AUDIO_API_KEY") or "dummy_key"
        client = FishAudio(api_key=key)
        
        # Use stream and collect for full audio
        audio = client.tts.stream(
            text=text,
            reference_id=voice_id
        ).collect()
        
        return audio
        
    except ImportError:
        # Fall back to old SDK if new one not available
        try:
            key = os.getenv("FISH_AUDIO_API_KEY") or "dummy_key"
            
            # Try synchronous approach with old SDK
            from fish_audio_sdk import Session as FishSession
            
            session = FishSession(key)
            audio_bytes = b""
            for chunk in session.tts(TTSRequest(text=text, reference_id=voice_id)):
                audio_bytes += chunk
            return audio_bytes
        except Exception as e2:
            print(f"Fish Audio TTS Error (fallback): {e2}")
            return None
    except Exception as e:
        print(f"Fish Audio TTS Error: {e}")
        return None

async def generate_deepgram_audio(text: str, voice_id: str) -> Optional[bytes]:
    """Generate speech using Deepgram TTS (Aura-2)."""
    try:
        from deepgram import SpeakOptions
        
        # Configure TTS options
        options = SpeakOptions(
            model=voice_id,
            encoding="mp3"
        )
        
        # Use stream_memory() to get audio as BytesIO (new API, replaces deprecated stream())
        response = deepgram_client.speak.rest.v("1").stream_memory(
            source={"text": text},
            options=options
        )
        
        # Get the audio bytes from the BytesIO stream
        audio_data = response.stream.getvalue()
        return audio_data
        
    except Exception as e:
        print(f"Deepgram TTS Error: {e}")
        return None

async def generate_speech(text: str, speaker_id: int = 0) -> Optional[bytes]:
    """Generate speech using the configured service."""
    service = config.get("tts_service", "elevenlabs")
    
    # Get voice based on speaker ID (for diarization)
    voice_id = get_speaker_voice(speaker_id)
    
    # Format text with SSML breaks for natural pauses (applies mostly to EL)
    # Start with standard formatting
    formatted_text = format_text_for_tts(text)
    
    try:
        if service == "elevenlabs":
            from elevenlabs import VoiceSettings
            
            # Get speed from config and clamp to valid range (0.7-1.2)
            speed = config.get("tts_speed", 1.0)
            speed = max(0.7, min(1.2, speed))
            
            # ElevenLabs returns a generator - must collect all chunks
            audio_generator = elevenlabs_client.generate(
                text=formatted_text,
                voice=voice_id,
                model="eleven_turbo_v2",
                voice_settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    speed=speed
                )
            )
            # Collect audio chunks into bytes
            audio_bytes = b""
            for chunk in audio_generator:
                audio_bytes += chunk
            return audio_bytes
            
        elif service == "openai":
            return await generate_openai_audio(text, voice_id)
            
        elif service == "fish_audio":
            return await generate_fish_audio(text, voice_id)
        
        elif service == "deepgram":
            return await generate_deepgram_audio(text, voice_id)
            
        else:
            print(f"Unknown TTS service: {service}")
            return None
            
    except Exception as e:
        print(f"TTS Error ({service}): {e}")
        return None

async def broadcast_to_listeners(message: dict):
    """Broadcast a message to all connected listeners"""
    if not active_session:
        return
    
    disconnected = set()
    for ws in active_session.listener_ws_set:
        try:
            await ws.send_json(message)
        except Exception:
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
            except Exception:
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
        except Exception:
            pass
        producer.stream = None
    
    if producer.pyaudio_instance:
        try:
            producer.pyaudio_instance.terminate()
        except Exception:
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

@app.get("/settings", response_class=HTMLResponse)
async def serve_settings():
    return FileResponse("static/settings.html")

@app.get("/host", response_class=HTMLResponse)
async def serve_host():
    return FileResponse("static/host.html")

@app.get("/personal", response_class=HTMLResponse)
async def serve_personal():
    return FileResponse("static/personal.html")

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

# ============== GUI Control API Routes ==============

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "source_lang": config["source_lang"],
        "source_lang_name": config["source_lang_name"],
        "target_lang": config["target_lang"],
        "target_lang_name": config["target_lang_name"],
        "voice_id": config["voice_id"],
        "voice_name": config["voice_name"],
        "mic_device_index": config["mic_device_index"],
        "mic_device_name": config["mic_device_name"],
        "session_id": config["session_id"],
        "main_speaker_name": config["main_speaker_name"],
        "prompt_key": config["prompt_key"],
        "enable_review_pass": config["enable_review_pass"],
        "translation_model": config.get("translation_model", "gpt-4o-mini"),
    }

@app.post("/api/config")
async def update_config(new_config: dict):
    """Update configuration (only when producer is stopped)"""
    if producer.is_running:
        raise HTTPException(status_code=400, detail="Cannot update config while server is running")
    
    # Update allowed fields
    allowed_fields = [
        "source_lang", "source_lang_name", "target_lang", "target_lang_name",
        "voice_id", "voice_name", "mic_device_index", "mic_device_name",
        "session_id", "main_speaker_name", "prompt_key", "enable_review_pass",
        "translation_model"
    ]
    
    for field in allowed_fields:
        if field in new_config:
            config[field] = new_config[field]
    
    return {"message": "Configuration updated", "config": config}

@app.post("/api/config/live")
async def update_config_live(new_config: dict):
    """Update configuration WHILE running (only safe fields).
    
    Fields that can be changed live:
    - voice_id, voice_name: Change TTS voice
    - enable_review_pass: Toggle second-pass review
    - translation_model: Switch between gpt-4o and gpt-4o-mini
    - main_speaker_name: Change speaker display name
    """
    # Fields safe to change while running
    live_changeable = ["voice_id", "voice_name", "enable_review_pass", "translation_model", "main_speaker_name", "tts_service", "tts_speed"]
    
    changed = []
    for field in live_changeable:
        if field in new_config and new_config[field] != config.get(field):
            old_val = config.get(field)
            config[field] = new_config[field]
            changed.append(f"{field}: {old_val} → {new_config[field]}")
            print(f"⚡ Live config change: {field} = {new_config[field]}")
    
    if changed:
        return {"message": "Live configuration updated", "changes": changed}
    return {"message": "No changes made"}

@app.get("/api/producer/status")
async def get_producer_status():
    """Get producer status"""
    return {
        "is_running": producer.is_running,
        "connection_healthy": producer.connection_healthy,
        "is_reconnecting": producer.is_reconnecting,
        "buffer_pending": producer.translation_buffer.pending_count() if producer.translation_buffer else 0,
        "session_id": active_session.id if active_session else None,
        "listener_count": len(active_session.listener_ws_set) if active_session else 0,
    }

@app.post("/api/producer/start")
async def api_start_producer():
    """Start the producer"""
    global active_session
    
    if producer.is_running:
        return {"message": "Producer already running"}
    
    # Create session if not exists
    if not active_session:
        session_id = config["session_id"].upper()
        active_session = Session(
            id=session_id,
            source_lang=config["source_lang"],
            target_lang=config["target_lang"],
            source_lang_name=config["source_lang_name"],
            target_lang_name=config["target_lang_name"],
            voice_id=config["voice_id"]
        )
    
    # Store event loop
    producer.loop = asyncio.get_running_loop()
    
    # Start producer
    start_producer()
    
    # Start buffer timeout checker
    producer.buffer_timeout_task = asyncio.create_task(buffer_timeout_checker())
    
    return {"message": "Producer started", "session_id": active_session.id}

@app.post("/api/producer/stop")
async def api_stop_producer():
    """Stop the producer"""
    if not producer.is_running:
        return {"message": "Producer not running"}
    
    # Cancel buffer timeout task
    if producer.buffer_timeout_task:
        producer.buffer_timeout_task.cancel()
        try:
            await producer.buffer_timeout_task
        except asyncio.CancelledError:
            pass
    
    stop_producer()
    
    return {"message": "Producer stopped"}

@app.get("/api/languages")
async def get_languages():
    """Get available languages"""
    return {"languages": [{"code": c, "name": n} for c, n, _ in LANGUAGES]}

@app.get("/api/voices")
async def get_voices():
    """Get available voices"""
    return {"voices": [{"id": vid, "name": name, "description": desc} for vid, name, desc in VOICES]}

@app.get("/api/prompts")
async def get_prompts():
    """Get available translation prompts"""
    return {
        "prompts": [
            {"key": k, "name": v.get("name", k), "description": v.get("description", "")}
            for k, v in TRANSLATION_PROMPTS.items()
        ]
    }

@app.get("/api/microphones")
async def get_microphones():
    """Get available microphones"""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels', 0) > 0:
            devices.append({
                "index": i,
                "name": info.get('name', f'Device {i}')
            })
    
    p.terminate()
    return {"microphones": devices}

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
                except Exception:
                    break  # Connection lost
    
    except WebSocketDisconnect:
        pass
    
    finally:
        active_session.listener_ws_set.discard(websocket)
        print(f"Listener disconnected ({len(active_session.listener_ws_set)} remaining)")


@app.websocket("/ws/personal")
async def personal_translation_ws(websocket: WebSocket):
    """WebSocket endpoint for Personal Mode translation.
    
    Flow:
    1. Receive config with API keys, source/target lang
    2. Receive audio chunks from browser
    3. Send to Deepgram for transcription
    4. Translate via OpenAI
    5. Generate TTS
    6. Send audio back to browser
    """
    await websocket.accept()
    print("Personal Mode: Client connected")
    
    dg_connection = None
    user_config = {}
    
    try:
        # Wait for config message first
        config_msg = await websocket.receive_json()
        if config_msg.get("type") != "config":
            await websocket.send_json({"type": "error", "message": "Expected config message first"})
            return
        
        user_config = config_msg
        user_keys = config_msg.get("api_keys", {})
        user_context = config_msg.get("context", "")  # User-provided translation context
        
        # Initialize API clients with user's keys
        init_api_clients(user_keys)
        
        # Get language codes and names
        source_lang_code = config_msg.get("source_lang", "tr")
        target_lang_code = config_msg.get("target_lang", "en")
        source_lang_name = next((name for code, name, dg in LANGUAGES if code == source_lang_code), "Turkish")
        target_lang_name = next((name for code, name, dg in LANGUAGES if code == target_lang_code), "English")
        dg_lang_code = next((dg for code, name, dg in LANGUAGES if code == source_lang_code), "en")
        
        # Update config with user preferences including language names
        config.update({
            "source_lang": source_lang_code,
            "target_lang": target_lang_code,
            "source_lang_name": source_lang_name,
            "target_lang_name": target_lang_name,
            "tts_service": config_msg.get("tts_service", "elevenlabs"),
            "user_context": user_context,  # Store user context for translation
        })
        
        if user_context:
            print(f"Personal Mode: Using custom context ({len(user_context)} chars)")
        
        # Initialize Deepgram connection
        dg_connection = deepgram_client.listen.asyncwebsocket.v("1")
        
        # Buffer for accumulating transcripts - SAME SETTINGS AS GUI MODE
        # timeout_seconds=15.0 for batching, batch_threshold=10 words, min_words=3
        transcript_buffer = TranslationBuffer(timeout_seconds=15.0, batch_threshold=10)
        current_speaker = 0
        
        # Deepgram handlers - matching GUI mode logic
        async def on_transcript(self, result, **kwargs):
            nonlocal current_speaker
            
            transcript = result.channel.alternatives[0].transcript
            is_final = result.is_final
            
            if not transcript:
                return
            
            # Check audio confidence and warn if low
            confidence = result.channel.alternatives[0].confidence
            if confidence is not None and confidence < 0.7 and is_final:
                await websocket.send_json({
                    "type": "warning",
                    "message": "Low audio quality detected. Try moving closer to the speaker or reducing background noise.",
                    "confidence": round(confidence, 2)
                })
                print(f"  ⚠️ Low confidence ({confidence:.0%}): '{transcript[:40]}...'")
            
            # Extract speaker ID if diarization is enabled
            words = result.channel.alternatives[0].words
            speaker_id = current_speaker
            if words and len(words) > 0:
                first_word = words[0]
                if hasattr(first_word, 'speaker') and first_word.speaker is not None:
                    speaker_id = min(first_word.speaker, 2)
            
            if is_final:
                # Check if speaker changed - flush buffer first
                if speaker_id != current_speaker and transcript_buffer.has_pending():
                    content, old_speaker, history = transcript_buffer.flush()
                    if content:
                        await process_and_send(content, old_speaker, history)
                
                current_speaker = speaker_id
                
                # Add to buffer
                transcript_buffer.add_segment(transcript, speaker_id)
                
                # Check if we should flush
                if transcript_buffer.should_flush():
                    content, flush_speaker, history = transcript_buffer.flush()
                    if content:
                        await process_and_send(content, flush_speaker, history)
                else:
                    # Show buffering status on webpage
                    buffer_content = transcript_buffer.get_buffer_content()
                    word_count = transcript_buffer.get_word_count()
                    time_elapsed = transcript_buffer.get_time_elapsed()
                    
                    await websocket.send_json({
                        "type": "transcript", 
                        "text": buffer_content,
                        "buffering": True,
                        "word_count": word_count,
                        "time_elapsed": round(time_elapsed, 1)
                    })
                    
                    # Also show in console like GUI mode
                    preview = buffer_content[:60] + "..." if len(buffer_content) > 60 else buffer_content
                    print(f"  [Personal] ({word_count}w, {time_elapsed:.0f}s/{transcript_buffer.timeout_seconds:.0f}s) {preview}", end="\r")
            else:
                # Interim results - show live typing
                buffer_content = transcript_buffer.get_buffer_content()
                current = buffer_content + " " + transcript if buffer_content else transcript
                await websocket.send_json({
                    "type": "transcript",
                    "text": current,
                    "interim": True
                })
        
        async def process_and_send(text: str, speaker_id: int, history: list):
            """Process translation and send to client - matches GUI mode logic"""
            # Filter out short fragments
            words = text.strip().split()
            if len(words) < MIN_WORDS_TO_PROCESS:
                print(f"  (skipped short fragment: '{text}')")
                return
            
            # Echo prevention: Skip if text appears to already be in target language
            target_lang = config.get("target_lang", "en")
            if await is_likely_target_language(text, target_lang):
                print(f"  🔇 (skipped - detected target language: '{text[:50]}...')")
                return
            
            speaker_name = get_speaker_name(speaker_id)
            source_lang_name = config.get("source_lang_name", "Turkish")
            target_lang_name = config.get("target_lang_name", "English")
            
            # Clear buffering line and show formatted output like GUI mode
            print(" " * 120, end="\r")
            print(f"\n{'─'*60}")
            print(f"🎤 [Personal/{speaker_name}] [{source_lang_name}]")
            print(f"   {text}")
            
            try:
                # Translate with context
                translation = await translate_text(text, history=history)
                print(f"🔊 [Personal/{speaker_name}] [{target_lang_name}]")
                print(f"   {translation}")
                print(f"{'─'*60}")
                
                # Send translation to webpage
                await websocket.send_json({
                    "type": "translation", 
                    "text": translation,
                    "original": text,
                    "speaker": speaker_name
                })
                
                # Generate TTS
                audio_data = await generate_speech(translation, speaker_id)
                if audio_data:
                    audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                    await websocket.send_json({
                        "type": "audio", 
                        "audio": audio_b64,
                        "speaker": speaker_name
                    })
                    
            except Exception as e:
                print(f"Personal Mode error: {e}")
                await websocket.send_json({"type": "error", "message": str(e)})
        
        # Set handlers
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        
        # Start Deepgram connection
        options = LiveOptions(
            model="nova-3",
            language=dg_lang_code,
            smart_format=True,
            punctuate=True,
            diarize=False,  # No diarization for personal mode (single user)
            interim_results=True,  # Show live typing
            endpointing=300,
        )
        
        connected = await dg_connection.start(options)
        if not connected:
            await websocket.send_json({"type": "error", "message": "Failed to connect to Deepgram"})
            return
        
        print(f"Personal Mode: Deepgram connected ({source_lang_code} -> {config['target_lang']})")
        
        # Receive audio chunks
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
            
            if "bytes" in message:
                # Send audio to Deepgram
                await dg_connection.send(message["bytes"])
            elif "text" in message:
                # Handle text commands (ping, config updates)
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    pass
    
    except WebSocketDisconnect:
        print("Personal Mode: Client disconnected")
    except Exception as e:
        print(f"Personal Mode error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if dg_connection:
            try:
                await dg_connection.finish()
            except Exception:
                pass
        print("Personal Mode: Connection closed")


@app.websocket("/ws/host/{session_id}")
async def host_broadcast_ws(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for Host Mode broadcasting from browser.
    
    Similar to Personal Mode but also broadcasts to listeners.
    """
    global active_session
    
    await websocket.accept()
    session_id = session_id.upper()
    print(f"Host Mode: Client connected for session {session_id}")
    
    dg_connection = None
    
    try:
        # Wait for config message
        config_msg = await websocket.receive_json()
        if config_msg.get("type") != "config":
            await websocket.send_json({"type": "error", "message": "Expected config message first"})
            return
        
        user_keys = config_msg.get("api_keys", {})
        user_context = config_msg.get("context", "")  # User-provided translation context
        
        # Initialize API clients
        init_api_clients(user_keys)
        
        # Get language codes and names
        source_lang_code = config_msg.get("source_lang", "tr")
        target_lang_code = config_msg.get("target_lang", "en")
        source_lang_name = next((name for code, name, dg in LANGUAGES if code == source_lang_code), "Turkish")
        target_lang_name = next((name for code, name, dg in LANGUAGES if code == target_lang_code), "English")
        dg_lang_code = next((dg for code, name, dg in LANGUAGES if code == source_lang_code), "en")
        
        # Update config with language names
        config.update({
            "source_lang": source_lang_code,
            "target_lang": target_lang_code,
            "source_lang_name": source_lang_name,
            "target_lang_name": target_lang_name,
            "tts_service": config_msg.get("tts_service", "elevenlabs"),
            "session_id": session_id,
            "user_context": user_context,  # Store user context for translation
        })
        
        if user_context:
            print(f"Host Mode: Using custom context ({len(user_context)} chars)")
        
        # Create session if needed
        if not active_session or active_session.id != session_id:
            active_session = Session(
                id=session_id,
                source_lang=config["source_lang"],
                target_lang=config["target_lang"],
                is_live=True
            )
        else:
            active_session.is_live = True
        
        # Initialize translation buffer
        transcript_buffer = TranslationBuffer(timeout_seconds=8.0, batch_threshold=10)
        
        # Initialize Deepgram
        dg_connection = deepgram_client.listen.asyncwebsocket.v("1")
        
        async def on_transcript(self, result, **kwargs):
            transcript = result.channel.alternatives[0].transcript
            if not transcript:
                return
            
            speaker_id = getattr(result.channel.alternatives[0], "speaker", 0) or 0
            speaker_name = get_speaker_name(speaker_id)
            
            # Add to buffer first
            transcript_buffer.add_segment(transcript, speaker_id)
            
            # Show the current buffer content (what will be translated)
            buffer_content = transcript_buffer.get_buffer_content()
            if buffer_content:
                await websocket.send_json({
                    "type": "transcript", 
                    "text": buffer_content,
                    "speaker": speaker_name
                })
            
            # Check if ready
            if transcript_buffer.should_flush():
                content, spk_id, history = transcript_buffer.flush()
                if content:
                    try:
                        spk_name = get_speaker_name(spk_id)
                        
                        # Translate
                        translation = await translate_text(content, history)
                        
                        # Send to host
                        await websocket.send_json({
                            "type": "translation",
                            "text": translation,
                            "speaker": spk_name
                        })
                        
                        # Generate TTS
                        audio_data = await generate_speech(translation, spk_id)
                        
                        if audio_data:
                            audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                            
                            # Broadcast to all listeners (use 'data' key for consistency)
                            await broadcast_to_listeners({
                                "type": "audio",
                                "data": audio_b64,
                                "text": translation,
                                "original": content,
                                "speaker_id": spk_id,
                                "speaker_name": spk_name
                            })
                            
                    except Exception as e:
                        print(f"Host Mode translation error: {e}")
        
        dg_connection.on(LiveTranscriptionEvents.Transcript, on_transcript)
        
        options = LiveOptions(
            model="nova-3",
            language=dg_lang_code,
            smart_format=True,
            punctuate=True,
            diarize=True,
            diarize_version="3",
            interim_results=False,
            endpointing=300,
        )
        
        connected = await dg_connection.start(options)
        if not connected:
            await websocket.send_json({"type": "error", "message": "Failed to connect to Deepgram"})
            return
        
        print(f"Host Mode: Broadcasting session {session_id}")
        
        # Main loop
        while True:
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                break
            
            if "bytes" in message:
                await dg_connection.send(message["bytes"])
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "listener_count": len(active_session.listener_ws_set) if active_session else 0
                        })
                except Exception:
                    pass
    
    except WebSocketDisconnect:
        print(f"Host Mode: Client disconnected from session {session_id}")
    except Exception as e:
        print(f"Host Mode error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if dg_connection:
            try:
                await dg_connection.finish()
            except Exception:
                pass
        if active_session:
            active_session.is_live = False
        print(f"Host Mode: Session {session_id} ended")

# ============== Startup / Shutdown ==============

@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    global active_session
    
    check_api_keys()
    
    # Use custom session ID from config
    session_id = config.get("session_id", "LECTURE").upper()
    active_session = Session(
        id=session_id,
        source_lang=config.get("source_lang", "tr"),
        target_lang=config.get("target_lang", "en"),
        source_lang_name=config.get("source_lang_name", "Turkish"),
        target_lang_name=config.get("target_lang_name", "English"),
        voice_id=config.get("voice_id", "29vD33N1CtxCmqQRPOHJ")
    )
    
    # Store event loop for background thread
    producer.loop = asyncio.get_running_loop()
    
    # Only auto-start producer if NOT in web-only mode
    # Web-only mode uses browser mic via WebSocket, not desktop mic
    if config.get("auto_start_producer", False):
        start_producer()
        # Start buffer timeout checker background task
        producer.buffer_timeout_task = asyncio.create_task(buffer_timeout_checker())
        
        # Print listener link
        print(f"\n{'='*50}")
        print(f"Listener link: http://localhost:8000/listen/{session_id}")
        print("Share this link with your audience!")
        print(f"{'='*50}")
    else:
        # Web-only mode
        print(f"\n{'='*50}")
        print("🌐 Web App Mode - Ready!")
        print(f"   Landing:  http://localhost:8000/")
        print(f"   Host:     http://localhost:8000/host")
        print(f"   Personal: http://localhost:8000/personal")
        print(f"   Settings: http://localhost:8000/settings")
        print(f"{'='*50}\n")

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
    
    # Translation prompt selection
    print("\n" + "-" * 50)
    print("Translation Prompt Profile")
    prompt_keys = list(TRANSLATION_PROMPTS.keys())
    prompt_options = [f"{TRANSLATION_PROMPTS[k]['name']} - {TRANSLATION_PROMPTS[k].get('description', '')[:40]}" for k in prompt_keys]
    print_menu("", prompt_options)
    
    # Find default index (risale_i_nur if available, else default)
    default_prompt_idx = next((i+1 for i, k in enumerate(prompt_keys) if k == "risale_i_nur"), 1)
    prompt_choice = get_choice("Enter number", len(prompt_keys), default_prompt_idx)
    config["prompt_key"] = prompt_keys[prompt_choice - 1]
    print(f"  ✓ Prompt: {TRANSLATION_PROMPTS[config['prompt_key']]['name']}")
    
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
    print(f"  Prompt:       {TRANSLATION_PROMPTS[config['prompt_key']]['name']}")
    print(f"  Listener URL: http://localhost:8000/listen/{config['session_id']}")
    print("=" * 50)
    
    input("\nPress Enter to start the server...")

# ============== Main ==============

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser(description="Realtime Translator Server")
    parser.add_argument("--no-setup", action="store_true", 
                        help="Skip interactive setup (for GUI mode)")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    print("\n🎙️  REALTIME TRANSLATOR")
    print("   Speak → Transcribe → Translate → TTS → Broadcast")
    
    if not args.no_setup:
        # Run interactive setup for console mode
        interactive_setup()
    else:
        print("\n   Running in headless mode (configuration via API/GUI)")
    
    print("\nStarting server...")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_ping_interval=30,
        ws_ping_timeout=60
    )

