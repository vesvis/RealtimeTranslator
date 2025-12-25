# 🎙️ Realtime Translator

Real-time speech-to-speech translation system that captures audio, transcribes, translates, and broadcasts synthesized speech to listeners via web interface.

**Speak → Transcribe → Translate → TTS → Broadcast**

## Features

- 🎤 **Real-time transcription** via Deepgram with speaker diarization (up to 3 speakers)
- 🌍 **AI translation** via OpenAI GPT-4o-mini with context-aware prompting
- 🔊 **Natural TTS** via ElevenLabs with SSML pauses for natural speech
- 📡 **Live broadcast** to unlimited listeners via WebSocket
- 🎧 **Web listener interface** with audio queue buffering for smooth playback
- 💾 **Local recording** of translated audio and transcripts
- ⚙️ **Configurable** prompts, voices, and settings

## Requirements

- Python 3.10+
- API Keys:
  - [Deepgram](https://deepgram.com/) - Speech-to-text
  - [OpenAI](https://openai.com/) - Translation
  - [ElevenLabs](https://elevenlabs.io/) - Text-to-speech

## Installation

```bash
# Clone the repository
git clone https://github.com/vesvis/RealtimeTranslator.git
cd RealtimeTranslator

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
cp .env.example .env
# Edit .env and add your API keys
```

## Configuration

### Environment Variables (.env)

```env
DEEPGRAM_API_KEY=your_deepgram_key
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### Custom Voices (optional)

Create `custom_voices.json` to add your own ElevenLabs voices:

```json
{
  "custom_voices": [
    {
      "id": "your_voice_id",
      "name": "Voice Name",
      "description": "Description"
    }
  ]
}
```

### Custom Translation Prompts (optional)

Create `prompts.json` for specialized translation prompts:

```json
{
  "my_custom_prompt": {
    "name": "My Custom Prompt",
    "description": "Description of the prompt",
    "prompt": "You are a translator... [your full prompt here]"
  }
}
```

## Usage

### Start the Server

```bash
python server.py
```

You'll be prompted to configure:
1. Source language (what you speak)
2. Target language (translation output)
3. TTS voice
4. Microphone
5. Session ID (for persistent listener URLs)
6. Main speaker name
7. Translation prompt profile

### Listener Access

Share the listener URL with your audience:
```
http://your-server:8000/listen/YOUR_SESSION_ID
```

For remote access, use a tunnel like [ngrok](https://ngrok.com/):
```bash
ngrok http 8000
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Microphone │────▶│   Deepgram   │────▶│   OpenAI     │
│   (Audio)   │     │  (Transcribe)│     │  (Translate) │
└─────────────┘     └──────────────┘     └──────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Listeners  │◀────│  WebSocket   │◀────│  ElevenLabs  │
│   (Web)     │     │  (Broadcast) │     │    (TTS)     │
└─────────────┘     └──────────────┘     └──────────────┘
```

## Key Features Explained

### Buffered Translation (15s)
Text is accumulated for 15 seconds (or until sentence completion with 15+ words) before translation. This allows the AI to see full sentences for better grammar, especially for SOV→SVO language pairs (e.g., Turkish→English).

### SSML Pauses
The translation prompt instructs the AI to use line breaks and `||` markers for natural pauses in TTS output:
- Line breaks → 0.5s pause
- `||` markers → 1.5s pause (major topic shifts)

### Client-Side Audio Buffering
Listeners buffer 2 audio segments before playback starts, ensuring smooth continuous audio without gaps.

### Speaker Diarization
Up to 3 speakers are detected and assigned different TTS voices automatically.

## Files

| File | Description |
|------|-------------|
| `server.py` | Main server with all logic |
| `static/` | Web interface files |
| `recordings/` | Saved audio and transcripts |
| `.env` | API keys (not tracked) |
| `custom_voices.json` | Custom voices (not tracked) |
| `prompts.json` | Custom prompts (not tracked) |

## License

MIT

## Contributing

Pull requests welcome! Please open an issue first to discuss major changes.
