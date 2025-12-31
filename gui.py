"""
Realtime Translator GUI

PyQt6 desktop application for configuring and controlling the translation server.
"""

import sys
import os
import json
import asyncio
import threading
import requests
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTextEdit, QFrame,
    QCheckBox, QGroupBox, QGridLayout, QSplitter, QMessageBox,
    QScrollArea, QProgressBar, QSlider
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon


# Import configuration from server
sys.path.insert(0, os.path.dirname(__file__))
try:
    import pyaudio
except ImportError:
    print("PyAudio not installed. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyaudio"])
    import pyaudio


# ============== Configuration Data ==============

import server

# Import configuration from server to keep distinct voice lists in sync
from server import (
    LANGUAGES, 
    BUILTIN_VOICES, 
    OPENAI_VOICES, 
    FISH_AUDIO_VOICES,
    DEEPGRAM_VOICES,
    VOICES,
    load_custom_voices
)
from config_manager import get_config_manager

TTS_SERVICES = [
    ("elevenlabs", "ElevenLabs"),
    ("openai", "OpenAI"),
    ("fish_audio", "Fish Audio"),
    ("deepgram", "Deepgram"),
]

DEFAULT_PROMPT = {
    "name": "Default Translator",
    "description": "General-purpose translation with natural speech formatting",
}

def load_prompts():
    """Load translation prompts from prompts.json."""
    prompts = {"default": DEFAULT_PROMPT}
    prompts_file = os.path.join(os.path.dirname(__file__), "prompts.json")
    try:
        with open(prompts_file, "r", encoding="utf-8") as f:
            custom = json.load(f)
            prompts.update(custom)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return prompts

PROMPTS = load_prompts()

def get_input_devices():
    """Get list of available audio input devices."""
    p = pyaudio.PyAudio()
    devices = []
    
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info.get('maxInputChannels', 0) > 0:
            name = info.get('name', f'Device {i}')
            devices.append((i, name))
    
    p.terminate()
    return devices


# ============== Styles ==============

STYLESHEET = """
/* ===== Main Window ===== */
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #0a0a12, stop:0.5 #0f0f1a, stop:1 #12121f);
}

QWidget {
    color: #e0e0e0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 13px;
}

/* ===== Group Boxes (FIXED) ===== */
QGroupBox {
    background-color: rgba(25, 25, 45, 0.85);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 16px;
    margin-top: 30px; /* Space for the title outside */
    padding-top: 35px; /* CRITICAL FIX: Pushes content down so title doesn't overlap */
    padding-bottom: 20px;
    padding-left: 15px;
    padding-right: 15px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 8px 20px;
    left: 15px;
    top: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #6366f1, stop:1 #8b5cf6);
    border-radius: 12px;
    color: white;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 0.5px;
}

/* ===== Labels ===== */
QLabel {
    color: #9ca3af;
    padding: 2px 0;
    font-size: 13px;
    font-weight: 500;
}

QLabel#title {
    color: white;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
}

QLabel#sectionTitle {
    color: #e5e7eb;
    font-size: 14px;
    font-weight: 600;
    padding-bottom: 6px;
}

/* ===== Combo Boxes (FIXED) ===== */
QComboBox {
    background-color: rgba(35, 35, 55, 0.95);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 10px;
    padding: 8px 12px;
    min-height: 25px; /* Ensure height */
    color: #f3f4f6;
    font-weight: 500;
}

QComboBox:hover {
    border-color: rgba(99, 102, 241, 0.7);
    background-color: rgba(40, 40, 65, 0.95);
}

QComboBox:focus {
    border-color: #6366f1;
}

QComboBox::drop-down {
    border: none;
    padding-right: 12px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #6366f1;
    margin-right: 10px;
}

QComboBox QAbstractItemView {
    background-color: #1e1e32;
    border: 1px solid rgba(99, 102, 241, 0.5);
    border-radius: 10px;
    selection-background-color: #6366f1;
    padding: 6px;
    outline: none;
}

/* ===== Line Edits ===== */
QLineEdit {
    background-color: rgba(35, 35, 55, 0.95);
    border: 1px solid rgba(99, 102, 241, 0.35);
    border-radius: 10px;
    padding: 8px 12px;
    color: #f3f4f6;
    font-weight: 500;
    selection-background-color: #6366f1;
}

QLineEdit:hover {
    border-color: rgba(99, 102, 241, 0.5);
}

QLineEdit:focus {
    border-color: #6366f1;
    background-color: rgba(40, 40, 65, 0.95);
}

QLineEdit:read-only {
    background-color: rgba(25, 25, 40, 0.9);
    color: #d1d5db;
}

/* ===== Checkboxes ===== */
QCheckBox {
    spacing: 10px;
    color: #d1d5db;
}

QCheckBox::indicator {
    width: 20px;
    height: 20px;
    border-radius: 6px;
    border: 2px solid rgba(99, 102, 241, 0.4);
    background-color: rgba(35, 35, 55, 0.95);
}

QCheckBox::indicator:hover {
    border-color: rgba(99, 102, 241, 0.7);
}

QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #6366f1, stop:1 #8b5cf6);
    border-color: #6366f1;
}

/* ===== Primary Buttons ===== */
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #6366f1, stop:1 #8b5cf6);
    border: none;
    border-radius: 12px;
    padding: 14px 28px;
    color: white;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.3px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #7c7ff7, stop:1 #a78bfa);
}

QPushButton:pressed {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #5558e3, stop:1 #7c3aed);
}

QPushButton:disabled {
    background: rgba(55, 55, 75, 0.8);
    color: #6b7280;
}

/* ===== Stop Button (Red Gradient) ===== */
QPushButton#stopButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #ef4444, stop:1 #dc2626);
}

QPushButton#stopButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
        stop:0 #f87171, stop:1 #ef4444);
}

/* ===== Secondary/Copy Buttons ===== */
QPushButton#copyButton {
    background: rgba(99, 102, 241, 0.2);
    border: 1px solid rgba(99, 102, 241, 0.4);
    padding: 10px 18px;
    font-size: 12px;
    border-radius: 10px;
}

QPushButton#copyButton:hover {
    background: rgba(99, 102, 241, 0.35);
    border-color: rgba(99, 102, 241, 0.6);
}

QPushButton#clearButton {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 8px 14px;
    font-size: 11px;
    color: #fca5a5;
}

QPushButton#clearButton:hover {
    background: rgba(239, 68, 68, 0.25);
    border-color: rgba(239, 68, 68, 0.5);
}

/* ===== Text Edit (Log Area) ===== */
QTextEdit {
    background-color: rgba(15, 15, 25, 0.95);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 14px;
    color: #9ca3af;
    font-family: 'JetBrains Mono', 'Consolas', 'Monaco', monospace;
    font-size: 12px;
    line-height: 1.5;
}

/* ===== Status Frame & Specific Labels (RESTORED) ===== */
QFrame#statusFrame {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(30, 30, 55, 0.9), stop:1 rgba(25, 25, 45, 0.9));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 20px;
}

QLabel#statusLive {
    color: #22c55e;
    font-weight: 700;
    font-size: 18px;
    letter-spacing: 0.5px;
}

QLabel#statusOffline {
    color: #ef4444;
    font-weight: 700;
    font-size: 16px;
}

QLabel#statusConnecting {
    color: #f59e0b;
    font-weight: 700;
    font-size: 16px;
}

QLabel#listenerCount {
    color: #a5b4fc;
    font-size: 48px;
    font-weight: 800;
    letter-spacing: -2px;
}

QLabel#listenerLabel {
    color: #6b7280;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.5px;
}

QLabel#sessionId {
    color: #8b5cf6;
    font-size: 14px;
    font-weight: 600;
}

QLabel#durationLabel {
    color: #22c55e;
    font-size: 16px;
    font-weight: 600;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}

/* ===== Progress Bar (RESTORED) ===== */
QProgressBar {
    background-color: rgba(35, 35, 55, 0.8);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 6px;
    height: 8px;
    text-align: center;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #22c55e, stop:0.7 #84cc16, stop:1 #eab308);
    border-radius: 5px;
}

/* ===== Scroll Areas (FIXED & RESTORED) ===== */
QScrollArea {
    border: none;
    background: transparent; /* Needs to be transparent for new layout */
}

/* Restore the container background for the config panel specifically */
QWidget#configContainer {
    background: transparent;
}

QScrollBar:vertical {
    background-color: rgba(25, 25, 40, 0.5);
    width: 10px;
    border-radius: 5px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(99, 102, 241, 0.5), stop:1 rgba(139, 92, 246, 0.5));
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 rgba(99, 102, 241, 0.7), stop:1 rgba(139, 92, 246, 0.7));
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* ===== Divider Lines ===== */
QFrame#divider {
    background-color: rgba(99, 102, 241, 0.2);
    max-height: 1px;
    margin: 8px 0;
}

/* ===== Stats Group ===== */
QGroupBox#statsGroup {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(30, 30, 55, 0.85), stop:1 rgba(25, 25, 45, 0.85));
}

QLabel#statValue {
    color: #e5e7eb;
    font-weight: 600;
    font-size: 13px;
}

/* ===== Tooltips ===== */
QToolTip {
    background-color: #1e1e32;
    border: 1px solid rgba(99, 102, 241, 0.5);
    border-radius: 8px;
    padding: 8px 12px;
    color: #e5e7eb;
    font-size: 12px;
}
"""


# ============== Server Thread ==============

class ServerThread(QThread):
    """Thread to run the uvicorn server."""
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop_event = threading.Event()
        self.server = None
        
    def run(self):
        """Run the server in this thread."""
        try:
            import uvicorn
            import server
            
            # Update server config with GUI config BEFORE starting
            server.config.update(self.config)
            
            # Initialize API clients with user-provided keys
            if "api_keys" in self.config:
                server.init_api_clients(self.config["api_keys"])
            
            # Log config for debugging
            self.log_signal.emit(f"Session: {self.config['session_id']}")
            self.log_signal.emit(f"Speaker: {self.config['main_speaker_name']}")
            self.log_signal.emit(f"Prompt: {self.config['prompt_key']}")
            self.log_signal.emit(f"Mic: {self.config['mic_device_name']} (index: {self.config['mic_device_index']})")
            
            # Configure uvicorn - startup_event in server.py will handle producer init
            uvicorn_config = uvicorn.Config(
                server.app,
                host="0.0.0.0",
                port=8000,
                ws_ping_interval=30,
                ws_ping_timeout=60,
                log_level="info"  # Show more logs for debugging
            )
            self.server = uvicorn.Server(uvicorn_config)
            
            # Run server (blocking) - startup_event creates session and starts producer
            self.server.run()
            
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{str(e)}\n{traceback.format_exc()}")
    
    def stop(self):
        """Signal the server to stop."""
        if self.server:
            self.server.should_exit = True


# ============== Status Monitor ==============

class StatusMonitor(QObject):
    """Monitors server status periodically."""
    status_updated = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_status)
        self.running = False
        self.session_id = "LECTURE"  # Default, updated when server starts
        
    def start(self, session_id: str = None):
        if session_id:
            self.session_id = session_id
        self.running = True
        self.timer.start(1000)  # Check every second
        
    def stop(self):
        self.running = False
        self.timer.stop()
        
    def check_status(self):
        if not self.running:
            return
            
        try:
            # Use producer status endpoint - doesn't require session ID
            response = requests.get("http://localhost:8000/api/producer/status", timeout=1)
            if response.status_code == 200:
                data = response.json()
                self.status_updated.emit({
                    "connected": True,
                    "is_live": data.get("is_running", False),
                    "listener_count": data.get("listener_count", 0),
                    "connection_healthy": data.get("connection_healthy", False)
                })
                return
                    
            self.status_updated.emit({"connected": True, "is_live": False, "listener_count": 0})
        except:
            self.status_updated.emit({"connected": False, "is_live": False, "listener_count": 0})


# ============== Main Window ==============

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.server_thread: Optional[ServerThread] = None
        self.status_monitor = StatusMonitor(self)
        self.status_monitor.status_updated.connect(self.on_status_update)
        self.is_running = False
        self.current_session_id = "LECTURE"
        
        # Session timer
        self.session_start_time: Optional[datetime] = None
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self.update_session_duration)
        
        self.init_ui()
        self.setStyleSheet(STYLESHEET)
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("🎙️ Realtime Translator")
        self.setMinimumSize(1100, 750) # Slightly larger minimum
        self.resize(1250, 850)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Left panel - Configuration
        left_panel = self.create_config_panel()
        left_panel.setFixedWidth(360)
        
        # Center panel - Controls
        center_panel = self.create_control_panel()
        
        # Right panel - Status
        right_panel = self.create_status_panel()
        right_panel.setFixedWidth(300)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(center_panel, 1)
        main_layout.addWidget(right_panel)

    def create_config_panel(self) -> QWidget:   
        """Create the configuration panel with a Scroll Area."""
        # 1. Create a Scroll Area wrapper
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # 2. Create a container widget that goes INSIDE the scroll area
        container = QWidget()
        # Using specific object name to apply transparent background via stylesheet
        container.setObjectName("configContainer") 
        
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 10, 30, 20) # Right margin 30px for scrollbar
        layout.setSpacing(20) # Increase spacing between groups
        
        # --- Content Start ---
        
        # Title
        title = QLabel("⚙️ Configuration")
        title.setObjectName("title")
        layout.addWidget(title)
        
        # Language group
        lang_group = QGroupBox("Languages")
        lang_layout = QVBoxLayout(lang_group)
        lang_layout.setSpacing(15) # Spacing inside box
        lang_layout.setContentsMargins(15, 30, 15, 15) # Internal margins
        
        lang_layout.addWidget(QLabel("Source Language:"))
        self.source_lang = QComboBox()
        for code, name, _ in LANGUAGES:
            self.source_lang.addItem(f"{name} ({code})", code)
        idx = next((i for i, (c, _, _) in enumerate(LANGUAGES) if c == "tr"), 0)
        self.source_lang.setCurrentIndex(idx)
        lang_layout.addWidget(self.source_lang)
        
        lang_layout.addWidget(QLabel("Target Language:"))
        self.target_lang = QComboBox()
        for code, name, _ in LANGUAGES:
            self.target_lang.addItem(f"{name} ({code})", code)
        self.target_lang.setCurrentIndex(0)
        lang_layout.addWidget(self.target_lang)
        
        layout.addWidget(lang_group)
        
        # Voice & Audio group
        audio_group = QGroupBox("Voice && Audio")
        audio_layout = QVBoxLayout(audio_group)
        audio_layout.setSpacing(15)
        audio_layout.setContentsMargins(15, 30, 15, 15)
        
        audio_layout.addWidget(QLabel("TTS Service:"))
        self.tts_service = QComboBox()
        for service_id, name in TTS_SERVICES:
            self.tts_service.addItem(name, service_id)
        self.tts_service.currentIndexChanged.connect(self.update_voice_list)
        # Connect to live config change too
        self.tts_service.currentIndexChanged.connect(self.on_live_config_change)
        audio_layout.addWidget(self.tts_service)

        audio_layout.addWidget(QLabel("TTS Voice:"))
        self.voice_combo = QComboBox()
        # Set adjust policy to allow text to fit better
        # self.voice_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
        self.update_voice_list() # Populate with default (ElevenLabs)
        self.voice_combo.currentIndexChanged.connect(self.on_live_config_change)
        audio_layout.addWidget(self.voice_combo)
        
        # Speech Speed dropdown (simple options instead of slider)
        audio_layout.addWidget(QLabel("Speech Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("🐢 Slow (0.8x)", 0.8)
        self.speed_combo.addItem("Normal (1.0x)", 1.0)
        self.speed_combo.addItem("🐇 Fast (1.15x)", 1.15)
        self.speed_combo.setCurrentIndex(1)  # Default to Normal
        self.speed_combo.currentIndexChanged.connect(self.on_live_config_change)
        audio_layout.addWidget(self.speed_combo)
        
        mic_layout = QHBoxLayout()
        mic_layout.addWidget(QLabel("Microphone:"))
        refresh_btn = QPushButton("🔄")
        refresh_btn.setFixedSize(24, 24)
        refresh_btn.clicked.connect(self.refresh_microphones)
        mic_layout.addWidget(refresh_btn)
        mic_layout.addStretch()
        audio_layout.addLayout(mic_layout)
        
        self.mic_combo = QComboBox()
        self.refresh_microphones()
        audio_layout.addWidget(self.mic_combo)
        
        audio_layout.addWidget(QLabel("Audio Level:"))
        self.mic_level = QProgressBar()
        self.mic_level.setRange(0, 100)
        self.mic_level.setValue(0)
        self.mic_level.setTextVisible(False)
        self.mic_level.setFixedHeight(6)
        audio_layout.addWidget(self.mic_level)
        
        layout.addWidget(audio_group)
        
        # Session group
        session_group = QGroupBox("Session Details")
        session_layout = QVBoxLayout(session_group)
        session_layout.setSpacing(15)
        session_layout.setContentsMargins(15, 30, 15, 15)
        
        session_layout.addWidget(QLabel("Session ID:"))
        self.session_id = QLineEdit()
        self.session_id.setText("LECTURE")
        session_layout.addWidget(self.session_id)
        
        session_layout.addWidget(QLabel("Speaker Name:"))
        self.speaker_name = QLineEdit()
        self.speaker_name.setText("Speaker")
        session_layout.addWidget(self.speaker_name)
        
        layout.addWidget(session_group)
        
        # Translation group
        prompt_group = QGroupBox("Translation Engine")
        prompt_layout = QVBoxLayout(prompt_group)
        prompt_layout.setSpacing(15)
        prompt_layout.setContentsMargins(15, 30, 15, 15)
        
        prompt_layout.addWidget(QLabel("Prompt:"))
        self.prompt_combo = QComboBox()
        for key, data in PROMPTS.items():
            name = data.get("name", key)
            self.prompt_combo.addItem(f"{name}", key)
        idx = next((i for i, k in enumerate(PROMPTS.keys()) if k == "risale_i_nur"), 0)
        self.prompt_combo.setCurrentIndex(idx)
        prompt_layout.addWidget(self.prompt_combo)
        
        prompt_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("GPT-4o Mini (Fast)", "gpt-4o-mini")
        self.model_combo.addItem("GPT-4o (Quality)", "gpt-4o")
        self.model_combo.currentIndexChanged.connect(self.on_live_config_change)
        prompt_layout.addWidget(self.model_combo)
        
        self.review_checkbox = QCheckBox("Enable Review Pass")
        self.review_checkbox.setChecked(False)
        self.review_checkbox.stateChanged.connect(self.on_live_config_change)
        prompt_layout.addWidget(self.review_checkbox)
        
        layout.addWidget(prompt_group)
        
        # API Keys group (BYOK - Bring Your Own Keys)
        api_keys_group = QGroupBox("🔑 API Keys (BYOK)")
        api_keys_layout = QVBoxLayout(api_keys_group)
        api_keys_layout.setSpacing(12)
        api_keys_layout.setContentsMargins(15, 30, 15, 15)
        
        # Info label
        info_label = QLabel("Enter your own API keys to use the translator.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #6b7280; font-size: 11px;")
        api_keys_layout.addWidget(info_label)
        
        # Deepgram Key
        api_keys_layout.addWidget(QLabel("Deepgram:"))
        self.deepgram_key_input = QLineEdit()
        self.deepgram_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.deepgram_key_input.setPlaceholderText("Enter Deepgram API key...")
        api_keys_layout.addWidget(self.deepgram_key_input)
        
        # OpenAI Key
        api_keys_layout.addWidget(QLabel("OpenAI:"))
        self.openai_key_input = QLineEdit()
        self.openai_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key_input.setPlaceholderText("Enter OpenAI API key...")
        api_keys_layout.addWidget(self.openai_key_input)
        
        # ElevenLabs Key
        api_keys_layout.addWidget(QLabel("ElevenLabs:"))
        self.elevenlabs_key_input = QLineEdit()
        self.elevenlabs_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.elevenlabs_key_input.setPlaceholderText("Enter ElevenLabs API key...")
        api_keys_layout.addWidget(self.elevenlabs_key_input)
        
        # Fish Audio Key (optional)
        api_keys_layout.addWidget(QLabel("Fish Audio (optional):"))
        self.fish_audio_key_input = QLineEdit()
        self.fish_audio_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.fish_audio_key_input.setPlaceholderText("Enter Fish Audio API key...")
        api_keys_layout.addWidget(self.fish_audio_key_input)
        
        # Remember keys checkbox
        self.remember_keys_checkbox = QCheckBox("Remember my keys")
        self.remember_keys_checkbox.setChecked(True)
        self.remember_keys_checkbox.setToolTip("Keys are stored locally on your computer")
        api_keys_layout.addWidget(self.remember_keys_checkbox)
        
        # Load saved keys
        self._load_saved_api_keys()
        
        layout.addWidget(api_keys_group)
        layout.addStretch()
        
        # 3. Set the container as the scroll widget and return the scroll area
        scroll_area.setWidget(container)
        return scroll_area
        
    def create_control_panel(self) -> QWidget:
        """Create the central control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)
        
        # Title with enhanced styling
        title = QLabel("🎙️ Realtime Translator")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Speak → Transcribe → Translate → TTS → Broadcast")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #6b7280; font-size: 13px; letter-spacing: 0.5px;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Start/Stop button - ENHANCED with larger size and better styling
        self.start_btn = QPushButton("▶  Start Server")
        self.start_btn.setMinimumHeight(70)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.5px;
                border-radius: 16px;
            }
        """)
        self.start_btn.clicked.connect(self.toggle_server)
        layout.addWidget(self.start_btn)
        
        layout.addSpacing(16)
        
        # URL display frame
        url_frame = QFrame()
        url_frame.setObjectName("statusFrame")
        url_layout = QVBoxLayout(url_frame)
        url_layout.setSpacing(12)
        
        url_label = QLabel("📡 Listener URL:")
        url_label.setObjectName("sectionTitle")
        url_layout.addWidget(url_label)
        
        url_row = QHBoxLayout()
        url_row.setSpacing(10)
        self.url_display = QLineEdit()
        self.url_display.setReadOnly(True)
        self.url_display.setText("Start server to generate URL...")
        self.url_display.setStyleSheet("font-size: 12px;")
        url_row.addWidget(self.url_display)
        
        copy_btn = QPushButton("📋 Copy")
        copy_btn.setObjectName("copyButton")
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.clicked.connect(self.copy_url)
        url_row.addWidget(copy_btn)
        
        url_layout.addLayout(url_row)
        
        # ngrok URL optional
        ngrok_label = QLabel("🌐 ngrok URL (optional):")
        url_layout.addWidget(ngrok_label)
        
        self.ngrok_input = QLineEdit()
        self.ngrok_input.setPlaceholderText("Paste ngrok URL for remote access...")
        self.ngrok_input.textChanged.connect(self.update_listener_url)
        url_layout.addWidget(self.ngrok_input)
        
        layout.addWidget(url_frame)
        
        layout.addSpacing(12)
        
        # Log header with clear button
        log_header = QHBoxLayout()
        log_label = QLabel("📝 Server Log:")
        log_label.setObjectName("sectionTitle")
        log_header.addWidget(log_label)
        log_header.addStretch()
        
        clear_log_btn = QPushButton("🗑 Clear")
        clear_log_btn.setObjectName("clearButton")
        clear_log_btn.setFixedWidth(80)
        clear_log_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_log_btn.clicked.connect(self.clear_log)
        log_header.addWidget(clear_log_btn)
        
        layout.addLayout(log_header)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(180)
        self.log_output.setPlaceholderText("Server logs will appear here...")
        layout.addWidget(self.log_output)
        
        layout.addStretch()
        
        return panel
    
    def clear_log(self):
        """Clear the log output."""
        if hasattr(self, 'log_output'):
            self.log_output.clear()
        
        
    def create_status_panel(self) -> QWidget:
        """Create the status panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Title
        title = QLabel("📊 Status")
        title.setObjectName("title")
        layout.addWidget(title)
        
        # Main status frame
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame")
        status_layout = QVBoxLayout(status_frame)
        status_layout.setSpacing(16)
        
        # Server status row
        status_row = QHBoxLayout()
        server_label = QLabel("Server:")
        server_label.setStyleSheet("font-weight: 600; color: #9ca3af;")
        status_row.addWidget(server_label)
        self.status_label = QLabel("● Offline")
        self.status_label.setObjectName("statusOffline")
        status_row.addWidget(self.status_label)
        status_row.addStretch()
        status_layout.addLayout(status_row)
        
        # Listener count - ENHANCED prominent display
        listener_frame = QWidget()
        listener_layout = QVBoxLayout(listener_frame)
        listener_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        listener_layout.setSpacing(4)
        
        self.listener_count = QLabel("0")
        self.listener_count.setObjectName("listenerCount")
        self.listener_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        listener_layout.addWidget(self.listener_count)
        
        listener_label = QLabel("Connected Listeners")
        listener_label.setObjectName("listenerLabel")
        listener_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        listener_layout.addWidget(listener_label)
        
        status_layout.addWidget(listener_frame)
        
        # Session duration timer
        duration_row = QHBoxLayout()
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet("font-weight: 600; color: #9ca3af;")
        duration_row.addWidget(duration_label)
        self.duration_display = QLabel("00:00:00")
        self.duration_display.setObjectName("durationLabel")
        duration_row.addWidget(self.duration_display)
        duration_row.addStretch()
        status_layout.addLayout(duration_row)
        
        # Divider
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFrameShape(QFrame.Shape.HLine)
        status_layout.addWidget(divider)
        
        # Recording status
        rec_row = QHBoxLayout()
        rec_label = QLabel("Recording:")
        rec_label.setStyleSheet("font-weight: 600; color: #9ca3af;")
        rec_row.addWidget(rec_label)
        self.recording_label = QLabel("Not started")
        self.recording_label.setStyleSheet("color: #6b7280;")
        rec_row.addWidget(self.recording_label)
        rec_row.addStretch()
        status_layout.addLayout(rec_row)
        
        # Session info
        session_row = QHBoxLayout()
        session_label = QLabel("Session:")
        session_label.setStyleSheet("font-weight: 600; color: #9ca3af;")
        session_row.addWidget(session_label)
        self.session_label = QLabel("--")
        self.session_label.setObjectName("sessionId")
        session_row.addWidget(self.session_label)
        session_row.addStretch()
        status_layout.addLayout(session_row)
        
        layout.addWidget(status_frame)
        
        # Quick stats - ENHANCED styling
        stats_group = QGroupBox("Translation Stats")
        stats_group.setObjectName("statsGroup")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setSpacing(10)
        
        self.stats_labels = {}
        stat_items = [
            ("Source", "🌍"),
            ("Target", "🎯"),
            ("Voice", "🔊"),
            ("Prompt", "📝")
        ]
        for label, icon in stat_items:
            row = QHBoxLayout()
            name_label = QLabel(f"{icon} {label}:")
            name_label.setStyleSheet("font-weight: 500; color: #9ca3af;")
            row.addWidget(name_label)
            val = QLabel("--")
            val.setObjectName("statValue")
            self.stats_labels[label.lower()] = val
            row.addWidget(val)
            row.addStretch()
            stats_layout.addLayout(row)
        
        layout.addWidget(stats_group)
        
        layout.addStretch()
        
        return panel
        
    def refresh_microphones(self):
        """Refresh the microphone list."""
        self.mic_combo.clear()
        devices = get_input_devices()
        for idx, name in devices:
            self.mic_combo.addItem(name, idx)
        # Only log if log_output is available (after init)
        if devices and hasattr(self, 'log_output'):
            self.log("Microphones refreshed")
            
    def log(self, message: str):
        """Add a message to the log output."""
        if not hasattr(self, 'log_output'):
            return
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        
    def copy_url(self):
        """Copy the listener URL to clipboard."""
        url = self.url_display.text()
        if url and not url.startswith("Start"):
            QApplication.clipboard().setText(url)
            self.log("URL copied to clipboard!")
            
    def update_listener_url(self):
        """Update the listener URL based on ngrok input or localhost."""
        session_id = self.current_session_id
        if not session_id:
            return
            
        ngrok_url = self.ngrok_input.text().strip()
        
        if ngrok_url:
            # Remove trailing slash if present
            ngrok_url = ngrok_url.rstrip('/')
            # Construct full listener URL
            listener_url = f"{ngrok_url}/listen/{session_id}"
        else:
            listener_url = f"http://localhost:8000/listen/{session_id}"
            
        self.url_display.setText(listener_url)
    
    def update_voice_list(self):
        """Update voice options based on selected TTS service."""
        service = self.tts_service.currentData()
        self.voice_combo.blockSignals(True) # Prevent triggering config change during clear
        self.voice_combo.clear()
        
        voices = []
        if service == "elevenlabs":
            voices = VOICES # Start with built-in + loaded custom
        elif service == "openai":
            voices = OPENAI_VOICES
            # Add customs if any? For now just built-in
            custom_voices = load_custom_voices()
            # If we wanted to check if custom voices have a 'service' field we coould
            # For now assume custom_voices are for ElevenLabs or generic
            # voices += custom_voices 
        elif service == "fish_audio":
            voices = FISH_AUDIO_VOICES
            # Add custom voices too?
            # voices += load_custom_voices() 
        elif service == "deepgram":
            voices = DEEPGRAM_VOICES
        
        for vid, name, desc in voices:
            self.voice_combo.addItem(f"{name} - {desc}", vid)
            
        self.voice_combo.blockSignals(False)
        
        # Trigger config update with new voice
        if self.is_running:
            self.on_live_config_change() 
    
    def on_live_config_change(self):
        """Handle live config changes while server is running."""
        if not self.is_running:
            return  # Only send live updates if server is running
        
        # Gather live-changeable config values
        live_config = {
            "voice_id": self.voice_combo.currentData(),
            "voice_name": self.voice_combo.currentText().split(" - ")[0],
            "enable_review_pass": self.review_checkbox.isChecked(),
            "translation_model": self.model_combo.currentData(),
            "tts_service": self.tts_service.currentData(),
            "tts_speed": self.speed_combo.currentData(),  # Get speed directly from dropdown
        }
        
        # Send to server via API
        try:
            response = requests.post(
                "http://localhost:8000/api/config/live",
                json=live_config,
                timeout=2
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("changes"):
                    for change in data["changes"]:
                        self.log(f"⚡ {change}")
        except Exception as e:
            self.log(f"Live config update failed: {e}")
            
    def toggle_server(self):
        """Toggle the server on/off."""
        if self.is_running:
            self.stop_server()
        else:
            self.start_server()
            
    def get_config(self) -> dict:
        """Get current configuration from UI."""
        source_idx = self.source_lang.currentIndex()
        target_idx = self.target_lang.currentIndex()
        
        config = {
            "source_lang": LANGUAGES[source_idx][0],
            "source_lang_name": LANGUAGES[source_idx][1],
            "target_lang": LANGUAGES[target_idx][0],
            "target_lang_name": LANGUAGES[target_idx][1],
            "voice_id": self.voice_combo.currentData(),
            "voice_name": self.voice_combo.currentText().split(" - ")[0],
            "mic_device_index": self.mic_combo.currentData(),
            "mic_device_name": self.mic_combo.currentText(),
            "session_id": self.session_id.text().upper().strip() or "LECTURE",
            "main_speaker_name": self.speaker_name.text().strip() or "Speaker",
            "prompt_key": self.prompt_combo.currentData(),
            "enable_review_pass": self.review_checkbox.isChecked(),
            "translation_model": self.model_combo.currentData(),
            "tts_service": self.tts_service.currentData(),
            "tts_speed": self.speed_combo.currentData(),  # Get speed directly from dropdown
        }
        
        # Add API keys from inputs
        config["api_keys"] = {
            "deepgram": self.deepgram_key_input.text().strip(),
            "openai": self.openai_key_input.text().strip(),
            "elevenlabs": self.elevenlabs_key_input.text().strip(),
            "fish_audio": self.fish_audio_key_input.text().strip(),
        }
        
        # Save keys if "Remember" is checked
        if self.remember_keys_checkbox.isChecked():
            self._save_api_keys()
        
        return config
    
    def _load_saved_api_keys(self):
        """Load API keys from persistent storage."""
        try:
            config_manager = get_config_manager()
            keys = config_manager.get_all_api_keys()
            
            self.deepgram_key_input.setText(keys.get("deepgram", ""))
            self.openai_key_input.setText(keys.get("openai", ""))
            self.elevenlabs_key_input.setText(keys.get("elevenlabs", ""))
            self.fish_audio_key_input.setText(keys.get("fish_audio", ""))
            
            # Set remember checkbox state
            self.remember_keys_checkbox.setChecked(config_manager.should_remember_keys())
        except Exception as e:
            print(f"Warning: Could not load saved API keys: {e}")
    
    def _save_api_keys(self):
        """Save API keys to persistent storage."""
        try:
            config_manager = get_config_manager()
            keys = {
                "deepgram": self.deepgram_key_input.text().strip(),
                "openai": self.openai_key_input.text().strip(),
                "elevenlabs": self.elevenlabs_key_input.text().strip(),
                "fish_audio": self.fish_audio_key_input.text().strip(),
            }
            config_manager.set_all_api_keys(keys, save_to_disk=True)
            config_manager.set_remember_keys(self.remember_keys_checkbox.isChecked())
        except Exception as e:
            print(f"Warning: Could not save API keys: {e}")
        
    def start_server(self):
        """Start the translation server."""
        config = self.get_config()
        self.current_session_id = config["session_id"]
        
        # Update UI - Enhanced stop button styling
        self.start_btn.setText("⏹  Stop Server")
        self.start_btn.setObjectName("stopButton")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #ef4444, stop:1 #dc2626);
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.5px;
                border-radius: 16px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                    stop:0 #f87171, stop:1 #ef4444);
            }
        """)
        
        # Disable config while running
        self.set_config_enabled(False)
        
        # Update URL (respects ngrok if entered)
        self.update_listener_url()
        
        # Update stats display
        self.stats_labels["source"].setText(config["source_lang_name"])
        self.stats_labels["target"].setText(config["target_lang_name"])
        self.stats_labels["voice"].setText(config["voice_name"])
        prompt_name = PROMPTS.get(config["prompt_key"], {}).get("name", config["prompt_key"])
        self.stats_labels["prompt"].setText(prompt_name[:15] + "..." if len(prompt_name) > 15 else prompt_name)
        self.session_label.setText(config["session_id"])
        
        self.log(f"Configuration: {config['source_lang_name']} → {config['target_lang_name']}")
        self.log(f"Voice: {config['voice_name']}, Mic: {config['mic_device_name']}")
        review_status = "enabled" if config['enable_review_pass'] else "disabled"
        self.log(f"Review pass: {review_status}")
        self.log("Starting server...")
        
        # Start server thread
        self.server_thread = ServerThread(config, self)
        self.server_thread.log_signal.connect(self.log)
        self.server_thread.error_signal.connect(self.on_server_error)
        self.server_thread.start()
        
        # Start status monitor with session ID
        self.status_monitor.start(config["session_id"])
        
        # Start session timer
        self.session_start_time = datetime.now()
        self.duration_display.setText("00:00:00")
        self.session_timer.start(1000)  # Update every second
        
        self.is_running = True
        self.status_label.setText("● Starting...")
        self.status_label.setObjectName("statusConnecting")
        self.status_label.setStyleSheet("color: #f59e0b; font-weight: 700; font-size: 16px;")
        
    def stop_server(self):
        """Stop the translation server."""
        self.log("Stopping server...")
        
        # Stop session timer
        self.session_timer.stop()
        self.session_start_time = None
        
        if self.server_thread:
            self.server_thread.stop()
            self.server_thread.quit()
            self.server_thread.wait(5000)
            self.server_thread = None
            
        self.status_monitor.stop()
        
        # Update UI - Enhanced start button styling
        self.start_btn.setText("▶  Start Server")
        self.start_btn.setObjectName("")
        self.start_btn.setStyleSheet("""
            QPushButton {
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 0.5px;
                border-radius: 16px;
            }
        """)
        
        self.set_config_enabled(True)
        self.is_running = False
        
        # Reset status displays
        self.status_label.setText("● Offline")
        self.status_label.setObjectName("statusOffline")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: 700; font-size: 16px;")
        self.listener_count.setText("0")
        self.recording_label.setText("Stopped")
        self.recording_label.setStyleSheet("color: #6b7280;")
        self.mic_level.setValue(0)
        
        self.log("Server stopped")
    
    def update_session_duration(self):
        """Update the session duration display."""
        if self.session_start_time:
            elapsed = datetime.now() - self.session_start_time
            hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.duration_display.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
    def set_config_enabled(self, enabled: bool):
        """Enable/disable configuration widgets.
        
        Note: voice_combo, model_combo, and review_checkbox stay enabled
        because they can be changed live while the server is running.
        """
        # These can only be changed when stopped
        self.source_lang.setEnabled(enabled)
        self.target_lang.setEnabled(enabled)
        self.mic_combo.setEnabled(enabled)
        self.session_id.setEnabled(enabled)
        self.speaker_name.setEnabled(enabled)
        self.prompt_combo.setEnabled(enabled)
        # voice_combo, model_combo, review_checkbox stay enabled for live changes
        
    def on_status_update(self, status: dict):
        """Handle status updates from monitor."""
        if status.get("connected"):
            if status.get("is_live"):
                self.status_label.setText("● LIVE")
                self.status_label.setObjectName("statusLive")
                self.status_label.setStyleSheet("color: #22c55e; font-weight: 700; font-size: 18px; letter-spacing: 0.5px;")
                self.recording_label.setText("🔴 Recording...")
                self.recording_label.setStyleSheet("color: #22c55e; font-weight: 600;")
                # Simulate audio level (would be real data in production)
                import random
                self.mic_level.setValue(random.randint(30, 85))
            else:
                self.status_label.setText("● Connected")
                self.status_label.setObjectName("statusConnecting")
                self.status_label.setStyleSheet("color: #f59e0b; font-weight: 700; font-size: 16px;")
                self.mic_level.setValue(0)
                
            self.listener_count.setText(str(status.get("listener_count", 0)))
        else:
            if self.is_running:
                self.status_label.setText("● Connecting...")
                self.status_label.setStyleSheet("color: #f59e0b; font-weight: 700; font-size: 16px;")
                self.mic_level.setValue(0)
                
    def on_server_error(self, error: str):
        """Handle server errors."""
        self.log(f"ERROR: {error}")
        QMessageBox.critical(self, "Server Error", error)
        self.stop_server()
        
    def closeEvent(self, event):
        """Handle window close."""
        if self.is_running:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Server is running. Stop and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.stop_server()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# ============== Main ==============

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set app icon
    # app.setWindowIcon(QIcon("icon.ico"))
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
