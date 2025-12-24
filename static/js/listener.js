/**
 * Listener Logic for Integrated Server
 * iOS-compatible audio using HTML5 Audio element
 */

let websocket = null;
let isConnected = false;
let sessionCode = null;
let reconnectTimeout = null;
let audioUnlocked = false;
let pendingAudioQueue = [];
let audioElement = null;
let isPlaying = false;

// Debug logging for mobile
function debugLog(message) {
    console.log(message);
    // Also show in transcript area temporarily for mobile debugging
    const transcriptEl = document.getElementById('liveTranscriptText');
    if (transcriptEl && transcriptEl.dataset.debugMode !== 'false') {
        const time = new Date().toLocaleTimeString();
        transcriptEl.innerHTML = `<small style="opacity:0.6">[${time}] ${message}</small><br>${transcriptEl.innerHTML}`;
        // Limit debug messages
        const lines = transcriptEl.innerHTML.split('<br>');
        if (lines.length > 8) {
            transcriptEl.innerHTML = lines.slice(0, 8).join('<br>');
        }
    }
}

// DOM Elements
const elements = {
    status: document.getElementById('connectionStatus'),
    statusText: document.getElementById('statusText'),
    sessionCode: document.getElementById('sessionCodeDisplay'),
    targetLanguage: document.getElementById('targetLanguage'),
    visualizer: document.getElementById('audioVisualizer'),
    volumeSlider: document.getElementById('volumeSlider'),
    volumeValue: document.getElementById('volumeValue'),
    transcriptText: document.getElementById('liveTranscriptText'),
    errorMessage: document.getElementById('errorMessage'),
    audioUnlockOverlay: document.getElementById('audioUnlockOverlay'),
    audioUnlockBtn: document.getElementById('audioUnlockBtn')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Create hidden audio element for playback
    audioElement = document.createElement('audio');
    audioElement.setAttribute('playsinline', '');
    audioElement.setAttribute('webkit-playsinline', '');
    audioElement.preload = 'auto';
    document.body.appendChild(audioElement);

    // When audio ends, play next in queue
    audioElement.addEventListener('ended', () => {
        console.log("Audio ended");
        isPlaying = false;
        if (elements.visualizer) elements.visualizer.classList.remove('active');
        playNextInQueue();
    });

    audioElement.addEventListener('error', (e) => {
        console.error("Audio element error:", audioElement.error);
        isPlaying = false;
        playNextInQueue();
    });

    audioElement.addEventListener('canplaythrough', () => {
        console.log("Audio can play through");
    });

    // Get session code from URL
    const pathParts = window.location.pathname.split('/');
    if (pathParts.length > 2 && pathParts[2]) {
        sessionCode = pathParts[2];
        connectWebSocket(sessionCode);
    } else {
        showError("No session code in URL");
    }

    // Audio unlock - just mark as ready on any tap
    if (elements.audioUnlockBtn) {
        const unlockHandler = function (e) {
            e.preventDefault();
            e.stopPropagation();
            unlockAudio();
        };
        elements.audioUnlockBtn.addEventListener('click', unlockHandler);
        elements.audioUnlockBtn.addEventListener('touchend', unlockHandler);
    }

    // Also unlock on any tap on the overlay itself
    if (elements.audioUnlockOverlay) {
        elements.audioUnlockOverlay.addEventListener('click', unlockAudio);
        elements.audioUnlockOverlay.addEventListener('touchend', unlockAudio);
    }

    // Volume control
    if (elements.volumeSlider) {
        elements.volumeSlider.addEventListener('input', (e) => {
            const val = e.target.value;
            elements.volumeValue.textContent = `${val}%`;
            if (audioElement) {
                audioElement.volume = val / 100;
            }
        });
        // Set initial volume
        if (audioElement) {
            audioElement.volume = elements.volumeSlider.value / 100;
        }
    }
});

function unlockAudio() {
    console.log("Unlock audio requested");

    // Just mark as unlocked - actual play will happen with real audio
    audioUnlocked = true;

    // Hide overlay immediately
    if (elements.audioUnlockOverlay) {
        elements.audioUnlockOverlay.style.display = 'none';
    }

    console.log("Audio marked as unlocked, queue has", pendingAudioQueue.length, "items");

    // Try to play queued audio
    if (pendingAudioQueue.length > 0) {
        playNextInQueue();
    }
}

function playNextInQueue() {
    if (isPlaying || pendingAudioQueue.length === 0) {
        return;
    }

    const item = pendingAudioQueue.shift();
    playAudioInternal(item.audio, item.text, item.speaker);
}

function showError(message) {
    if (elements.status) elements.status.className = 'listener-status offline';
    if (elements.statusText) elements.statusText.textContent = 'Error';
    if (elements.errorMessage) {
        elements.errorMessage.textContent = message;
        elements.errorMessage.classList.remove('hidden');
    }
}

function connectWebSocket(code) {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/listener/${code}`;

    debugLog(`Connecting to ${wsUrl}...`);
    websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
        debugLog("WebSocket connected!");
        if (elements.status) elements.status.className = 'listener-status connecting';
        if (elements.statusText) elements.statusText.textContent = 'Connected - Waiting for speaker...';
        if (elements.errorMessage) elements.errorMessage.classList.add('hidden');
        isConnected = true;
    };

    websocket.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        debugLog(`Message: ${data.type}`);

        if (data.type === 'session_info') {
            updateSessionInfo(data.session);
        }
        else if (data.type === 'audio') {
            // Queue audio with its associated text (text will be shown when audio plays)
            playAudio(data.data, data.text, data.speaker_name);
        }
        else if (data.type === 'status') {
            if (data.is_live === false) {
                if (elements.status) elements.status.className = 'listener-status connecting';
                if (elements.statusText) elements.statusText.textContent = data.message || 'Waiting for speaker...';
                if (elements.visualizer) elements.visualizer.classList.remove('active');
            } else if (data.is_live === true) {
                if (elements.status) elements.status.className = 'listener-status live';
                if (elements.statusText) elements.statusText.textContent = 'LIVE';
            }
        }
    };

    websocket.onclose = (event) => {
        debugLog(`WebSocket closed: ${event.code} ${event.reason}`);
        if (elements.status) elements.status.className = 'listener-status offline';
        if (elements.statusText) elements.statusText.textContent = 'Disconnected';
        isConnected = false;

        // Reconnect with short delay
        reconnectTimeout = setTimeout(() => {
            if (!isConnected) {
                if (elements.statusText) elements.statusText.textContent = 'Reconnecting...';
                connectWebSocket(code);
            }
        }, 2000);
    };

    websocket.onerror = (error) => {
        console.error("WebSocket error:", error);
    };

    // Send periodic heartbeat to keep connection alive
    const heartbeatInterval = setInterval(() => {
        if (websocket.readyState === WebSocket.OPEN) {
            websocket.send("ping");
        } else {
            clearInterval(heartbeatInterval);
        }
    }, 25000);

    // Clean up on close
    websocket.addEventListener('close', () => {
        clearInterval(heartbeatInterval);
    });
}

function updateSessionInfo(session) {
    if (elements.sessionCode) elements.sessionCode.textContent = session.id;
    if (elements.targetLanguage) {
        elements.targetLanguage.textContent = session.target_lang_name || getLanguageName(session.target_lang);
    }

    if (session.is_live) {
        if (elements.status) elements.status.className = 'listener-status live';
        if (elements.statusText) elements.statusText.textContent = 'LIVE';
    } else {
        if (elements.status) elements.status.className = 'listener-status connecting';
        if (elements.statusText) elements.statusText.textContent = 'Waiting for speaker...';
    }
}

function playAudio(base64Data, text, speakerName) {
    // Queue the audio with its associated text
    pendingAudioQueue.push({ audio: base64Data, text: text, speaker: speakerName });
    debugLog(`Audio queued (${pendingAudioQueue.length} in queue)`);

    // Limit queue size
    if (pendingAudioQueue.length > 10) {
        pendingAudioQueue.shift();
    }

    // If audio is unlocked and not currently playing, start playback
    if (audioUnlocked && !isPlaying) {
        playNextInQueue();
    }
}

function playAudioInternal(base64Data, text, speakerName) {
    if (!audioElement) {
        console.warn("No audio element");
        return;
    }

    // Display the text when audio starts playing (synced with audio)
    if (text && elements.transcriptText) {
        if (speakerName) {
            elements.transcriptText.innerHTML = `<strong style="color: var(--primary);">[${speakerName}]</strong> ${text}`;
        } else {
            elements.transcriptText.textContent = text;
        }
    }

    try {
        // Create a Blob from base64 and use Object URL (better iOS support)
        const byteCharacters = atob(base64Data);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
            byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'audio/mpeg' });
        const audioUrl = URL.createObjectURL(blob);

        isPlaying = true;
        if (elements.visualizer) elements.visualizer.classList.add('active');

        // Set source and play
        audioElement.src = audioUrl;
        audioElement.volume = elements.volumeSlider ? elements.volumeSlider.value / 100 : 0.8;

        const playPromise = audioElement.play();

        if (playPromise !== undefined) {
            playPromise.then(() => {
                console.log("Playing audio successfully");
            }).catch(err => {
                console.error("Play failed:", err.name, err.message);
                isPlaying = false;
                if (elements.visualizer) elements.visualizer.classList.remove('active');

                // If it's a user interaction required error, show the overlay again
                if (err.name === 'NotAllowedError') {
                    audioUnlocked = false;
                    if (elements.audioUnlockOverlay) {
                        elements.audioUnlockOverlay.style.display = 'flex';
                    }
                } else {
                    // Other error, try next
                    setTimeout(playNextInQueue, 100);
                }
            });
        }

        // Clean up blob URL after playback
        audioElement.addEventListener('ended', () => {
            URL.revokeObjectURL(audioUrl);
        }, { once: true });

    } catch (err) {
        console.error("Error in playAudioInternal:", err);
        isPlaying = false;
    }
}

function getLanguageName(code) {
    const langs = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
        "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "tr": "Turkish"
    };
    return langs[code] || code;
}
