"""
Configuration Manager for Realtime Translator

Handles persistent storage of user settings and API keys.
Keys are stored locally in the user's home directory.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any


class ConfigManager:
    """Manages persistent configuration storage for the application."""
    
    # Default config directory in user's home
    CONFIG_DIR = Path.home() / ".realtime_translator"
    CONFIG_FILE = CONFIG_DIR / "config.json"
    
    # Default configuration structure
    DEFAULT_CONFIG = {
        "api_keys": {
            "deepgram": "",
            "openai": "",
            "elevenlabs": "",
            "fish_audio": ""
        },
        "preferences": {
            "remember_keys": True,
            "default_source_lang": "tr",
            "default_target_lang": "en",
            "default_tts_service": "elevenlabs"
        }
    }
    
    def __init__(self):
        """Initialize the config manager and load existing config."""
        self._ensure_config_dir()
        self.config = self._load_config()
    
    def _ensure_config_dir(self):
        """Create config directory if it doesn't exist."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file, or create default if not exists."""
        if self.CONFIG_FILE.exists():
            try:
                with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    # Merge with defaults to handle any missing keys
                    return self._merge_with_defaults(saved_config)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load config file: {e}")
                return self.DEFAULT_CONFIG.copy()
        return self.DEFAULT_CONFIG.copy()
    
    def _merge_with_defaults(self, saved_config: Dict) -> Dict:
        """Merge saved config with defaults to ensure all keys exist."""
        result = self.DEFAULT_CONFIG.copy()
        
        # Deep merge
        for key, value in saved_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = {**result[key], **value}
            else:
                result[key] = value
        
        return result
    
    def save(self):
        """Save current configuration to file."""
        try:
            with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            print(f"Warning: Could not save config file: {e}")
    
    # ==================== API Keys ====================
    
    def get_api_key(self, service: str) -> str:
        """Get API key for a specific service."""
        return self.config.get("api_keys", {}).get(service, "")
    
    def set_api_key(self, service: str, key: str):
        """Set API key for a specific service."""
        if "api_keys" not in self.config:
            self.config["api_keys"] = {}
        self.config["api_keys"][service] = key
    
    def get_all_api_keys(self) -> Dict[str, str]:
        """Get all API keys."""
        return self.config.get("api_keys", {}).copy()
    
    def set_all_api_keys(self, keys: Dict[str, str], save_to_disk: bool = True):
        """Set all API keys at once."""
        self.config["api_keys"] = keys
        if save_to_disk and self.should_remember_keys():
            self.save()
    
    def has_api_key(self, service: str) -> bool:
        """Check if an API key is set for a service."""
        key = self.get_api_key(service)
        return bool(key and key.strip())
    
    def has_all_required_keys(self, tts_service: str = "elevenlabs") -> bool:
        """Check if all required API keys are set for the given TTS service."""
        required = ["deepgram", "openai"]
        
        # Add TTS-specific key requirements
        if tts_service in ["elevenlabs", "openai", "fish_audio", "deepgram"]:
            if tts_service != "openai":  # OpenAI key already required for translation
                required.append(tts_service)
        
        return all(self.has_api_key(service) for service in required)
    
    # ==================== Preferences ====================
    
    def should_remember_keys(self) -> bool:
        """Check if user wants to remember API keys."""
        return self.config.get("preferences", {}).get("remember_keys", True)
    
    def set_remember_keys(self, remember: bool):
        """Set whether to remember API keys."""
        if "preferences" not in self.config:
            self.config["preferences"] = {}
        self.config["preferences"]["remember_keys"] = remember
        self.save()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a preference value."""
        return self.config.get("preferences", {}).get(key, default)
    
    def set_preference(self, key: str, value: Any):
        """Set a preference value."""
        if "preferences" not in self.config:
            self.config["preferences"] = {}
        self.config["preferences"][key] = value
        self.save()
    
    # ==================== Clear / Reset ====================
    
    def clear_api_keys(self):
        """Clear all stored API keys."""
        self.config["api_keys"] = {
            "deepgram": "",
            "openai": "",
            "elevenlabs": "",
            "fish_audio": ""
        }
        self.save()
    
    def reset_to_defaults(self):
        """Reset all configuration to defaults."""
        self.config = self.DEFAULT_CONFIG.copy()
        self.save()


# Singleton instance for easy access
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the singleton ConfigManager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
