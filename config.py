"""
Configuration management for the Adaptive Neural Network system.
Centralizes all configuration parameters for easy modification and maintenance.
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class NetworkConfig:
    """Configuration for the neural network architecture."""
    input_size: int = 10
    hidden_size: int = 20
    output_size: int = 5
    learning_rate: float = 0.01
    modulation_rate: float = 0.1
    temporal_window: int = 5
    plasticity_threshold: float = 0.3
    decay_rate: float = 0.95
    max_synaptic_strength: float = 2.0
    min_synaptic_strength: float = 0.1

@dataclass
class FirebaseConfig:
    """Configuration for Firebase integration."""
    project_id: Optional[str] = os.getenv("FIREBASE_PROJECT_ID")
    credentials_path: Optional[str] = os.getenv("FIREBASE_CREDENTIALS_PATH")
    collection_name: str = "neural_network_states"
    performance_collection: str = "performance_metrics"
    
    def validate(self) -> bool:
        """Validate Firebase configuration."""
        if not self.project_id:
            raise ValueError("FIREBASE_PROJECT_ID environment variable not set")
        if not self.credentials_path:
            raise ValueError("FIREBASE_CREDENTIALS_PATH environment variable not set")
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(f"Firebase credentials file not found: {self.credentials_path}")
        return True

@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    performance_history_size: int = 100
    save_interval: int = 10

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file: str = "adaptive_network.log"
    enable_console: bool = True
    enable_file: bool = True
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

class ConfigManager:
    """Manages all configuration for the system."""
    
    def __init__(self):
        self.network = NetworkConfig()
        self.firebase = FirebaseConfig()
        self.training = TrainingConfig()
        self.logging = LoggingConfig()
        
    def validate_all(self) -> bool:
        """Validate all configurations."""
        try:
            self.firebase.validate()
            return True
        except (ValueError, FileNotFoundError) as e:
            print(f"Configuration validation failed: {e}")
            return False