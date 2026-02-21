"""
Firebase integration manager for state persistence and real-time updates.
Handles all Firebase operations with robust error handling and connection management.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin.exceptions import FirebaseError

from config import ConfigManager

logger = logging.getLogger(__name__)

class FirebaseManager:
    """Manages Firebase connections and operations."""
    
    def __init__(self, config: ConfigManager):
        """Initialize Firebase connection with configuration."""
        self.config = config
        self.app = None
        self.db = None
        self._initialize_firebase()
        
    def _initialize_firebase(self) -> None:
        """Initialize Firebase connection with error handling."""
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.config.firebase.credentials_path)
                self.app = firebase_admin.initialize_app(
                    cred,
                    {'projectId': self.config.firebase.project_id}
                )
                logger.info("Firebase app initialized successfully")
            else:
                self.app = firebase_admin.get_app()
                logger.info("Using existing Firebase app")
            
            self.db = firestore.client()
            logger.info("Firestore client initialized")
            
        except FileNotFoundError as e:
            logger.error(f"Firebase credentials file not found: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid Firebase configuration: {e}")
            raise
        except FirebaseError as e:
            logger.error(f"Firebase initialization error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during Firebase initialization: {e}")
            raise
    
    def save_network_state(self, 
                          network_id: str, 
                          state_data: Dict[str, Any]) -> bool:
        """
        Save neural network state to Firestore.
        
        Args:
            network_id: Unique identifier for the network
            state_data: Dictionary containing network state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.db:
                logger.error("Firestore client not initialized")
                return False
            
            # Add metadata
            state_data['last_updated'] = datetime.utcnow()
            state_data['network_id'] = network_id
            
            # Save to Firestore
            doc_ref = self.db.collection(
                self.config.firebase.collection_name
            ).document(network_id)
            
            doc_ref.set(state_data)
            logger.info(f"Network state saved for ID: {network_id}")
            return True
            
        except FirebaseError as e:
            logger.error(f"Firebase error saving network state: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving network state: {e}")
            return False
    
    def load_network_state(self, network_id: str) -> Optional[Dict[str, Any]]:
        """
        Load neural network state from Firestore.
        
        Args:
            network_id: Unique identifier for the network
            
        Returns:
            Optional[Dict]: Network state if found, None otherwise
        """
        try:
            if not self.db:
                logger.error("Firestore client not initialized")
                return None
            
            doc_ref = self.db.collection(
                self.config.firebase.collection_name
            ).document(network_id)
            
            doc = doc_ref.get()
            
            if doc.exists:
                state_data = doc.to_dict()
                logger.info(f"Network state loaded for ID: {network_id}")
                return state_data
            else:
                logger.warning(f"No network state found for ID: {network_id}")
                return None
                
        except FirebaseError as e:
            logger.error(f"Firebase error loading network state: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error loading network state: {e}")
            return None
    
    def save_performance_metrics(self,
                                network_id: str,
                                metrics: Dict[str, Any]) -> bool:
        """
        Save performance metrics to Firestore.
        
        Args:
            network_id: Unique identifier for the network
            metrics: Dictionary containing performance metrics
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.db:
                logger.error("Firestore client not initialized")
                return False
            
            # Add metadata
            metrics['timestamp'] = datetime.utcnow()
            metrics['network_id'] = network_id
            
            # Save to Firestore
            doc_ref = self.db.collection(
                self.config.firebase.performance_collection
            ).document()
            
            doc_ref.set(metrics)
            logger.debug(f"Performance metrics saved for network: {network_id}")
            return True
            
        except FirebaseError as e:
            logger.error(f"Firebase error saving metrics: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error saving metrics: {e}")
            return False
    
    def get_performance_history(self, 
                               network_id: str, 
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve performance history for a network.
        
        Args:
            network_id: Unique identifier for the network
            limit: Maximum number of records to retrieve
            
        Returns:
            List[Dict]: List of performance metrics
        """
        try:
            if not self.db:
                logger.error("Firestore client not initialized")
                return []
            
            query = (self.db.collection(self.config.firebase.performance_collection)
                    .where('network_id', '