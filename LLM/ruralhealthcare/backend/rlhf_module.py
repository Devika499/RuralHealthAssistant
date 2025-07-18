import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import os
from typing import Dict, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

class RLHFModule:
    def __init__(self):
        self.reward_model = None
        self.reward_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        
    def load_reward_model(self) -> bool:
        """Load the reward model and tokenizer"""
        try:
            logger.info("Loading reward model for RLHF...")
            
            # Path to the reward model
            model_path = "models/reward_model"
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Reward model not found at {model_path}")
                return False
            
            # Load tokenizer
            self.reward_tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.reward_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=3  # Match training setup
            ).to(self.device)
            
            # Set model to evaluation mode
            self.reward_model.eval()
            
            self.is_loaded = True
            logger.info("✅ Reward model loaded successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load reward model: {str(e)}")
            self.is_loaded = False
            return False
    
    def score_response(self, prompt: str, response: str) -> Dict:
        """
        Score a response using the reward model
        
        Args:
            prompt: The user's input prompt
            response: The model's response to score
            
        Returns:
            Dictionary containing scores and predicted rank
        """
        if not self.is_loaded:
            logger.error("Reward model not loaded. Cannot score response.")
            return {
                "scores": [0.33, 0.33, 0.34],  # Default uniform distribution
                "predicted_rank": 1,
                "error": "Reward model not loaded"
            }
        
        try:
            # Format input text according to the model's expected format
            input_text = f"<|user|>: {prompt} <|assistant|>: {response}"
            
            # Tokenize input
            inputs = self.reward_tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=384
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.reward_model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)[0].tolist()
                predicted_rank = int(torch.argmax(outputs.logits))
            
            return {
                "scores": scores,  # e.g., [0.1, 0.3, 0.6] — rank 2 most preferred
                "predicted_rank": predicted_rank  # e.g., 2
            }
            
        except Exception as e:
            logger.error(f"Error scoring response: {str(e)}")
            return {
                "scores": [0.33, 0.33, 0.34],
                "predicted_rank": 1,
                "error": str(e)
            }
    
    def get_model_status(self) -> Dict:
        """Get the status of the reward model"""
        return {
            "is_loaded": self.is_loaded,
            "device": self.device,
            "model_path": "models/reward_model" if os.path.exists("models/reward_model") else "Not found"
        }

# Global instance
rlhf = RLHFModule() 