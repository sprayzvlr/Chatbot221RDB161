import nltk
import numpy as np
import random
import tensorflow as tf
from src.data_preparation import DataPreparation
from src.model import ChatbotModel
import time
import json
import os

class Chatbot:
    def __init__(self):
        self.data_prep = DataPreparation()
        self.model = ChatbotModel()
        self.intents = {}
        self.conversation_history = []
        self.confidence_threshold = 0.25
        
    def load(self, model_path='models/chatbot_model.h5', data_dir='models'):
        """Load the trained model and data objects"""
        try:
            # Ensure data directory exists
            if not os.path.exists('data'):
                os.makedirs('data')
                
            # Check if intents.json exists, if not create a minimal version
            if not os.path.exists('data/intents.json'):
                print("intents.json not found. Creating a minimal version...")
                self._create_minimal_intents()
                
            # Check if model exists, if not, train a new one
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found. Training a new model...")
                from src.training import ModelTrainer
                trainer = ModelTrainer()
                history, _ = trainer.train(epochs=50, batch_size=8)
                print("Model training complete.")
                
            # Load data objects
            if not self.data_prep.load_data_objects(data_dir):
                print("Error loading data objects. Trying to create them...")
                # Try to create the data objects
                intents_loaded = self.data_prep.load_data('data/intents.json')
                if not intents_loaded:
                    print("Error loading intents.json. Creating a minimal version...")
                    self._create_minimal_intents()
                    self.data_prep.load_data('data/intents.json')
                
                self.data_prep.preprocess_data()
                self.data_prep.prepare_sequence_data()
                self.data_prep.save_data_objects(data_dir)
                
            # Load model
            if not self.model.load_model(model_path):
                print(f"Error loading model from {model_path}")
                return False
                
            # Load intents
            try:
                with open('data/intents.json', 'r', encoding='utf-8') as f:
                    self.intents = json.load(f)
            except json.JSONDecodeError as json_err:
                print(f"Error parsing intents.json: {json_err}")
                print("Please check your intents.json file for syntax errors.")
                print(f"Error occurred around character position {json_err.pos}")
                
                # Try to provide more specific information about the error location
                with open('data/intents.json', 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content[:json_err.pos].count('\n') + 1
                    print(f"Approximate location: line {lines}")
                    
                    # Try to extract a snippet around the error
                    start = max(0, json_err.pos - 20)
                    end = min(len(content), json_err.pos + 20)
                    snippet = content[start:end]
                    print(f"Context around error: '...{snippet}...'")
                
                return False
            return True
        except Exception as e:
            print(f"Error in load method: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def get_response(self, message):
        """Generate a response for the user message"""
        start_time = time.time()
        
        # Check if intents are loaded
        if not self.intents or 'intents' not in self.intents:
            print("Warning: No intents loaded. Trying to load them now.")
            try:
                with open('data/intents.json', 'r', encoding='utf-8') as f:
                    self.intents = json.load(f)
            except Exception as e:
                print(f"Failed to load intents: {e}")
                return {
                    'response': "I'm experiencing a configuration issue. Please try again later.",
                    'intent': 'error',
                    'confidence': 0.0,
                    'response_time': time.time() - start_time
                }
        
        try:
            # Preprocess the message
            message_seq = self.data_prep.tokenizer.texts_to_sequences([message])
            message_padded = tf.keras.preprocessing.sequence.pad_sequences(
                message_seq,
                maxlen=self.data_prep.max_sequence_length,
                padding='post'
            )
            
            # Predict intent
            predictions = self.model.predict(message_padded)
            
            # Get the predicted intent
            intent_index = np.argmax(predictions[0])
            intent_prob = predictions[0][intent_index]
            
            # Get the intent tag and responses
            if intent_index < len(self.data_prep.classes):
                intent_tag = self.data_prep.classes[intent_index]
            else:
                print(f"Warning: intent_index {intent_index} out of range. Using fallback intent.")
                intent_tag = "unknown"
                intent_prob = 0.0
                
            # Check confidence threshold
            if intent_prob < self.confidence_threshold:
                print(f"Low confidence ({intent_prob:.4f}) for intent: {intent_tag}. Using fallback response.")
                response = "I'm not entirely sure what you mean. Could you please rephrase that?"
                intent_tag = "uncertain"
            else:
                # Find the intent in the intents list
                found_intent = False
                for intent in self.intents['intents']:
                    if intent['tag'] == intent_tag:
                        responses = intent['responses']
                        # Choose a random response
                        response = random.choice(responses)
                        found_intent = True
                        break
                
                # If no matching intent was found in intents.json
                if not found_intent:
                    print(f"Warning: No responses found for intent: {intent_tag}")
                    response = "I understand, but I'm not sure how to respond to that."
                    intent_tag = "no_response"
                    intent_prob = max(intent_prob, 0.1)  # Ensure some confidence value
        
            # Calculate response time
            response_time = time.time() - start_time
            
            # Add to conversation history
            self.conversation_history.append({
                'user': message,
                'bot': response,
                'intent': intent_tag,
                'confidence': float(intent_prob),
                'response_time': response_time
            })
            
            # Return the response with metadata
            return {
                'response': response,
                'intent': intent_tag,
                'confidence': float(intent_prob),
                'response_time': response_time
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback response in case of errors
            fallback_response = "I'm sorry, I encountered an error processing your request."
            response_time = time.time() - start_time
            
            # Add to conversation history
            self.conversation_history.append({
                'user': message,
                'bot': fallback_response,
                'intent': 'error',
                'confidence': 0.0,
                'response_time': response_time
            })
            
            return {
                'response': fallback_response,
                'intent': 'error',
                'confidence': 0.0,
                'response_time': response_time
            }
    
    def get_conversation_history(self):
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        return True
    
    def save_conversation_history(self, filepath='logs/conversation_history.json'):
        """Save the conversation history to a file"""
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save conversation history
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=4)
            
        print(f"Conversation history saved to {filepath}")
        return True
    
    def _create_minimal_intents(self):
        """Create a minimal intents.json file with basic functionality"""
        minimal_intents = {
            "intents": [
                {
                    "tag": "greeting",
                    "patterns": ["Hi", "Hello", "Hey", "Sveiki"],
                    "responses": ["Hello! How can I help you?", "Hi there!", "Greetings!"]
                },
                {
                    "tag": "goodbye",
                    "patterns": ["Bye", "Goodbye", "See you"],
                    "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
                },
                {
                    "tag": "thanks",
                    "patterns": ["Thanks", "Thank you", "Paldies"],
                    "responses": ["You're welcome!", "No problem!", "Happy to help!"]
                },
                {
                    "tag": "help",
                    "patterns": ["Help", "I need help", "Palīdzība"],
                    "responses": ["How can I help you?", "What do you need help with?"]
                }
            ]
        }
        
        try:
            with open('data/intents.json', 'w', encoding='utf-8') as f:
                json.dump(minimal_intents, f, indent=2)
            print("Created minimal intents.json file")
            return True
        except Exception as e:
            print(f"Error creating minimal intents.json: {e}")
            return False
            
    def get_intent_statistics(self):
        """Get statistics about intents used in conversation"""
        if not self.conversation_history:
            return {}
            
        # Count occurrences of each intent
        intent_counts = {}
        for exchange in self.conversation_history:
            intent = exchange['intent']
            if intent in intent_counts:
                intent_counts[intent] += 1
            else:
                intent_counts[intent] = 1
                
        # Calculate percentages
        total = len(self.conversation_history)
        intent_percentages = {intent: count / total * 100 for intent, count in intent_counts.items()}
        
        # Calculate average confidence for each intent
        intent_confidences = {}
        for exchange in self.conversation_history:
            intent = exchange['intent']
            if intent in intent_confidences:
                intent_confidences[intent].append(exchange['confidence'])
            else:
                intent_confidences[intent] = [exchange['confidence']]
                
        avg_confidences = {intent: sum(confs) / len(confs) for intent, confs in intent_confidences.items()}
        
        # Calculate average response time
        avg_response_time = sum(exchange['response_time'] for exchange in self.conversation_history) / total
        
        # Return statistics
        return {
            'total_exchanges': total,
            'intent_counts': intent_counts,
            'intent_percentages': intent_percentages,
            'avg_confidences': avg_confidences,
            'avg_response_time': avg_response_time
        }
