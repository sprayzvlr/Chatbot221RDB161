import os
import numpy as np
import matplotlib.pyplot as plt
from src.data_preparation import DataPreparation
from src.model import ChatbotModel
from sklearn.model_selection import train_test_split
import time
import json
import tensorflow as tf

class ModelTrainer:
    def __init__(self):
        self.data_prep = DataPreparation()
        self.model = ChatbotModel()
        self.history = None
        self.training_progress = {
            'epoch': 0,
            'accuracy': 0,
            'loss': 0,
            'val_accuracy': 0,
            'val_loss': 0,
            'is_training': False
        }
        self.train_time = 0

    def prepare_data(self, json_file='data/intents.json'):
        """Prepare the data for training"""
        # Load and preprocess the data
        self.data_prep.load_data(json_file)
        self.data_prep.preprocess_data()
        
        # Create sequence data for LSTM model
        X, y = self.data_prep.prepare_sequence_data()
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

    def train(self, epochs=150, batch_size=8, validation_split=0.1, use_advanced_model=True):
        """Train the model on the prepared data"""
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Create model
        vocab_size = len(self.data_prep.tokenizer.word_index) + 1
        output_shape = len(self.data_prep.classes)
        
        if use_advanced_model:
            self.model.create_advanced_model(vocab_size, output_shape)
        else:
            self.model.create_lstm_model(vocab_size, output_shape)
        
        # Reset training progress
        self.training_progress = {
            'epoch': 0,
            'accuracy': 0,
            'loss': 0,
            'val_accuracy': 0,
            'val_loss': 0,
            'is_training': True
        }
        
        # Create a custom callback to update training progress
        class TrainingProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, trainer):
                super().__init__()
                self.trainer = trainer
                
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                self.trainer.training_progress['epoch'] = epoch + 1
                self.trainer.training_progress['accuracy'] = float(logs.get('accuracy', 0))
                self.trainer.training_progress['loss'] = float(logs.get('loss', 0))
                self.trainer.training_progress['val_accuracy'] = float(logs.get('val_accuracy', 0))
                self.trainer.training_progress['val_loss'] = float(logs.get('val_loss', 0))
                print(f"Epoch {epoch+1}/{self.params['epochs']}: acc={logs.get('accuracy', 0):.4f}, loss={logs.get('loss', 0):.4f}")
        
        progress_callback = TrainingProgressCallback(self)
        
        # Train model
        start_time = time.time()
        self.history = self.model.train_model(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[progress_callback]
        )
        self.train_time = time.time() - start_time
        
        # Update training progress
        self.training_progress['is_training'] = False
        
        # Save model and data
        self.model.save_model()
        self.data_prep.save_data_objects()
        
        # Evaluate model
        evaluation = self.evaluate(X_test, y_test)
        
        return self.history, evaluation

    def evaluate(self, X_test, y_test):
        """Evaluate the model performance"""
        # Get model predictions
        predictions = self.model.predict(X_test)
        
        # Get the indices of the max values (predicted classes)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Calculate evaluation metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        # Find unique classes in the predictions and actual labels
        unique_classes = sorted(list(set(np.concatenate([np.unique(y_pred), np.unique(y_true)]))))
        class_names = [self.data_prep.classes[i] for i in unique_classes if i < len(self.data_prep.classes)]
        
        # Generate classification report with the right labels
        try:
            report = classification_report(y_true, y_pred, labels=unique_classes, 
                                         target_names=class_names, output_dict=True, zero_division=0)
        except Exception as e:
            print(f"Error generating classification report: {e}")
            # Create a simple report when the detailed one fails
            report = {
                "accuracy": float(accuracy),
                "macro avg": {"precision": 0, "recall": 0, "f1-score": 0},
                "weighted avg": {"precision": 0, "recall": 0, "f1-score": 0}
            }
        
        # Generate confusion matrix
        try:
            cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        except Exception as e:
            print(f"Error generating confusion matrix: {e}")
            cm = np.zeros((len(unique_classes), len(unique_classes)))
        
        # Prepare evaluation results
        evaluation = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'train_time': self.train_time
        }
        
        # Save evaluation results
        self._save_evaluation(evaluation)
        
        return evaluation

    def _save_evaluation(self, evaluation, filepath='models/evaluation.json'):
        """Save evaluation results to a file"""
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save evaluation results
        with open(filepath, 'w') as f:
            json.dump(evaluation, f, indent=4)
        
        print(f"Evaluation results saved to {filepath}")

    def load_evaluation(self, filepath='models/evaluation.json'):
        """Load saved evaluation results"""
        try:
            with open(filepath, 'r') as f:
                evaluation = json.load(f)
            return evaluation
        except Exception as e:
            print(f"Error loading evaluation: {e}")
            return None

    def plot_training_history(self, save_path='models/training_history.png'):
        """Plot the training history"""
        if self.history is None:
            print("No training history available.")
            return False
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'])
        ax1.plot(self.history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        ax2.plot(self.history.history['loss'])
        ax2.plot(self.history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Validation'], loc='upper left')
        
        # Save figure
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path)
        plt.close()
        
        print(f"Training history plot saved to {save_path}")
        return True

    def save_history(self, filepath='models/history.json'):
        """Save training history to a file"""
        if self.history is None:
            print("No training history available.")
            return False
        
        # Create history dictionary with all metrics
        history_dict = {key: [float(val) for val in values] for key, values in self.history.history.items()}
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save history dictionary
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Training history saved to {filepath}")
        return True

    def load_history(self, filepath='models/history.json'):
        """Load training history from a file"""
        try:
            with open(filepath, 'r') as f:
                history_dict = json.load(f)
            
            # Create a simple object to hold the history
            class SimpleHistory:
                def __init__(self, history_dict):
                    self.history = history_dict
            
            self.history = SimpleHistory(history_dict)
            print(f"Training history loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading history: {e}")
            return False
