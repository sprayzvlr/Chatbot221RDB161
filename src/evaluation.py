import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import time
import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ModelEvaluator:
    def __init__(self, model, data_prep):
        self.model = model
        self.data_prep = data_prep
        self.evaluation_results = {}
        self.test_patterns = []
        self.test_responses = []
        self.prediction_time = []
        self.confusion_matrix = None
        self.class_report = None
        
    def load_test_data(self, test_file=None):
        """Load test data or create from intents"""
        if test_file and os.path.exists(test_file):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_data = json.load(f)
                self.test_patterns = test_data['patterns']
                self.test_responses = test_data.get('responses', [])
                return True
            except Exception as e:
                print(f"Error loading test data: {e}")
                return False
        else:
            # Create test data from intents
            self.test_patterns = []
            self.test_responses = []
            
            # Take a few patterns from each intent for testing
            for intent in self.data_prep.intents['intents']:
                patterns = intent['patterns']
                responses = intent['responses']
                
                # Take up to 2 patterns from each intent
                num_patterns = min(2, len(patterns))
                for i in range(num_patterns):
                    self.test_patterns.append(patterns[i])
                    # Take the first response for each pattern
                    if responses:
                        self.test_responses.append(responses[0])
            
            return True
    
    def evaluate_model_performance(self, X_test, y_test):
        """Evaluate model performance on test data"""
        # Get predictions
        start_time = time.time()
        predictions = self.model.predict(X_test)
        end_time = time.time()
        
        # Calculate prediction time
        prediction_time = end_time - start_time
        avg_prediction_time = prediction_time / len(X_test)
        self.prediction_time = avg_prediction_time
        
        # Get predicted classes
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        # Calculate classification report
        self.class_report = classification_report(y_true, y_pred, 
                                                 target_names=self.data_prep.classes, 
                                                 output_dict=True)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Calculate precision, recall, f1-score
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        # Store evaluation results
        self.evaluation_results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_prediction_time': float(avg_prediction_time),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'classification_report': self.class_report
        }
        
        return self.evaluation_results
    
    def evaluate_response_quality(self):
        """Evaluate the quality of generated responses"""
        if not self.test_patterns:
            print("No test patterns available. Call load_test_data first.")
            return {}
            
        # Prepare test data
        test_sequences = self.data_prep.prepare_test_data(self.test_patterns)
        
        # Get predictions
        predictions = self.model.predict(test_sequences)
        
        # Get predicted classes
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get predicted intents
        predicted_intents = [self.data_prep.classes[idx] for idx in predicted_classes]
        
        # Get responses for predicted intents
        predicted_responses = []
        for intent_name in predicted_intents:
            response = self._get_response_for_intent(intent_name)
            predicted_responses.append(response)
        
        # Calculate BLEU scores if we have reference responses
        bleu_scores = []
        if self.test_responses:
            smooth = SmoothingFunction().method1
            for i, pred_resp in enumerate(predicted_responses):
                if i < len(self.test_responses):
                    reference = [self.test_responses[i].lower().split()]
                    candidate = pred_resp.lower().split()
                    try:
                        bleu = sentence_bleu(reference, candidate, smoothing_function=smooth)
                        bleu_scores.append(bleu)
                    except Exception as e:
                        print(f"Error calculating BLEU score: {e}")
                        bleu_scores.append(0)
        
        # Store response quality results
        response_quality = {
            'predicted_intents': predicted_intents,
            'predicted_responses': predicted_responses,
            'bleu_scores': bleu_scores,
            'avg_bleu_score': np.mean(bleu_scores) if bleu_scores else 0
        }
        
        # Update evaluation results
        self.evaluation_results.update(response_quality)
        
        return response_quality
    
    def _get_response_for_intent(self, intent_name):
        """Get a response for the given intent name"""
        for intent in self.data_prep.intents['intents']:
            if intent['tag'] == intent_name:
                if intent['responses']:
                    return np.random.choice(intent['responses'])
        return "I don't know how to respond to that."
    
    def plot_confusion_matrix(self, save_path='models/confusion_matrix.png'):
        """Plot the confusion matrix"""
        if self.confusion_matrix is None:
            print("No confusion matrix available. Evaluate the model first.")
            return False
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.data_prep.classes,
                   yticklabels=self.data_prep.classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path)
        plt.close()
        
        print(f"Confusion matrix plot saved to {save_path}")
        return True
    
    def plot_metrics(self, save_path='models/metrics.png'):
        """Plot evaluation metrics"""
        if not self.evaluation_results or not self.class_report:
            print("No evaluation results available. Evaluate the model first.")
            return False
        
        # Extract metrics from classification report
        metrics_data = {}
        for class_name, metrics in self.class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_data[class_name] = {
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1-score': metrics['f1-score']
                }
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(metrics_data).T
        
        # Plot metrics
        plt.figure(figsize=(12, 8))
        df.plot(kind='bar', figsize=(12, 8))
        plt.title('Performance Metrics by Class')
        plt.xlabel('Class')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        plt.savefig(save_path)
        plt.close()
        
        print(f"Metrics plot saved to {save_path}")
        return True
    
    def generate_evaluation_report(self, report_path='models/evaluation_report.json'):
        """Generate a comprehensive evaluation report"""
        if not self.evaluation_results:
            print("No evaluation results available. Evaluate the model first.")
            return False
        
        # Create the report
        report = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'LSTM' if self.model.seq_model is not None else 'Feed-Forward',
            'metrics': {
                'accuracy': self.evaluation_results.get('accuracy', 0),
                'precision': self.evaluation_results.get('precision', 0),
                'recall': self.evaluation_results.get('recall', 0),
                'f1_score': self.evaluation_results.get('f1_score', 0),
                'bleu_score': self.evaluation_results.get('avg_bleu_score', 0)
            },
            'performance': {
                'avg_prediction_time': self.evaluation_results.get('avg_prediction_time', 0),
                'num_classes': len(self.data_prep.classes),
                'num_test_samples': len(self.test_patterns)
            },
            'class_report': self.class_report,
            'classes': self.data_prep.classes
        }
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(report_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save the report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Evaluation report saved to {report_path}")
        return True
    
    def load_evaluation_report(self, report_path='models/evaluation_report.json'):
        """Load a saved evaluation report"""
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            # Assign relevant parts to evaluation_results
            self.evaluation_results = {
                'accuracy': report['metrics']['accuracy'],
                'precision': report['metrics']['precision'],
                'recall': report['metrics']['recall'],
                'f1_score': report['metrics']['f1_score'],
                'avg_prediction_time': report['performance']['avg_prediction_time'],
                'classification_report': report['class_report']
            }
            
            print(f"Evaluation report loaded from {report_path}")
            return report
        except Exception as e:
            print(f"Error loading evaluation report: {e}")
            return None
            
    def evaluate_response_time(self, num_samples=100):
        """Evaluate model response time with multiple samples"""
        if not self.data_prep.intents:
            print("No data loaded. Load data first.")
            return {}
            
        # Generate random test queries
        test_queries = []
        for intent in self.data_prep.intents['intents']:
            patterns = intent['patterns']
            if patterns:
                # Add some patterns from each intent
                num_to_add = min(2, len(patterns))
                test_queries.extend(patterns[:num_to_add])
        
        # If we need more queries, repeat some
        while len(test_queries) < num_samples:
            test_queries.append(np.random.choice(test_queries))
        
        # Trim to desired number
        test_queries = test_queries[:num_samples]
        
        # Prepare test data
        test_sequences = self.data_prep.prepare_test_data(test_queries)
        
        # Measure prediction time
        start_time = time.time()
        _ = self.model.predict(test_sequences)
        end_time = time.time()
        
        # Calculate statistics
        total_time = end_time - start_time
        avg_time = total_time / num_samples
        
        # Store results
        response_time_results = {
            'total_samples': num_samples,
            'total_prediction_time': float(total_time),
            'avg_prediction_time': float(avg_time),
            'predictions_per_second': float(num_samples / total_time)
        }
        
        # Update evaluation results
        self.evaluation_results.update({'response_time': response_time_results})
        
        return response_time_results
