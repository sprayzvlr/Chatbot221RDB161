import json
import numpy as np
import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

class DataPreparation:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.intents = {}
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_letters = ['?', '!', '.', ',']
        self.tokenizer = None
        self.max_sequence_length = 20

    def load_data(self, json_file='data/intents.json'):
        """Load and process the intents data from a JSON file"""
        try:
            with open(json_file, 'r', encoding='utf-8') as file:
                file_content = file.read()
                try:
                    self.intents = json.loads(file_content)
                except json.JSONDecodeError as e:
                    # Try to give helpful error information
                    lines = file_content.split('\n')
                    context_range = 3  # Show 3 lines before and after error
                    start_line = max(0, e.lineno - 1 - context_range)
                    end_line = min(len(lines), e.lineno - 1 + context_range + 1)
                    
                    error_context = "\n".join([
                        f"{i+1}: {line}" + (" <-- ERROR HERE" if i+1 == e.lineno else "")
                        for i, line in enumerate(lines[start_line:end_line])
                    ])
                    
                    print(f"JSON syntax error in {json_file}:")
                    print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
                    print("Context:")
                    print(error_context)
                    return False
            
            # Validate the intents structure
            if not isinstance(self.intents, dict) or 'intents' not in self.intents:
                print("Error: Invalid intents.json format. Missing 'intents' key.")
                return False
                
            if not isinstance(self.intents['intents'], list) or len(self.intents['intents']) == 0:
                print("Error: No intents found in the file.")
                return False
            
            # Check for duplicate tags - this can cause classification issues
            tags = [intent.get('tag') for intent in self.intents['intents'] if 'tag' in intent]
            if len(tags) != len(set(tags)):
                duplicates = [tag for tag in set(tags) if tags.count(tag) > 1]
                print(f"Warning: Found duplicate intent tags: {duplicates}")
                print("This may cause classification issues. Please ensure all tags are unique.")
                
            # Validate each intent
            for i, intent in enumerate(self.intents['intents']):
                if 'tag' not in intent:
                    print(f"Error: Intent at index {i} is missing a 'tag'.")
                    return False
                if 'patterns' not in intent or not isinstance(intent['patterns'], list):
                    print(f"Error: Intent '{intent.get('tag', f'at index {i}')}' is missing 'patterns' list.")
                    return False
                if 'responses' not in intent or not isinstance(intent['responses'], list):
                    print(f"Error: Intent '{intent.get('tag', f'at index {i}')}' is missing 'responses' list.")
                    return False
                
                # Check for empty patterns
                empty_patterns = [p for p in intent.get('patterns', []) if not p.strip()]
                if empty_patterns and len(empty_patterns) == len(intent.get('patterns', [])):
                    print(f"Warning: Intent '{intent.get('tag')}' has only empty patterns.")
            
            # Check for patterns overlap that might cause confusion
            all_patterns = {}
            for intent in self.intents['intents']:
                for pattern in intent.get('patterns', []):
                    normalized = pattern.lower().strip()
                    if normalized in all_patterns and all_patterns[normalized] != intent.get('tag'):
                        print(f"Warning: Pattern '{pattern}' exists in both '{all_patterns[normalized]}' and '{intent.get('tag')}' intents")
                    all_patterns[normalized] = intent.get('tag')
                    
            print(f"Successfully loaded {len(self.intents['intents'])} intents from {json_file}")
            return True
            
        except json.JSONDecodeError as e:
            print(f"JSON syntax error in {json_file}: {e}")
            print(f"Error at line {e.lineno}, column {e.colno}: {e.msg}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def preprocess_data(self):
        """Preprocess the loaded data for training"""
        # Extract all patterns and tags
        for intent in self.intents['intents']:
            tag = intent['tag']
            self.classes.append(tag)
            
            for pattern in intent['patterns']:
                # Tokenize and lemmatize each word
                word_list = nltk.word_tokenize(pattern)
                self.words.extend(word_list)
                self.documents.append((word_list, tag))
        
        # Lemmatize words and remove duplicates
        self.words = [self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_letters]
        self.words = sorted(list(set(self.words)))
        self.classes = sorted(list(set(self.classes)))
        
        print(f"Total patterns: {len(self.documents)}")
        print(f"Tags: {self.classes}")
        print(f"Unique lemmatized words: {len(self.words)}")
        
        return self.documents, self.classes, self.words

    def create_training_data(self):
        """Create the training data (X and y) for the model"""
        # Create an empty array for the training data
        training = []
        
        # Create an empty array for the output
        output_empty = [0] * len(self.classes)
        
        # Create the training set
        for doc in self.documents:
            # Initialize the bag of words
            bag = []
            
            # List of tokenized words for the pattern
            word_patterns = doc[0]
            
            # Lemmatize each word
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            
            # Create the bag of words array
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            # Create output row with 1 for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            
            # Add the bag of words and output row to training data
            training.append([bag, output_row])
        
        # Shuffle the training data
        random.shuffle(training)
        
        # Convert training data to numpy array
        training = np.array(training, dtype=object)
        
        # Split the features and target labels
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        return np.array(train_x), np.array(train_y)

    def prepare_sequence_data(self):
        """Prepare sequence data for LSTM model"""
        # Extract all patterns and their corresponding tags
        patterns = []
        tags = []
        
        # Download additional NLTK resources if needed
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
            
        # Common typing variations and abbreviations
        variations = {
            'hi': ['hii', 'hiiii', 'hiii', 'hai', 'hey'],
            'hello': ['helo', 'hullo', 'hallo'],
            'thanks': ['thx', 'thanx', 'thnks', 'thankyou', 'thank you'],
            'please': ['pls', 'plz'],
            'you': ['u', 'ya'],
            'for': ['4'],
            'to': ['2'],
            'are': ['r'],
            'weather': ['temp', 'temperature', 'forecast'],
            'joke': ['funny', 'humor', 'laugh', 'jokes']
        }
        
        for intent in self.intents['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                # Add the original pattern
                original_pattern = pattern.lower()
                patterns.append(original_pattern)
                tags.append(tag)
                
                # Tokenize the pattern
                words = nltk.word_tokenize(original_pattern)
                
                # Create word variations
                for i, word in enumerate(words):
                    if word in variations:
                        for variation in variations[word]:
                            variant_words = words.copy()
                            variant_words[i] = variation
                            variant_pattern = ' '.join(variant_words)
                            patterns.append(variant_pattern)
                            tags.append(tag)
                
                # Create grammar variations by tokenizing and slightly shuffling words
                if len(words) > 3:  # Only for longer phrases
                    for _ in range(3):  # Create 3 variations
                        # Keep first and last words, slight shuffle middle
                        middle = words[1:-1]
                        if len(middle) > 1:
                            # Swap two random adjacent words
                            idx = random.randint(0, len(middle)-2)
                            middle[idx], middle[idx+1] = middle[idx+1], middle[idx]
                            
                        new_pattern = ' '.join([words[0]] + middle + [words[-1]])
                        patterns.append(new_pattern)
                        tags.append(tag)
                        
                # Create word removal variations (simulate incomplete sentences)
                if len(words) > 3:
                    for _ in range(2):  # Create 2 variations
                        subset = words.copy()
                        # Remove a random non-essential word
                        removable_indices = [i for i, word in enumerate(subset) 
                                           if word.lower() in stop_words and i > 0 and i < len(subset)-1]
                        if removable_indices:
                            idx = random.choice(removable_indices)
                            del subset[idx]
                            new_pattern = ' '.join(subset)
                            patterns.append(new_pattern)
                            tags.append(tag)
        
        print(f"Total training patterns (with variations): {len(patterns)}")
        
        # Create a tokenizer
        self.tokenizer = Tokenizer(oov_token="<OOV>")
        self.tokenizer.fit_on_texts(patterns)
        
        # Convert patterns to sequences
        sequences = self.tokenizer.texts_to_sequences(patterns)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Create one-hot encoded outputs
        tag_to_index = {tag: i for i, tag in enumerate(self.classes)}
        y_data = np.zeros((len(tags), len(self.classes)))
        for i, tag in enumerate(tags):
            y_data[i, tag_to_index[tag]] = 1
        
        print(f"Vocabulary size: {len(self.tokenizer.word_index) + 1}")
        print(f"Number of classes: {len(self.classes)}")
        
        return padded_sequences, y_data

    def save_data_objects(self, save_dir='models'):
        """Save processed data objects for later use"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # Save words, classes, and tokenizer
        with open(f"{save_dir}/words.pkl", 'wb') as f:
            pickle.dump(self.words, f)
        
        with open(f"{save_dir}/classes.pkl", 'wb') as f:
            pickle.dump(self.classes, f)
            
        with open(f"{save_dir}/tokenizer.pkl", 'wb') as f:
            pickle.dump(self.tokenizer, f)
            
        print(f"Data objects saved to {save_dir}")
        return True
        
    def load_data_objects(self, load_dir='models'):
        """Load saved data objects"""
        try:
            with open(f"{load_dir}/words.pkl", 'rb') as f:
                self.words = pickle.load(f)
            
            with open(f"{load_dir}/classes.pkl", 'rb') as f:
                self.classes = pickle.load(f)
                
            with open(f"{load_dir}/tokenizer.pkl", 'rb') as f:
                self.tokenizer = pickle.load(f)
                
            print(f"Data objects loaded from {load_dir}")
            return True
        except Exception as e:
            print(f"Error loading data objects: {e}")
            return False

    def prepare_test_data(self, test_patterns):
        """Prepare test data for evaluation"""
        # Convert test patterns to sequences
        sequences = self.tokenizer.texts_to_sequences(test_patterns)
        
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        return padded_sequences
