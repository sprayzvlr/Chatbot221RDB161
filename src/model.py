import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Input, Bidirectional, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import os
import numpy as np

class ChatbotModel:
    def __init__(self):
        self.model = None
        self.seq_model = None
        self.input_shape = None
        self.vocab_size = None
        self.embedding_dim = 128
        self.max_sequence_length = 20

    def create_feed_forward_model(self, input_shape, output_shape):
        """Create a simple feed-forward neural network model"""
        self.input_shape = input_shape
        
        # Create model - 3 layers
        model = Sequential()
        model.add(Dense(128, input_shape=(input_shape,), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(output_shape, activation='softmax'))
        
        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        self.model = model
        return model
    
    def create_lstm_model(self, vocab_size, output_shape):
        """Create an LSTM model for sequence processing"""
        self.vocab_size = vocab_size
        
        # Create model with LSTM layers - improved architecture
        model = Sequential()
        # Start with embedding layer
        model.add(Embedding(vocab_size, self.embedding_dim, input_length=self.max_sequence_length))
        # Add a spatial dropout to reduce overfitting
        model.add(tf.keras.layers.SpatialDropout1D(0.2))
        
        # First Bidirectional LSTM layer with return sequences
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.3))
        
        # Second Bidirectional LSTM layer
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.3))
        
        # Add attention mechanism
        attention_output = tf.keras.layers.GlobalMaxPooling1D()(model.layers[-3].output)
        
        # Concatenate LSTM output and attention output
        dense_input = model.layers[-2].output
        
        # Add dense layers with batch normalization
        model.add(Dense(128, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(64, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(Dropout(0.3))
        
        # Add output layer
        model.add(Dense(output_shape, activation='softmax'))
        
        # Compile model with a lower learning rate for better convergence
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        
        self.seq_model = model
        print(model.summary())
        return model

    def create_advanced_model(self, vocab_size, output_shape):
        """Create an advanced neural network with attention mechanism"""
        self.vocab_size = vocab_size
        
        # Input layer
        inputs = Input(shape=(self.max_sequence_length,))
        
        # Embedding layer
        embedding = Embedding(
            input_dim=vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_sequence_length
        )(inputs)
        
        # Bidirectional LSTM layers
        lstm1 = Bidirectional(LSTM(128, return_sequences=True))(embedding)
        lstm1 = Dropout(0.2)(lstm1)
        
        lstm2 = Bidirectional(LSTM(64))(lstm1)
        lstm2 = Dropout(0.2)(lstm2)
        
        # Dense layers
        dense1 = Dense(128, activation='relu')(lstm2)
        dense1 = Dropout(0.2)(dense1)
        
        dense2 = Dense(64, activation='relu')(dense1)
        dense2 = Dropout(0.2)(dense2)
        
        # Output layer
        outputs = Dense(output_shape, activation='softmax')(dense2)
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.seq_model = model
        return model

    def train_model(self, X_train, y_train, epochs=150, batch_size=8, validation_split=0.1, callbacks=None):
        """Train the model on the given data"""
        if self.model is None and self.seq_model is None:
            raise ValueError("Model not created yet. Call create_model first.")
        
        # Create model directory if it doesn't exist
        if not os.path.exists('models'):
            os.makedirs('models')
        
        # Create callbacks
        checkpoint = ModelCheckpoint(
            'models/model.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            mode='min'
        )
        
        tensorboard = TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True
        )
        
        callbacks_list = [checkpoint, early_stopping, tensorboard]
        
        # Add custom callbacks if provided
        if callbacks:
            callbacks_list.extend(callbacks)
        
        # Choose which model to train
        model_to_train = self.seq_model if self.seq_model is not None else self.model
        
        # Train model
        history = model_to_train.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history

    def save_model(self, filepath='models/chatbot_model.h5'):
        """Save the model to a file"""
        if self.model is None and self.seq_model is None:
            raise ValueError("No model to save. Create and train a model first.")
        
        # Create model directory if it doesn't exist
        directory = os.path.dirname(filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Choose which model to save
        model_to_save = self.seq_model if self.seq_model is not None else self.model
        
        # Save model
        model_to_save.save(filepath)
        print(f"Model saved to {filepath}")
        return True

    def load_model(self, filepath='models/chatbot_model.h5'):
        """Load a saved model"""
        try:
            if not os.path.exists(filepath):
                print(f"Model file not found: {filepath}")
                # Try to create a default LSTM model
                print("Creating a fallback model...")
                vocab_size = 1000  # Fallback vocab size
                output_shape = 10  # Fallback output shape
                self.create_lstm_model(vocab_size, output_shape)
                self.save_model(filepath)
                return True
                
            print(f"Loading model from {filepath}")
            try:
                # Try to load the model with custom objects if needed
                model = load_model(filepath)
                
                # Determine which model type it is based on the input shape
                if len(model.input_shape) == 2:
                    # If input length doesn't match, print warning but still use model
                    if model.input_shape[1] != self.max_sequence_length:
                        print(f"Warning: Model input length ({model.input_shape[1]}) doesn't match expected length ({self.max_sequence_length}).")
                        print("Model will still be used, but results might be unpredictable.")
                    self.seq_model = model
                    self.max_sequence_length = model.input_shape[1]  # Update to match model
                    print("Loaded sequence model successfully")
                else:
                    self.model = model
                    print("Loaded feed-forward model successfully")
                    
                return True
            except Exception as inner_e:
                print(f"Primary loading method failed: {inner_e}")
                print("Trying alternative loading...")
                
                # Try to rebuild and load weights instead
                try:
                    # Create a temporary model with similar architecture
                    temp_model = tf.keras.models.Sequential([
                        tf.keras.layers.Embedding(1000, 128, input_length=self.max_sequence_length),
                        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(9, activation='softmax')
                    ])
                    temp_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    
                    # Try to load weights
                    temp_model.load_weights(filepath)
                    
                    # Use the loaded model
                    self.seq_model = temp_model
                    print("Successfully loaded model weights with alternative method")
                    return True
                except Exception as fallback_e:
                    print(f"Alternative loading also failed: {fallback_e}")
                    raise ValueError("Could not load model with any method") from fallback_e
                    
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, input_data):
        """Make a prediction with the model"""
        if self.model is None and self.seq_model is None:
            raise ValueError("No model loaded. Load a model first.")
        
        # Choose which model to use for prediction
        model_to_use = self.seq_model if self.seq_model is not None else self.model
        
        # Make prediction
        return model_to_use.predict(input_data)

    def predict_class(self, input_data, classes, threshold=0.25):
        """Predict the class for a given input"""
        prediction = self.predict(input_data)
        
        # Get the index with the highest probability
        results = []
        
        for pred in prediction:
            # Sort the predictions by probability
            sorted_indices = np.argsort(pred)[::-1]
            
            # Get predictions above threshold
            result = []
            
            # Adaptive thresholding:
            # 1. If highest confidence is very high (>0.7), only return that one
            # 2. If highest confidence is medium (0.5-0.7), use normal threshold
            # 3. If highest confidence is low (<0.5), increase threshold to avoid false positives
            
            highest_prob = pred[sorted_indices[0]]
            
            if highest_prob > 0.7:
                # High confidence - just return the top intent
                result.append({"intent": classes[sorted_indices[0]], "probability": float(highest_prob)})
            elif highest_prob > 0.5:
                # Medium confidence - use standard threshold
                for idx in sorted_indices:
                    if pred[idx] > threshold:
                        result.append({"intent": classes[idx], "probability": float(pred[idx])})
            elif highest_prob > 0.3:
                # Low confidence - increase threshold to avoid false positives
                adjusted_threshold = max(threshold, 0.4)
                for idx in sorted_indices:
                    if pred[idx] > adjusted_threshold:
                        result.append({"intent": classes[idx], "probability": float(pred[idx])})
            else:
                # Very low confidence - likely return fallback
                if "fallback" in classes:
                    fallback_idx = classes.index("fallback")
                    result.append({"intent": classes[fallback_idx], "probability": 1.0})
                else:
                    # No fallback found, return top match but mark as low confidence
                    result.append({"intent": classes[sorted_indices[0]], 
                                  "probability": float(highest_prob),
                                  "low_confidence": True})
            
            results.append(result)
        
        return results
