# AI Chatbot with TensorFlow

This project implements an intelligent chatbot using TensorFlow's neural network capabilities. The chatbot can understand user intents and generate appropriate responses.

## Features

- Intent-based response generation
- Neural network model with LSTM layers
- Interactive GUI with chat interface
- Real-time training and evaluation monitoring
- Statistics dashboard for performance metrics

## Requirements

- Python 3.8+
- TensorFlow 2.12.0
- Additional dependencies listed in requirements.txt

## Setup

1. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python main.py
   ```

## Project Structure

- `data/` - Contains the training data
- `models/` - Saved model files
- `src/` - Source code
  - `data_preparation.py` - Functions for processing the dataset
  - `model.py` - Neural network model implementation
  - `training.py` - Model training functionality
  - `evaluation.py` - Model evaluation metrics
  - `gui/` - GUI implementation files
- `main.py` - Application entry point

## Usage

1. Use the main chat window to interact with the bot
2. Training and Statistics windows show model performance
3. The model can be re-trained with new data
