
## Uzstadisana

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

## Projekta struktura

- `data/` - Contains the training data
- `models/` - Saved model files
- `src/` - Source code
  - `data_preparation.py` - Functions for processing the dataset
  - `model.py` - Neural network model implementation
  - `training.py` - Model training functionality
  - `evaluation.py` - Model evaluation metrics
  - `gui/` - GUI implementation files
- `main.py` - Application entry point


