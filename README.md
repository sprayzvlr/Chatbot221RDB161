
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

- `data/` - Satur apmācības datus
- `models/` - Saglabātie modeļu faili
- `src/` - Pirmkods
  - `data_preparation.py` - Funkcijas datu kopas apstrādei
  - `model.py` - Neironu tīkla modeļa implementācija
  - `training.py` - Modeļa apmācības funkcionalitāte
  - `evaluation.py` - Modeļa novērtēšanas metrikas
  - `gui/` - Grafiskās lietotāja saskarnes implementācijas faili
- `main.py` - Aplikācijas sākumpunkts


