# GenAI-Powered Algo Trading — Demo (scikit-learn baseline)

This repo is a modular Streamlit demo that:
- fetches historical price data (yfinance)
- computes technical indicators
- optionally computes (or fakes) sentiment embeddings
- trains a scikit-learn regression baseline (RandomForest) to predict multi-day returns
- backtests simple threshold-based signals
- provides Streamlit UI with caching and model artifact saving

**Important:** Educational only — not financial advice.

## Quick start (Windows / PowerShell)
```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Files created
- `app.py` — Streamlit app
- `utils/` — modular helpers
- `models/` — saved scikit-learn artifacts (joblib)
- `.vscode/` — VS Code tasks and launch configs
- `Dockerfile`, `Procfile` — deployment helpers

## Model & workflow notes
- This version uses **scikit-learn (RandomForestRegressor)** as a fast baseline.
- The app uses `st.cache_data` to cache fetched price data and embeddings for faster re-runs.
- Models are saved to `models/{ticker}_model.joblib`.
- Walk-forward CV: see the section **Walk-forward validation** below.

## Walk-forward validation (recommended)
1. Choose a number of folds or a window size (e.g., train on 2023-2024, validate 2025, then roll forward).
2. For each step:
   - Train model on the training window.
   - Validate on the next time window (hold-out).
   - Record metrics and aggregate.
3. Use only past data to train; never use future data in features/scaling for the validation period.
4. The repository includes a simple example in `utils/modeling.py` showing a time-based split function.

## Caching & artifacts
- Streamlit caching prevents repeated downloads and re-computation.
- Models saved with `joblib` in `models/` to speed up reloads.

## Deploy
- Build Docker image and run as in Dockerfile, or deploy to Streamlit Cloud/Render with `requirements.txt` and `.env`.
