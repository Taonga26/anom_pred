# Anomaly Detection UI

This workspace contains a Flask-based anomaly detection backend (`app.py`) and a simple web UI (templates + static files) to upload a CSV and visualize the detected anomalies.

How to run

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Start the Flask app:

```powershell
python app.py
```

3. Open http://localhost:8000 in your browser and upload a CSV with a `Date` column and one or more numeric columns.

Notes

- The backend expects model files in `models/` and scalers in `scalers/` as referenced in `app.py`.
- If some model/scaler files are missing, the backend will return an error; ensure models exist or adapt `app.py`.
