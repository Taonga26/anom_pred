import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# API version
API_VERSION = 'v1'

# Load models and scalers
try:
    var_model = joblib.load('./models/var_model.pkl')
    var_scaler = joblib.load('./scalers/var_scaler.pkl')
    lstm_scaler = joblib.load('./scalers/lstm_scaler.pkl')
    lstm_model = load_model('./models/lstm_model.keras')
    iso_forest = joblib.load('./models/isolation_forest_model.pkl')
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {str(e)}")

# ------------------------------
# Helper functions
# ------------------------------

def create_sequences(X, seq_len=10):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X.iloc[i:i+seq_len].values)
        ys.append(X.iloc[i+seq_len].values)
    return np.array(Xs), np.array(ys)

# ------------------------------
# Anomaly Detection Logic
# ------------------------------

def detect_anomalies(df):
    
    if 'Date' in df.columns:
        if df['Date'].dtype == 'object':
            df['Date'] = pd.to_datetime(df['Date'], format='mixed')

        df.set_index('Date', inplace=True)

    # Scale with VAR scaler
    scaled_data = pd.DataFrame(
        var_scaler.transform(df),
        columns=df.columns,
        index=df.index
    )

    # Forecast with VAR
    k = var_model.k_ar
    steps = len(scaled_data)
    forecast_values = var_model.forecast(scaled_data.values[-k:], steps=steps)
    forecast_df = pd.DataFrame(forecast_values, index=scaled_data.index, columns=scaled_data.columns)

    # Compute residuals
    residuals = scaled_data - forecast_df
    residuals = residuals.dropna()

    # Scale residuals for LSTM
    residuals_scaled = pd.DataFrame(
        lstm_scaler.transform(residuals),
        columns=residuals.columns,
        index=residuals.index
    )

    # Create sequences
    X_sequences, y_true = create_sequences(residuals_scaled, seq_len=10)

    if len(X_sequences) == 0:
        # Not enough data
        return pd.DataFrame(), [], [], {'total_points': 0, 'anomalies': 0, 'percentages': {'anomalies': 0, 'normal': 0}}

    # LSTM predictions
    y_pred = lstm_model.predict(X_sequences)
    mse = np.mean(np.power(y_true - y_pred, 2), axis=1)

    # Isolation Forest anomaly detection
    labels = iso_forest.predict(mse.reshape(-1, 1))
    anomalies = np.where(labels == -1, 1, 0)

    # Results DataFrame
    res_index = residuals_scaled.iloc[-len(mse):].index
    results = pd.DataFrame({
        "Reconstruction_Error": mse,
        "Anomaly": anomalies
    }, index=res_index)

    # Identify anomalous points
    if results.empty:
        anomaly_count = 0
        anomalous_points = pd.DataFrame()
    else:
        anomaly_count = int(results['Anomaly'].sum())
        anom_idx = results.index[results['Anomaly'] == 1]
        common_idx = df.index.intersection(anom_idx)
        anomalous_points = df.loc[common_idx]

    # ---------------------------
    # JSON Payloads
    # ---------------------------

    # 1️⃣ EDA: full dataframe
    eda_payload = df.reset_index().to_dict(orient='records')

    # 2️⃣ Recent: anomalous points (Date, Close)
    if not anomalous_points.empty and 'Close' in anomalous_points.columns:
        recent_payload = anomalous_points.reset_index()[['Date', 'Close']].to_dict(orient='records')
    else:
        recent_payload = []

    # 3️⃣ Pie: total vs anomalies
    total_points = len(df)
    total_anoms = anomaly_count
    total = total_points if total_points > 0 else 1  # avoid zero division
    pie_payload = {
        'total_points': total_points,
        'anomalies': total_anoms,
        'percentages': {
            'anomalies': round((total_anoms / total) * 100, 2),
            'normal': round(((total_points - total_anoms) / total) * 100, 2)
        }
    }

    return anomalous_points, eda_payload, recent_payload, pie_payload

# ------------------------------
# API Routes
# ------------------------------

@app.route(f'/api/{API_VERSION}/status', methods=['GET'])
def status():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'version': API_VERSION,
        'service': 'anomaly-detection'
    })

@app.route(f'/api/{API_VERSION}/predict', methods=['POST'])
def predict():
    try:
        # Get uploaded file
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Read CSV
        df = pd.read_csv(file, parse_dates=['Date'])

        # Validate required columns

        cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        df = df[[col for col in df.columns if col in cols]]

        missing = [col for col in cols if col not in df.columns]
        if missing:
            return jsonify({
                'error': 'Missing required columns',
                'missing_columns': missing,
                'expected': cols,
                'received': list(df.columns)
            }), 400


        df= df[cols]

        # Detect anomalies
        anomalous_points, eda_payload, recent_payload, pie_payload = detect_anomalies(df)

        # Convert anomalies to JSON
        anomalies_json = anomalous_points.reset_index().to_dict(orient='records') if not anomalous_points.empty else []

        return jsonify({
            'status': 'success',
            'anomalies': anomalies_json,
            'eda': eda_payload,
            'recent': recent_payload,
            'pie': pie_payload,
            'anomalies_count': len(anomalies_json),
            'message': f'Found {len(anomalies_json)} anomalies'
        })

    except Exception as e:
        print("🔥 SERVER ERROR:", str(e))
        import traceback; 
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
# ------------------------------
# Run server
# ------------------------------

#if __name__ == '__main__':
#    app.run(debug=True, port=8000, host='0.0.0.0')
