import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from flask import Flask
import json
import sys
import requests
from datetime import datetime

def notebook_implementation():
    # Load the data
    df = pd.read_csv('Microsoft_stock_data.csv', parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    # Load all models and scalers
    var_model = joblib.load('models/var_model.pkl')
    lstm_model = load_model('models/lstm_model.keras')
    iso_model = joblib.load('models/isolation_forest_model.pkl')
    scaler_var = joblib.load('scalers/var_scaler.pkl')
    scaler_lstm = joblib.load('scalers/lstm_scaler.pkl')
    
    # Scale data using VAR scaler
    scaled_data = pd.DataFrame(
        scaler_var.transform(df),
        columns=df.columns,
        index=df.index
    )
    
    # Generate VAR predictions
    forecast_values = var_model.forecast(
        scaled_data.values[-var_model.k_ar:],
        steps=len(df)
    )
    forecast_df = pd.DataFrame(
        forecast_values,
        index=df.index,
        columns=df.columns
    )
    
    # Calculate residuals
    residuals = scaled_data - forecast_df
    residuals = residuals.fillna(0)
    
    # Scale residuals for LSTM
    residuals_scaled = pd.DataFrame(
        scaler_lstm.transform(residuals),
        columns=residuals.columns,
        index=residuals.index
    )
    
    # Create sequences for LSTM
    def create_sequences(X, seq_len=10):
        Xs = []
        for i in range(len(X) - seq_len):
            Xs.append(X.iloc[i:i+seq_len].values)
        return np.array(Xs)
    
    X = create_sequences(residuals_scaled)
    
    # Get LSTM predictions
    lstm_pred = lstm_model.predict(X)
    
    # Calculate reconstruction error
    mse = np.mean(np.power(X[-len(lstm_pred):, -1, :] - lstm_pred, 2), axis=1)
    
    # Use Isolation Forest for anomaly detection
    labels = iso_model.predict(mse.reshape(-1, 1))
    anomalies = np.where(labels == -1, 1, 0)
    
    # Create results DataFrame
    results = pd.DataFrame({
        "Reconstruction_Error": mse,
        "Anomaly": anomalies
    }, index=df.index[-len(mse):])
    
    return results, df

def flask_implementation():
    # Test the Flask endpoint
    url = 'http://localhost:8000/predict'
    files = {'file': open('Microsoft_stock_data.csv', 'rb')}
    response = requests.post(url, files=files)
    return response.json()

def compare_results():
    print("Starting comparison test...")
    print("\n1. Running notebook implementation...")
    notebook_results, original_df = notebook_implementation()
    
    print("\n2. Running Flask implementation...")
    flask_results = flask_implementation()
    
    if not flask_results.get('success', False):
        print("❌ Flask implementation failed:", flask_results.get('error'))
        return
    
    # Compare results
    print("\n3. Comparing results:")
    
    # Get anomalous dates from notebook
    notebook_anomalies = notebook_results[notebook_results['Anomaly'] == 1].index
    notebook_dates = set(notebook_anomalies.strftime('%Y-%m-%d').tolist())
    
    # Get anomalous dates from Flask
    flask_dates = set(flask_results['anomalous_dates'])
    
    # Compare dates
    print("\nDates Comparison:")
    print(f"Notebook anomalies: {len(notebook_dates)}")
    print(f"Flask anomalies: {len(flask_dates)}")
    
    # Check for exact matches
    matching_dates = notebook_dates.intersection(flask_dates)
    only_in_notebook = notebook_dates - flask_dates
    only_in_flask = flask_dates - notebook_dates
    
    print(f"\nMatching anomalies: {len(matching_dates)}")
    if len(matching_dates) > 0:
        print("✅ Sample matching dates:", list(matching_dates)[:5])
    
    if len(only_in_notebook) > 0:
        print("\n❌ Dates only in notebook:", list(only_in_notebook))
    
    if len(only_in_flask) > 0:
        print("\n❌ Dates only in Flask:", list(only_in_flask))
    
    # Compare error values
    print("\nError Values Comparison:")
    flask_data = {item['Date']: item['Reconstruction_Error'] 
                 for item in flask_results['anomalous_data']}
    
    error_diffs = []
    for date in matching_dates:
        notebook_error = notebook_results.loc[date, 'Reconstruction_Error']
        flask_error = flask_data[date]
        diff = abs(notebook_error - flask_error)
        error_diffs.append(diff)
    
    if error_diffs:
        avg_error_diff = sum(error_diffs) / len(error_diffs)
        max_error_diff = max(error_diffs)
        print(f"Average error difference: {avg_error_diff:.6f}")
        print(f"Maximum error difference: {max_error_diff:.6f}")
    
    # Final assessment
    if len(only_in_notebook) == 0 and len(only_in_flask) == 0:
        print("\n✅ SUCCESS: Both implementations detect the same anomalies!")
    else:
        print("\n❌ MISMATCH: Implementations produce different results")
        print(f"Match rate: {len(matching_dates)/(len(notebook_dates)):.2%}")

if __name__ == "__main__":
    # Ensure Flask server is running first
    try:
        requests.get('http://localhost:8000')
    except requests.exceptions.ConnectionError:
        print("❌ Error: Flask server is not running. Please start the Flask application first.")
        sys.exit(1)
    
    compare_results()