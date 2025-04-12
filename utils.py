import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

def preprocess_input(user_input):
    input_array = np.array(user_input).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    print("🔍 Scaled Input:", input_scaled)  # Debugging
    input_reshaped = input_scaled.reshape(1, input_scaled.shape[1], 1)
    return input_reshaped

def predict_intrusion(input_data, threshold=0.8):
    prediction = model.predict(input_data)[0][0]
    print(f"🔍 Raw Prediction Score: {prediction:.4f}")  # Debugging

    label = "Attack" if prediction > threshold else "Normal"
    return label
