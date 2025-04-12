import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

def preprocess_input(user_input):
    # Expect exactly 10 values for the model to work correctly.
    required_length = 10
    if len(user_input) < required_length:
        raise ValueError(f"Expected {required_length} values, but got {len(user_input)}. Please provide all values or pad with zeros.")
    elif len(user_input) > required_length:
        raise ValueError(f"Expected {required_length} values, but got {len(user_input)}. Please provide exactly {required_length} values.")
    
    # Convert to NumPy array and reshape for scaler.
    input_array = np.array(user_input).reshape(1, -1)  # shape: (1,10)
    input_scaled = scaler.transform(input_array)        # shape: (1,10)
    
    # Reshape to (batch_size, timesteps, features) as (1,10,1)
    input_reshaped = input_scaled.reshape(1, required_length, 1)
    return input_reshaped

def predict_intrusion(input_data, threshold=0.8):
    # Get the raw prediction score.
    prediction_score = model.predict(input_data)[0][0]
    # Print raw score for debugging (remove or comment out in production)
    print(f"Raw prediction score: {prediction_score:.4f}")
    # Return a boolean value based on threshold (True indicates an attack).
    return prediction_score > threshold
