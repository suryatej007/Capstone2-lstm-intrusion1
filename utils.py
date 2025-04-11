import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

def preprocess_input(user_input):
    data = np.array(user_input).reshape(1, -1)
    scaled = scaler.transform(data)
    scaled = scaled.reshape(scaled.shape[0], 1, scaled.shape[1])
    return scaled

def predict_intrusion(processed_input):
    prediction = model.predict(processed_input)
    return int(prediction[0][0] > 0.5)
