
import streamlit as st
from utils import preprocess_input, predict_intrusion

st.title("LSTM Intrusion Detection System")
user_input = st.text_input("Enter comma-separated input values:")
if user_input:
    try:
        user_input = list(map(float, user_input.split(",")))
        preprocessed = preprocess_input(user_input)
        prediction = predict_intrusion(preprocessed)
        st.write("Prediction:", prediction)
    except Exception as e:
        st.error(f"Invalid input: {e}")
