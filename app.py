import streamlit as st
from utils import preprocess_input, predict_intrusion

st.set_page_config(page_title="Cloud Intrusion Detection")
st.title("Cloud Intrusion Detection using LSTM")

st.markdown("Enter the input features (same number as model was trained on).")

user_input = st.text_input("Comma-separated input:", "0,0,1,2,...")  # Put example values here

if st.button("Detect Intrusion"):
    try:
        input_values = list(map(float, user_input.split(",")))
        processed = preprocess_input(input_values)
        result = predict_intrusion(processed)
        st.success("Attack Detected" if result else "Normal Traffic")
    except Exception as e:
        st.error(f"Invalid input: {e}")
