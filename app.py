import streamlit as st  
from utils import preprocess_input, predict_intrusion  

st.set_page_config(page_title="Intrusion Detection", layout="centered")  
st.title("Cloud Intrusion Detection using LSTM")  
st.write("Enter 10 features to analyze traffic behavior:")

input_values = []  
for i in range(10):  
    val = st.number_input(f"Feature {i+1}", value=0.0, step=0.1)  
    input_values.append(val)  

if st.button("Predict"):  
    processed = preprocess_input(input_values)  
    score = predict_intrusion(processed)  

    if score >= 0.5:  
        st.error(f"Intrusion Detected! (score: {score:.2f})")  
    else:  
        st.success(f"Normal Traffic (score: {score:.2f})")
