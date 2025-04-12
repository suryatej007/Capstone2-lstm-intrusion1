import streamlit as st
from utils import preprocess_input, predict_intrusion

# Page configuration
st.set_page_config(page_title="Cloud Intrusion Detection", page_icon="🛡️")
st.title("🛡️ Cloud Intrusion Detection using LSTM")
st.markdown("Enter **comma-separated** input values matching the number of features your model was trained on.")

# Example input placeholder (change this to match actual feature count)
placeholder_text = "0,0,1,2,3,4,5,6,7,8"  # example with 10 features

user_input = st.text_input("Input Features:", placeholder_text)

if st.button("🔍 Detect Intrusion"):
    try:
        input_values = list(map(float, user_input.strip().split(",")))
        processed = preprocess_input(input_values)
        result = predict_intrusion(processed)

        if result:
            st.error("🚨 Intrusion Detected!")
        else:
            st.success("✅ Normal Traffic")

    except Exception as e:
        st.error(f"❌ Invalid input: {e}")
