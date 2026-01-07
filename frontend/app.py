import streamlit as st
import requests

st.title('Moral Values Analysis')

st.write("This app analyzes the moral values present in a given text.")

text_input = st.text_area("Enter text for analysis:")
model_name = st.selectbox("Select Model:", 
                          ["moral_model", "moralpolarity_model", "multimoral_model", "multimoralpolarity_model"])

data = {"text": text_input, "model": model_name}

if st.button("Analyze"):
    response = requests.post("http://moral-values-api:8000/predict", json=data)
    if response.status_code == 200:
        st.success("Analysis complete!")
        st.json(response.json())
    else:
        st.error("Error occurred during analysis.")