import streamlit as st
import requests

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 style="color:black; text-align:left;">Moral Values Analysis</h1>', unsafe_allow_html=True)

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