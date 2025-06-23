import streamlit as st
import pickle
import numpy as np

# Load models
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

st.title("Algerian Forest Fire Prediction")

st.write("Enter the following details:")

Temperature = st.number_input("Temperature", value=20.0)
RH = st.number_input("Relative Humidity (RH)", value=40.0)
Ws = st.number_input("Wind Speed (Ws)", value=5.0)
Rain = st.number_input("Rain", value=0.0)
FFMC = st.number_input("FFMC", value=85.0)
DMC = st.number_input("DMC", value=50.0)
ISI = st.number_input("ISI", value=10.0)
Classes = st.number_input("Classes", value=1.0)
Region = st.number_input("Region", value=1.0)

if st.button("Predict"):
    input_data = np.array([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
    input_scaled = standard_scaler.transform(input_data)
    result = ridge_model.predict(input_scaled)
    st.success(f"Prediction: {result[0]}")
