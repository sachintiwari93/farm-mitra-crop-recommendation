import streamlit as st
import numpy as np
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
standscaler = pickle.load(open("standscaler.pkl", "rb"))
minmaxscaler = pickle.load(open("minmaxscaler.pkl", "rb"))
labelencoder = pickle.load(open("labelencoder.pkl", "rb"))

st.title("🌱 Farm Mitra - Crop Recommendation System")

st.write("Enter soil and weather details")

# Inputs
N = st.number_input("Nitrogen")
P = st.number_input("Phosphorus")
K = st.number_input("Potassium")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

if st.button("Predict Crop"):

    features = np.array([[N,P,K,temperature,humidity,ph,rainfall]])

    # Scaling
    features = standscaler.transform(features)
    features = minmaxscaler.transform(features)

    # Prediction
    prediction = model.predict(features)

    # Number → Crop Name
    crop = labelencoder.inverse_transform(prediction)

    st.success("Recommended Crop: " + crop[0])