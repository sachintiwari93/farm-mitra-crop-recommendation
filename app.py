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

    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    features = standscaler.transform(features)
    features = minmaxscaler.transform(features)

    prediction = model.predict(features)

    crop = labelencoder.inverse_transform(prediction)

    st.success("Recommended Crop: " + crop[0])

    crop_images = {
        "rice":"https://upload.wikimedia.org/wikipedia/commons/6/6f/Rice_plants.jpg",
        "maize":"https://upload.wikimedia.org/wikipedia/commons/0/0c/Maize.jpg",
        "banana":"https://upload.wikimedia.org/wikipedia/commons/8/8a/Banana-Single.jpg",
        "apple":"https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"
    }

    fertilizer = {
        "rice":"Use Nitrogen rich fertilizer",
        "maize":"Use NPK fertilizer",
        "banana":"Use Potassium rich fertilizer",
        "apple":"Use organic compost"
    }

    if crop[0] in crop_images:
        st.image(crop_images[crop[0]], width=300)

    st.write("Fertilizer Suggestion:", fertilizer.get(crop[0], "General fertilizer recommended"))


   


    
