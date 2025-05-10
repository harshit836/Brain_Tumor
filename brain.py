import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv

# Configuration
load_dotenv()

def generate_text(prompt):
    genai.configure(api_key="AIzaSyDOqcv4FyIEVNrAp7phjfFFmoOtgL-BX4Q")
    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash-thinking-exp-1219")
    response = model.generate_content(prompt)
    return response.text

# Load the trained models
with open("BrainTumor_predict.pkl", "rb") as file:
    model = pickle.load(file)

tf_model = load_model('cnn_model2.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_tumor(img_path):
    img_array = preprocess_image(img_path)
    prediction = tf_model.predict(img_array)
    return prediction

# Streamlit App
st.title("Brain Tumor Detection App")

# Sidebar Menu
menu = ["MRI Image Classification", "Feature-Based Prediction"]
choice = st.sidebar.selectbox("Select an Option", menu)

if choice == "MRI Image Classification":
    st.header("MRI Image Classification")
    uploaded_file = st.file_uploader("Upload a brain MRI image...", type=["jpg", "png"])
    
    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file)
        st.image(img, caption='Uploaded MRI.', use_column_width=True)
        st.write("Classifying...")
        prediction = predict_tumor(uploaded_file)
        class_label = np.argmax(prediction)
        confidence = np.max(prediction)

        if class_label == 0:
            tumor_type = "No Tumor"
        else:
            tumor_type = "Tumor Detected"

        st.write(f"Prediction: {tumor_type}")
        st.write(f"Confidence: {confidence:.2f}")

        if class_label == 1:
            tumor_description = generate_text("Describe the characteristics and types of brain tumors.")
            st.write("### Tumor Types Description")
            st.write(tumor_description)

elif choice == "Feature-Based Prediction":
    st.header("Feature-Based Brain Tumor Prediction")

    # Input fields
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    genetic_risk = st.selectbox("Genetic Risk", ["Low", "Moderate", "High"])
    smoking_history = st.selectbox("Smoking History", ["Never", "Former Smoker", "Current Smoker"])
    alcohol_consumption = st.selectbox("Alcohol Consumption", ["Never", "Occasionally", "Frequently"])
    radiation_exposure = st.selectbox("Radiation Exposure", ["No", "Yes"])
    head_injury_history = st.selectbox("Head Injury History", ["No", "Yes"])
    chronic_illness = st.selectbox("Chronic Illness", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    family_history = st.selectbox("Family History of Brain Tumor", ["No", "Yes"])
    symptom_severity = st.selectbox("Symptom Severity",["Mild","Severe","Moderate"])

    # Convert categorical variables into numerical form
    gender = 1 if gender == "Male" else 0
    genetic_risk = {"Low": 0, "Moderate": 1, "High": 2}[genetic_risk]
    smoking_history = {"Never": 0, "Former Smoker": 1, "Current Smoker": 2}[smoking_history]
    alcohol_consumption = {"Never": 0, "Occasionally": 1, "Frequently": 2}[alcohol_consumption]
    radiation_exposure = 1 if radiation_exposure == "Yes" else 0
    head_injury_history = 1 if head_injury_history == "Yes" else 0
    chronic_illness = 1 if chronic_illness == "Yes" else 0
    diabetes = 1 if diabetes == "Yes" else 0
    family_history = 1 if family_history == "Yes" else 0
    symptom_severity = {"Mild": 0, "Moderate": 1, "Severe": 2}[symptom_severity]

    # Prediction
    if st.button("Predict Based on Features"):
        input_data = np.array([[
            age, gender, genetic_risk, smoking_history, alcohol_consumption,
            radiation_exposure, head_injury_history, chronic_illness,
            diabetes, family_history, symptom_severity
        ]])

        prediction = model.predict(input_data)
        result = "Brain Tumor Detected" if prediction[0] == 1 else "No Brain Tumor Detected"

        st.subheader("Prediction Result:")
        st.write(result)