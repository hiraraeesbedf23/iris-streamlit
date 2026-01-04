# app.py
import streamlit as st
import joblib
import numpy as np

# Title
st.title("🌸 Iris Flower Classification")

st.write("Enter the details of the flower below:")

# Input fields for features
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

# Load the trained model
model = joblib.load("iris_rf_model.joblib")

# Button for prediction
if st.button("Predict"):
    # Prepare input for model
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    
    # Map numeric prediction to species name
    species = ["Iris Setosa", "Iris Versicolor", "Iris Virginica"]
    predicted_species = species[prediction[0]]
    
    # Display result
    st.success(f"Predicted Flower Species: {predicted_species}")
