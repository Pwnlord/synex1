import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('model/wine_cultivar_model.pkl')
scaler = joblib.load('model/scaler.pkl')

st.set_page_config(page_title="Wine Cultivar Predictor", page_icon="🍷")

st.title("🍷 Wine Cultivar Origin Prediction")
st.markdown("""
Predict the origin of a wine based on its chemical properties.
Enter the details below to see the predicted cultivar.
""")

# Input Fields for the 6 selected features
st.sidebar.header("Input Chemical Properties")

def user_input_features():
    alcohol = st.sidebar.slider('Alcohol', 11.0, 15.0, 13.0)
    magnesium = st.sidebar.slider('Magnesium', 70.0, 162.0, 100.0)
    flavanoids = st.sidebar.slider('Flavanoids', 0.3, 5.0, 2.0)
    color_intensity = st.sidebar.slider('Color Intensity', 1.0, 13.0, 5.0)
    hue = st.sidebar.slider('Hue', 0.4, 1.7, 1.0)
    proline = st.sidebar.slider('Proline', 270.0, 1680.0, 750.0)
    
    data = {
        'alcohol': alcohol,
        'magnesium': magnesium,
        'flavanoids': flavanoids,
        'color_intensity': color_intensity,
        'hue': hue,
        'proline': proline
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction Logic
if st.button("Predict Cultivar"):
    # 1. Scale input
    scaled_input = scaler.transform(input_df)
    
    # 2. Predict
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    # 3. Output
    cultivars = ['Cultivar 1', 'Cultivar 2', 'Cultivar 3']
    result = cultivars[prediction[0]]
    
    st.success(f"### The predicted origin is: **{result}**")
    
    # Visualizing probability
    st.write("#### Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=cultivars)
    st.bar_chart(prob_df.T)