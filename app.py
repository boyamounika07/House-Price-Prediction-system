import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64

# 🔹 Function to add background image
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# 🔹 Set background image
set_background("background1.jpg")

# 🔹 Load model and feature list
model = joblib.load("Model/house_price_model.pkl")
features = joblib.load("Model/features.pkl")

# 🔹 UI: House Price Form
st.title("🏠 House Price Predictor")

area = st.number_input("Area (sq ft):", min_value=500, max_value=10000)
bedrooms = st.number_input("Number of Bedrooms:", min_value=1, max_value=10)
bathrooms = st.number_input("Number of Bathrooms:", min_value=1, max_value=10)
stories = st.selectbox("Number of Stories", [1, 2, 3, 4])
mainroad = st.selectbox("Main Road Access", ['yes', 'no'])
guestroom = st.selectbox("Guest Room", ['yes', 'no'])
basement = st.selectbox("Basement", ['yes', 'no'])
hotwaterheating = st.selectbox("Hot Water Heating", ['yes', 'no'])
airconditioning = st.selectbox("Air Conditioning", ['yes', 'no'])
parking = st.slider("Parking Space", 0, 4)
prefarea = st.selectbox("Preferred Area", ['yes', 'no'])
furnishingstatus = st.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

# 🔹 Create input data for prediction
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'mainroad_' + mainroad: 1,
    'guestroom_' + guestroom: 1,
    'basement_' + basement: 1,
    'hotwaterheating_' + hotwaterheating: 1,
    'airconditioning_' + airconditioning: 1,
    'parking': parking,
    'prefarea_' + prefarea: 1,
    'furnishingstatus_' + furnishingstatus: 1,
}

# 🔹 Create empty row & assign values
input_df = pd.DataFrame(columns=features)
input_df.loc[0] = 0  # Fill with zeros
for k, v in input_data.items():
    if k in input_df.columns:
        input_df.at[0, k] = v

# 🔹 Predict
if st.button("Predict Price"):
    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated House Price: ₹ {price:,.0f}")
