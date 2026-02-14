import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os

# ---------------- Load Model & Resources ----------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
LE_GENDER_PATH = os.path.join(BASE_DIR, "label_encoder_gender.pkl")
OHE_GEO_PATH = os.path.join(BASE_DIR, "onehot_encoder_geo.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

with open(LE_GENDER_PATH, "rb") as file:
    label_encoder_gender = pickle.load(file)

with open(OHE_GEO_PATH, "rb") as file:
    onehot_encoder_geo = pickle.load(file)


# ---------------- Streamlit App ----------------

st.title("Customer Churn Prediction")

# Get known geography categories
known_geo = onehot_encoder_geo.get_feature_names_out(['Geography'])

# Inputs
geography = st.selectbox("Geography", list(known_geo) + ["Unknown"])
gender = st.selectbox("Gender", list(label_encoder_gender.classes_))
age = st.slider("Age", 18, 100, 30)
balance = st.number_input("Balance", 0.0, value=1000.0)
credit_score = st.number_input("Credit Score", 300, 850, 600)
estimated_salary = st.number_input("Estimated Salary", 0.0, value=50000.0)
tenure = st.slider("Tenure (years)", 0, 10, 3)
num_of_products = st.slider("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

# Prepare base input
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender],       # still string for now
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Gender to numeric using saved LabelEncoder
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# One-hot encode Geography manually
geo_vector = np.zeros((1, len(known_geo)))
if geography in known_geo:
    idx = list(known_geo).index(geography)
    geo_vector[0, idx] = 1
geo_df = pd.DataFrame(geo_vector, columns=known_geo)

# Combine everything
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Scale numeric features
input_data_scaled = scaler.transform(input_data)

# Predict churn probability
prediction_proba = model.predict(input_data_scaled)[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

# Show result
if prediction_proba > 0.5:
    st.write("The customer is likely to churn")
else:
    st.write("The customer is not likely to churn")
