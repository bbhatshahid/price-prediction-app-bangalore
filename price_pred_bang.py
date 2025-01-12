import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature list
model = joblib.load('price_prediction_model.pkl')
X_train_columns = joblib.load('X_train_columns.pkl')  # Save your X_train column names during preprocessing

# Title of the app
st.title("House Price Prediction App")

# Create dropdowns for all encoded features
input_data = pd.DataFrame(np.zeros((1, len(X_train_columns))), columns=X_train_columns)

# Dropdown for encoded categorical features
for column in X_train_columns:
    if "area_type_" in column or "location_" in column:  # Example for one-hot encoded columns
        selected_option = st.selectbox(f"Select {column}", [0, 1])
        input_data[column] = selected_option
    else:
        # For numerical features, use a slider or input box
        input_value = st.number_input(f"Enter value for {column}", min_value=0.0, value=0.0)
        input_data[column] = input_value

# Predict price
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"The predicted price is: ₹{prediction[0]:,.2f}")
