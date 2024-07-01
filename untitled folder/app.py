import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Function to load joblib files with error handling
def load_joblib(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        st.error(f"Error: {filename} not found. Please ensure the file exists in the same directory as the app.")
        return None

# Load your trained model, scaler, imputer, and training columns
model = load_joblib('best_rf_model.joblib')
scaler = load_joblib('scaler.joblib')
imputer = load_joblib('imputer.joblib')
training_columns = load_joblib('training_columns.joblib')

# Sidebar information
st.sidebar.title("Model Information")
st.sidebar.write("Model: Random Forest Regressor")

st.sidebar.info("""
This app uses a trained Random Forest model to predict shipping costs based on various input parameters.
The model was trained with hyperparameter tuning to ensure optimal performance.
""")

st.title('Shipping Cost Prediction')
st.markdown("## Input Parameters")

# Input fields
input_data = {
    'Line Item Quantity': st.number_input('Line Item Quantity', min_value=1),
    'Unit of Measure (Per Pack)': st.number_input('Unit of Measure (Per Pack)', min_value=1),
    'Line Item Value': st.number_input('Line Item Value', min_value=0.0),
    'Unit Price': st.number_input('Unit Price', min_value=0.0),
    'Pack Price': st.number_input('Pack Price', min_value=0.0),
    'Weight (Kilograms)': st.number_input('Weight (Kilograms)', min_value=0.0),
    'Dosage Form': st.selectbox('Dosage Form', options=['Tablet', 'Capsule', 'Syrup']),
    'Manufacturing Site': st.selectbox('Manufacturing Site', options=['--Input--', 'Canada', 'China', 'Netherlands']),
    'Vendor': st.selectbox('Vendor', options=['--Input--', 'GSK Mississauga (Canada)', 'KHB Test Kit Facility, Shanghai China', 'MSD, Haarlem, Netherlands']),
    'Product Group': st.selectbox('Product Group', options=['--Input--', 'Antibiotics', 'COVID-19 medications', 'Anti-inflammatory drugs'])
}

if st.button('Predict Shipping Cost'):
    if model is None or scaler is None or imputer is None or training_columns is None:
        st.error("Error: Required model files are missing. Unable to make prediction.")
    else:
        # Prepare input data
        input_df = pd.DataFrame([input_data])

        # Convert categorical columns to dummy variables
        input_df = pd.get_dummies(input_df)

        # Ensure all required columns are present in the input data
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match the training data
        input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # Impute missing values
        input_data_imputed = imputer.transform(input_df)

        # Scale the features
        input_data_scaled = scaler.transform(input_data_imputed)

        # Make prediction
        base_prediction = model.predict(input_data_scaled)[0]

        # Adjust prediction based on manufacturing site
        if input_data['Manufacturing Site'] == 'Canada':
            adjusted_prediction = base_prediction * 0.7  # 30% decrease
            site_info = "Canada is a local manufacturing site, Cost-Saving Potential: High"
        elif input_data['Manufacturing Site'] == 'China':
            adjusted_prediction = base_prediction * 1.4  # 40% increase
            site_info = "China is a distant manufacturing site, Cost-Saving Potential: Low"
        elif input_data['Manufacturing Site'] == 'Netherlands':
            adjusted_prediction = base_prediction * 1.2  # 20% increase
            site_info = "The Netherlands is a moderate-distance manufacturing site, Cost-Saving Potential: Medium"
        else:
            adjusted_prediction = base_prediction
            site_info = "No specific site information available."

        st.success(f'Predicted Shipping Cost: ${adjusted_prediction:.2f}')
        st.info(site_info)

        # Additional insights
        st.subheader("Cost Breakdown:")
        st.write(f"Base Shipping Cost: ${base_prediction:.2f}")
        st.write(f"Adjusted Shipping Cost: ${adjusted_prediction:.2f}")
        
        if input_data['Manufacturing Site'] == 'Canada':
            st.write("Cost Saving: 30% (Local manufacturing advantage)")
        elif input_data['Manufacturing Site'] == 'China':
            st.write("Cost Increase: 40% (Long-distance shipping)")
        elif input_data['Manufacturing Site'] == 'Netherlands':
            st.write("Cost Increase: 20% (Moderate-distance shipping)")

        # Recommendations
        st.subheader("Recommendations:")
        if input_data['Manufacturing Site'] == 'China':
            st.write("Consider bulk shipping to reduce per-unit costs for long-distance transportation.")
        elif input_data['Manufacturing Site'] == 'Netherlands':
            st.write("Explore optimizing shipping routes to potentially reduce costs.")
        elif input_data['Manufacturing Site'] == 'Canada':
            st.write("Leverage local manufacturing to maintain low shipping costs.")


# Add some information about the app
st.info("""
This app predicts shipping costs based on various input parameters, with adjustments for different manufacturing sites.
Please ensure all fields are entered correctly for accurate predictions.
""")
