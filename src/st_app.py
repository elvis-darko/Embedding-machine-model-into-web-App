import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import time
import os 

# Load the numerical imputer
num_imputer_filepath = "D:/Projects/Sepsis Classification/ML components/numerical_imputer.joblib"
num_imputer = joblib.load(num_imputer_filepath)

# Load the scaler
scaler_filepath = "D:/Projects/Sepsis Classification/ML components/scaler.joblib"
scaler = joblib.load(scaler_filepath)

# Load the Random Forest model
model_filepath = "D:/Projects/Sepsis Classification/ML components/rf_model.joblib"
model = joblib.load(model_filepath)

# Define a function to preprocess the input data
def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    num_columns = input_data_df.select_dtypes(include='number').columns

    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)

    return input_scaled_df

# Define a function to make the sepsis prediction
def predict_sepsis(input_data):
    input_scaled_df = preprocess_input_data(input_data)
    prediction = model.predict(input_scaled_df)[0]
    probabilities = model.predict_proba(input_scaled_df)[0]
    sepsis_status = "Positive" if prediction == 1 else "Negative"

    output_df = pd.DataFrame(input_data, columns=['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age', 'Insurance'])
    output_df['Prediction'] = sepsis_status
    output_df['Negative Probability'] = probabilities[0]
    output_df['Positive Probability'] = probabilities[1]

    return output_df, probabilities

# Create a Streamlit app
# Create a Streamlit app
def main():
    st.title('Sepsis Prediction App')

    # Display the image using HTML
    image_url = "https://lh3.googleusercontent.com/-UU_-cM2FZnI/YLgc3z-EFCI/AAAAAAAAAuo/sORie7aJNgsM8UY7_qAUTZUSeSxKtA7UQCLcBGAsYHQ/s16000/streamlit_log.png"
    st.markdown(f'<img src="{image_url}" alt="Streamlit Logo" style="width: 300px;">', unsafe_allow_html=True)

