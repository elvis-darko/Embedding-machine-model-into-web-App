from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()


# Load the numerical imputer, scaler, and model
num_imputer_filepath = "/com.docker.devenvironments.code/project_directory/ML components/numerical_imputer.joblib"
scaler_filepath = "/com.docker.devenvironments.code/project_directory/ML components/scaler.joblib"
model_filepath = "/com.docker.devenvironments.code/project_directory/ML components/rf_model.joblib"

num_imputer = joblib.load(num_imputer_filepath)
scaler = joblib.load(scaler_filepath)
model = joblib.load(model_filepath)

class PatientData(BaseModel):
    PRG: float
    PL: float
    PR: float
    SK: float
    TS: float
    M11: float
    BD2: float
    Age: float
    Insurance: int


def preprocess_input_data(input_data):
    input_data_df = pd.DataFrame([input_data])
    num_columns = [col for col in input_data_df.columns if input_data_df[col].dtype != 'object']
    input_data_imputed_num = num_imputer.transform(input_data_df[num_columns])
    input_scaled_df = pd.DataFrame(scaler.transform(input_data_imputed_num), columns=num_columns)
    return input_scaled_df
