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