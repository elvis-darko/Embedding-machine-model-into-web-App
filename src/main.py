from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import uvicorn
from fastapi.responses import JSONResponse

app = FastAPI()