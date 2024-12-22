from fastapi import FastAPI
from pydantic import BaseModel
import dill
import pandas as pd

app = FastAPI()

with open("xgboost_gridsearch.pkl", "rb") as f:
    model = dill.load(f)

