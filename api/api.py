from fastapi import FastAPI
from pydantic import BaseModel
import dill
import pandas as pd

app = FastAPI()

with open("xgboost_gridsearch.pkl", "rb") as f:
    model = dill.load(f)

class ScoringItem(BaseModel):
    TransactionDate: str
    HouseAge: float
    DistanceToStation: float
    NumberOfPubs: float
    PostCode: str

@app.post("/predict/")
async def scoring_endpoint(item: ScoringItem):
    data = pd.DataFrame([item.model_dump().values()], columns=item.model_dump().keys())
    prediction = model.predict(data)
    return {"prediction": int(prediction)}

# Run the API
# uvicorn api:app --reload