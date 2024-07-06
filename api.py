from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import joblib
import io
import matplotlib.pyplot as plt
import numpy as np
from fastapi.responses import StreamingResponse

# Load the models
lr = joblib.load('linear_regression_model.pkl')
rf = joblib.load('random_forest_model.pkl')

app = FastAPI()

# Define the input data model
class FeatureDict(BaseModel):
    MolLogP: float
    MolWt: float
    NumRotatableBonds: int
    AromaticProportion: float

@app.post("/predict/graph")
def predict_graph(features: FeatureDict):
    try:
        df = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        
        # Generate predictions
        lr_pred = lr.predict(df)[0]
        rf_pred = rf.predict(df)[0]

        # Generate the graph
        fig, ax = plt.subplots()
        models = ['Linear Regression', 'Random Forest']
        predictions = [lr_pred, rf_pred]

        ax.bar(models, predictions, color=['blue', 'green'])
        ax.set_ylabel('Predicted LogS')
        ax.set_title('Predicted Solubility (LogS)')

        # Save the graph to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)  # Close the figure to free memory

        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(features: FeatureDict):
    try:
        df = pd.DataFrame([features.dict().values()], columns=features.dict().keys())
        lr_pred = lr.predict(df)[0]
        rf_pred = rf.predict(df)[0]
        return {"Linear Regression Prediction": lr_pred, "Random Forest Prediction": rf_pred}
    except Exception as e:
        raise HTTPException(status_code=400,detail=str(e))