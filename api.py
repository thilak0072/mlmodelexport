from fastapi import FastAPI, HTTPException
import joblib
try:
    model = .load('train_model.plk') 
except FileNotFoundError:
    raise HTTPException(status_code=500,detail="Model file not found")
app = FastAPI()
@app.post("/predict")
async def predict(data: dict):
    try:
        predictions = model.predict(data['features'])
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))