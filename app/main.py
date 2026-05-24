import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Adult Income Prediction API")

# Load pipeline safely
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Define request schema matching the dataset exactly
class UserData(BaseModel):
    age: int
    fnlwgt: int
    educational_num: int
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    occupation: str
    relationship: str
    race: str
    gender: str
    native_country: str

@app.get("/")
def home():
    return {"message": "API is running. Send POST request to /predict"}

@app.post("/predict")
def predict(data: UserData):
    # Convert Pydantic data to dict, matching the exact CSV headers expected by the pipeline
    raw_data = data.model_dump()
    
    # Rename fields back to match dataset CSV structures (hyphens instead of underscores)
    formatted_data = {
        'age': raw_data['age'],
        'fnlwgt': raw_data['fnlwgt'],
        'educational-num': raw_data['educational_num'],
        'capital-gain': raw_data['capital_gain'],
        'capital-loss': raw_data['capital_loss'],
        'hours-per-week': raw_data['hours_per_week'],
        'workclass': raw_data['workclass'],
        'education': raw_data['education'],
        'marital-status': raw_data['marital_status'],
        'occupation': raw_data['occupation'],
        'relationship': raw_data['relationship'],
        'race': raw_data['race'],
        'gender': raw_data['gender'],
        'native-country': raw_data['native_country']
    }
    
    df = pd.DataFrame([formatted_data])
    
    # Run prediction
    prediction = model.predict(df)[0]
    
    # Optional: Get probability score if your model supports it
    try:
        probability = model.predict_proba(df)[0][prediction] * 100
    except:
        probability = None

    result = ">50K" if prediction == 1 else "<=50K"
    
    return {
        "income_class": result,
        "confidence": f"{probability:.2f}%" if probability else "N/A"
    }
