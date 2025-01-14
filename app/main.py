from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from app.model.model import __version__ as model 
from scripts.ML_pipline_functions import *

input_cols = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'StoreType', 'Assortment', 'Promo2', 'Day', 'Month', 'Year']
target_col = 'Sales'
# Define your column categories
num_cols = ['Store', 'DayOfWeek', 'Day', 'Month', 'Year']
cat_cols = ['DayOfWeek', 'Promo', 'StoreType', 'Assortment', 'Promo2']

app = FastAPI()

# Define the input schema for the prediction
class StoreData(BaseModel):
    Store: int
    DayOfWeek: int
    Day: int
    Month: int
    Year: int
    Promo: int
    StoreType: str
    Assortment: str
    Promo2: int

# Root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Predictive model API is running!"}

# Prediction endpoint
@app.post("/predict/")
def predict_sales(data: StoreData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Create pipeline
    pipeline = create_pipeline(input_cols, num_cols, cat_cols)

    # Preprocess the data using the pipeline
    preprocessed_data = pipeline.transform(input_data)

    # Make the prediction
    predicted_sales = model.predict(preprocessed_data)

    # Return the prediction
    return {"Predicted Sales": predicted_sales[0]}
