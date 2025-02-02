from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load the model and encoders
model = joblib.load('voting_regressor_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
soil_le = joblib.load('soil_label_encoder.pkl')

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input data model
class PredictionInput(BaseModel):
    soil_type: str
    soil_moisture: float
    temperature: float
    rainfall: float
    humidity: float
    seed_type: str

import requests

# Replace this with your actual OpenWeather API key
API_KEY = "YOUR_API_KEY"
CITY = "YOUR_CITY"  # For example, "London"
UNIT = "metric"  # You can change this to "imperial" for Fahrenheit, or "metric" for Celsius

@app.get("/fetch_real_data")
async def fetch_real_data():
    # OpenWeather API endpoint for current weather data
    url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units={UNIT}"

    try:
        # Make a GET request to the OpenWeather API
        response = requests.get(url)
        data = response.json()

        # Check if the response contains valid data
        if response.status_code == 200:
            # Extract required fields from OpenWeather response
            real_data = {  # You might need additional sensor data for soil moisture
                "temperature": data['main']['temp'],
                "rainfall": data.get('rain', {}).get('24h', 0),  # Rainfall for the last 24 hours, default 0 if no rain
                "humidity": data['main']['humidity']
            }
            return real_data
        else:
            # If there's an error with the API request
            return {"error": "Unable to fetch data from OpenWeather API"}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


@app.post("/predict")
async def predict_depth(data: PredictionInput):
    try:
        # Prepare input data for model
        new_sample = pd.DataFrame({
            'Soil Type': [data.soil_type],
            'Soil Moisture (%)': [data.soil_moisture],
            'Temperature (°C)': [data.temperature],
            'Rainfall (mm)': [data.rainfall],
            'Humidity (%)': [data.humidity],
            'Seed Type': [data.seed_type],
        })
        
        # Encode categorical data
        new_sample['Soil Type'] = soil_le.transform(new_sample['Soil Type'])
        new_sample['Seed Type'] = le.transform(new_sample['Seed Type'])
        
        # Scale the data
        new_sample_scaled = scaler.transform(new_sample)
        
        # Predict the planting depth
        prediction = model.predict(new_sample_scaled)
        
        return {"prediction": round(prediction[0], 3)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
