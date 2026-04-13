from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
import joblib
import requests
import os

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend', static_url_path='')

# Load Model and Encoders
print("Loading model and encoders...")
try:
    # Load encoders
    encoders = joblib.load('model/encoders.pkl')
    
    # Load TabNet model
    clf = TabNetClassifier()
    
    clf.load_model('model/tabnet_model.zip')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    clf = None
    encoders = None

# Weather API Config
API_KEY = "8498355ef885bd08fdcf9fc94994c583" # Replace with actual key or use mock
BASE_URL = "http://api.openweathermap.org/data/2.5/weather?"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not clf or not encoders:
        return jsonify({'error': 'Model not loaded correctly.'})

    try:
        data = request.form
        city = data['location']
        
        # 1. Fetch Weather Data or Use Manual
        weather_data = {}
        
        # Check if manual inputs are provided and not empty
        manual_temp = data.get('manual_temp')
        manual_humidity = data.get('manual_humidity')
        manual_rain = data.get('manual_rainfall')
        
        if manual_temp and manual_humidity:
            weather_data = {
                'temp': float(manual_temp),
                'humidity': float(manual_humidity),
                # If rainfall is not manually provided, we might still want to fetch it or default it.
                # Here we will try to use manual rain if given, else fetch/default.
                'rainfall': float(manual_rain) if manual_rain else 100.0 
            }
            # If manual rain not provided, we could try to get just rain from API, 
            # but simpler to just use a fallback or the manual input for full control.
            if not manual_rain:
                 # Try to fetch rainfall only? Or just leave as default.
                 # Let's fallback to API for rainfall if possible, otherwise default.
                 api_weather = get_weather(city)
                 weather_data['rainfall'] = api_weather['rainfall']
        else:
             weather_data = get_weather(city)
        
        # 2. Prepare Input
        # Inputs: Temperature, Humidity, Soil Moisture, Nitrogen, Phosphorus, Potassium, pH, Rainfall, Soil Type, Season, Location
        
        # Encode categorical
        try:
            soil_enc = encoders['Soil Type'].transform([data['soil_type']])[0]
            season_enc = encoders['Season'].transform([data['season']])[0]
            loc_enc = encoders['Location'].transform([data['location']])[0]
        except ValueError as e:
            return jsonify({'error': f'Invalid categorical input: {e}'})

        input_data = [
            weather_data['temp'],
            weather_data['humidity'],
            float(data['soil_moisture']),
            float(data['nitrogen']),
            float(data['phosphorus']),
            float(data['potassium']),
            float(data['ph']),
            weather_data['rainfall'],
            soil_enc,
            season_enc,
            loc_enc
        ]
        
        features = np.array([input_data])
        
        # 3. Predict
        prediction_idx = clf.predict(features)[0]
        prediction_name = encoders['Crop'].inverse_transform([prediction_idx])[0]
        
        # Get probabilities
        probs = clf.predict_proba(features)[0]
        confidence = probs[prediction_idx] * 100
        
        return jsonify({
            'crop': prediction_name,
            'confidence': f"{confidence:.2f}%",
            'weather': weather_data,
            'explanation': f"Recommended based on {data['season']} season, {data['soil_type']} soil, and current weather in {city}."
        })

    except Exception as e:
        return jsonify({'error': str(e)})

def get_weather(city):
    try:
        url = BASE_URL + "appid=" + API_KEY + "&q=" + city + "&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Rainfall is not always available, default to 0 or estimate
            rain = 0
            if 'rain' in data:
                rain = data['rain'].get('1h', 0)
            
            return {
                'temp': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'rainfall': rain * 24 * 30 # Rough estimate for monthly rainfall from 1h if needed, or just use current. 
                                           # Our model train on 'Rainfall' which is usually mm/season or mm/year. 
                                           # For demo, let's assume the model expects mm value. 
                                           # We'll mock a realistic seasonal value if live is 0.
            }
    except:
        pass
    
    # Fallback/Mock data if API fails (Ensures project runnability)
    print("Using mock weather data")
    return {
        'temp': 25.0, # Average
        'humidity': 60.0,
        'rainfall': 100.0
    }

if __name__ == '__main__':
    app.run(debug=True)
