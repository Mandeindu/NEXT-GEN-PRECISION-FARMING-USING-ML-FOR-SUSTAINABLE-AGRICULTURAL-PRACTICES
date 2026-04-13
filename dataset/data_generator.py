import pandas as pd
import numpy as np
import random

# Define crops and their ideal conditions (approximate ranges)
# Temp, Hum, pH, Water(Rainfall), N, P, K
crop_profiles = {
    'Rice': {'temp': (20, 27), 'hum': (80, 85), 'ph': (6.0, 7.0), 'rain': (180, 220), 'N': (60, 90), 'P': (35, 60), 'K': (35, 45)},
    'Maize': {'temp': (18, 27), 'hum': (50, 70), 'ph': (5.5, 7.0), 'rain': (60, 100), 'N': (60, 90), 'P': (35, 60), 'K': (15, 25)},
    'Chickpea': {'temp': (17, 22), 'hum': (50, 70), 'ph': (5.5, 6.5), 'rain': (30, 60), 'N': (20, 40), 'P': (55, 80), 'K': (75, 85)},
    'Jute': {'temp': (23, 26), 'hum': (70, 85), 'ph': (5.5, 7.0), 'rain': (140, 180), 'N': (60, 90), 'P': (35, 60), 'K': (35, 45)},
    'Cotton': {'temp': (22, 26), 'hum': (50, 70), 'ph': (6.0, 7.5), 'rain': (60, 90), 'N': (100, 140), 'P': (35, 60), 'K': (15, 25)},
    'Coconut': {'temp': (25, 29), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (150, 220), 'N': (20, 40), 'P': (10, 30), 'K': (25, 35)},
    'Papaya': {'temp': (23, 27), 'hum': (90, 95), 'ph': (6.0, 7.0), 'rain': (120, 220), 'N': (45, 65), 'P': (45, 65), 'K': (45, 55)},
    'Orange': {'temp': (10, 35), 'hum': (90, 95), 'ph': (6.0, 7.5), 'rain': (100, 130), 'N': (20, 40), 'P': (10, 30), 'K': (10, 15)},
    'Apple': {'temp': (-10, 15), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (100, 120), 'N': (20, 40), 'P': (120, 145), 'K': (195, 205)},
    'Muskmelon': {'temp': (25, 29), 'hum': (90, 95), 'ph': (5.5, 6.5), 'rain': (20, 30), 'N': (90, 120), 'P': (10, 30), 'K': (45, 55)},
    'Watermelon': {'temp': (24, 27), 'hum': (90, 95), 'ph': (6.0, 7.0), 'rain': (40, 60), 'N': (90, 120), 'P': (10, 30), 'K': (45, 55)},
    'Grapes': {'temp': (20, 29), 'hum': (80, 85), 'ph': (5.5, 6.5), 'rain': (60, 80), 'N': (20, 40), 'P': (120, 145), 'K': (195, 205)},
    'Mango': {'temp': (27, 34), 'hum': (45, 55), 'ph': (4.5, 6.5), 'rain': (85, 105), 'N': (20, 40), 'P': (15, 40), 'K': (25, 35)},
    'Banana': {'temp': (25, 29), 'hum': (75, 85), 'ph': (5.5, 6.5), 'rain': (90, 120), 'N': (90, 120), 'P': (65, 95), 'K': (45, 55)},
    'Pomegranate': {'temp': (18, 25), 'hum': (85, 95), 'ph': (5.5, 7.0), 'rain': (100, 115), 'N': (20, 40), 'P': (10, 30), 'K': (35, 45)},
    'Lentil': {'temp': (18, 29), 'hum': (60, 70), 'ph': (5.5, 7.0), 'rain': (35, 55), 'N': (10, 30), 'P': (55, 80), 'K': (15, 25)},
    'Blackgram': {'temp': (25, 33), 'hum': (60, 70), 'ph': (6.5, 7.5), 'rain': (60, 75), 'N': (40, 60), 'P': (55, 80), 'K': (15, 25)},
    'Mungbean': {'temp': (27, 30), 'hum': (60, 70), 'ph': (6.0, 7.5), 'rain': (35, 60), 'N': (20, 40), 'P': (35, 60), 'K': (15, 25)},
    'Mothbeans': {'temp': (24, 30), 'hum': (45, 65), 'ph': (4.0, 8.5), 'rain': (30, 70), 'N': (20, 40), 'P': (35, 60), 'K': (15, 25)},
    'Pigeonpeas': {'temp': (18, 30), 'hum': (45, 65), 'ph': (4.5, 7.5), 'rain': (90, 150), 'N': (20, 40), 'P': (55, 80), 'K': (15, 25)},
    'Kidneybeans': {'temp': (15, 25), 'hum': (20, 30), 'ph': (5.5, 6.0), 'rain': (60, 150), 'N': (20, 40), 'P': (55, 80), 'K': (15, 25)},
    'Coffee': {'temp': (23, 27), 'hum': (50, 70), 'ph': (6.0, 7.0), 'rain': (110, 190), 'N': (90, 120), 'P': (15, 35), 'K': (25, 35)}
}

soil_types = ['Black', 'Red', 'Loamy', 'Clay']
seasons = ['Kharif', 'Rabi', 'Zaid']
locations = ['Pune', 'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Nagpur', 'Nashik', 'Ahmedabad']

data = []

# Generate 2000 rows
for _ in range(2000):
    crop = random.choice(list(crop_profiles.keys()))
    profile = crop_profiles[crop]
    
    # Generate values with some noise around the ideal range
    # Reduced noise slightly for better separability to improve confidence
    temp = round(random.uniform(profile['temp'][0] - 1, profile['temp'][1] + 1), 1)
    hum = round(random.uniform(profile['hum'][0] - 3, profile['hum'][1] + 3), 1)
    ph = round(random.uniform(profile['ph'][0] - 0.2, profile['ph'][1] + 0.2), 1)
    rain = round(random.uniform(profile['rain'][0] - 5, profile['rain'][1] + 5), 1)
    N = int(random.uniform(profile['N'][0] - 5, profile['N'][1] + 5))
    P = int(random.uniform(profile['P'][0] - 5, profile['P'][1] + 5))
    K = int(random.uniform(profile['K'][0] - 5, profile['K'][1] + 5))
    
    # Add logic to correlate soil type and season roughly with crops for realism, 
    # but for synthetic data random choice is acceptable if feature dist is good.
    # We will pick random soil/season/location to show model robustness to these checks.
    
    soil = random.choice(soil_types)
    season = random.choice(seasons)
    location = random.choice(locations)
    
    # Refine season based on crop to make it slightly more realistic (Optional but good)
    if crop in ['Rice', 'Maize', 'Cotton', 'Jute', 'Soybean']:
        season = 'Kharif'
    elif crop in ['Wheat', 'Barley', 'Mustard', 'Peas', 'Chickpea']:
        season = 'Rabi'
    elif crop in ['Watermelon', 'Muskmelon', 'Cucumber']:
        season = 'Zaid'
        
    # Ensure values are non-negative
    temp = max(5.0, temp)
    hum = min(100.0, max(10.0, hum))
    rain = max(0.0, rain)
    N = max(0, N)
    P = max(0, P)
    K = max(0, K)
    
    row = {
        'Temperature': temp,
        'Humidity': hum,
        'Soil Moisture': hum - random.uniform(5, 15), # correlated with humidity roughly
        'Nitrogen': N,
        'Phosphorus': P,
        'Potassium': K,
        'pH': ph,
        'Rainfall': rain,
        'Soil Type': soil,
        'Season': season,
        'Location': location,
        'Crop': crop
    }
    # Fix soil moisture range
    row['Soil Moisture'] = float(max(10.0, min(90.0, row['Soil Moisture'])))
    
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('dataset/precision_farming_data.csv', index=False)
print("Dataset generated successfully at dataset/precision_farming_data.csv")
print(df.head())
