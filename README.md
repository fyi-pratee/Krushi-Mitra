# Krushi Mitra

Krushi Mitra is a Karnataka-focused crop recommendation and advisory web app designed for farmers. It combines machine learning with agronomic data to suggest the best crops, explain soil suitability, reveal profit potential, and provide a bilingual AI assistant for practical farming advice.

This project brings together:
- Crop recommendation using soil and environmental inputs
- Detailed crop profiles and lookup by crop name
- Profit and risk analysis
- Interactive analytics dashboard
- FarmerHelp AI assistant in English and Kannada
- District and soil-type aware recommendations

---

## Overview

The app helps farmers make better crop decisions by analyzing inputs such as:
- N, P, and K levels
- Soil pH
- Moisture
- Temperature
- District
- Soil type

It recommends the top 3 suitable crops and provides agronomic details such as:
- Best growing conditions
- Growth requirements
- Estimated profit range
- Fertilizer guidance
- Season suitability
- Regional crop recommendations

---

## Features

### 1. Crop Recommendation Engine
- Recommends the top 3 best crop options for a selected district
- Uses a trained Random Forest model
- Considers soil nutrients, pH, moisture, temperature, district, and soil type
- Shows confidence scores for each recommendation

### 2. Crop Details Lookup
- Allows lookup of crop details by English name, Kannada name, or alias
- Displays agronomic information such as:
  - Nutrient ranges
  - Soil preferences
  - Temperature and moisture needs
  - Suitable regions
  - Fertilizer guidance
  - Seasonal suitability

### 3. Profitability and Risk Analysis
- Estimates expected crop profit per acre using market-rate data
- Shows expected income and input cost ranges
- Includes risk level indicators such as Low, Medium, and High
- Helps compare crops based on profitability and viability

### 4. Interactive Analytics Dashboard
The app includes a dedicated analytics page with interactive charts and visualizations, including:
- Soil nutrient distribution (NPK heatmap)
- Temperature requirement chart
- pH vs moisture scatter plot
- Profitability analysis
- Risk vs profit bubble chart
- Feature importance ranking
- Seasonal distribution analysis
- Price trend analysis

These charts are designed for quick comparison between crops and support deeper understanding of agricultural trends.

### 5. FarmerHelp AI Assistant
A floating chatbot is available across the app to answer practical farming-related questions. It can help with:
- Weather and seasonal guidance
- Crop profitability questions
- Fertilizer calculations
- Soil requirement interpretation
- Crop comparisons
- Recommendations based on farmer input
- Context-aware agricultural advice

### 6. Bilingual Support (English + Kannada)
- Chat interface supports both English and Kannada
- Toggle between languages in the UI
- Crop names and response content are translated where applicable
- Helps improve accessibility for Kannada-speaking farmers

### 7. District and Soil Coverage
The app supports multiple Karnataka districts and soil types to provide more accurate and contextual recommendations.

Supported districts include:
- Bagalkot
- Ballari
- Belagavi
- Bengaluru Rural
- Bengaluru Urban
- Bidar
- Chamarajanagar
- Chikballapur
- Chikkamagaluru
- Chitradurga
- Dakshina Kannada
- Davanagere
- Dharwad
- Gadag
- Hassan
- Haveri
- Kalaburagi
- Kodagu
- Kolar
- Koppal
- Mandya
- Mysuru
- Raichur
- Ramanagara
- Shivamogga
- Tumakuru
- Udupi
- Uttara Kannada
- Vijayanagara
- Vijayapura
- Yadgir

Supported crops include:
- Arecanut
- Cardamom
- Chillies
- Coconut
- Coffee
- Cotton
- Mulberry
- Mustard
- Ragi
- Soybean
- Sugarcane
- Tea
- Turmeric

---

## Project Structure

```text
crop_app_final/
├── app.py                          # Flask app with routes, recommendation logic, analytics, and chatbot
├── final_train.py                  # Train/retrain the crop recommendation model
├── requirements.txt                # Python dependency list
├── karnataka_crop_recommendation_v5.csv
├── crop_model.pkl                 # Generated ML model
├── location_encoder.pkl            # Generated district encoder
├── soil_type_encoder.pkl           # Generated soil type encoder
├── templates/
│   ├── index.html                  # Home page with input form
│   ├── result.html                 # Recommendation results page
│   ├── analytics.html              # Analytics dashboard UI
│   └── chat_widget.html            # Floating chatbot UI
├── README.md                       # Project documentation
└── data files / model artifacts    # Used by the app for recommendations and analysis
```

---

## System Requirements

- Python 3.7 and above
- Flask
- scikit-learn
- pandas
- numpy
- plotly
- matplotlib

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model (required once)

```bash
python final_train.py
```

This step generates the model and encoder files needed by the app.

### 3. Run the application

```bash
python app.py
```

### 4. Open in browser

- Main app: http://127.0.0.1:5000
- Analytics page: http://127.0.0.1:5000/analytics

---

## How to Use the App

### Recommendation Flow
1. Open the app in the browser.
2. Select the district.
3. Select the soil type.
4. Enter soil values such as N, P, K, pH, moisture, and temperature.
5. Click the recommendation button.
6. View the top 3 crop suggestions with details and profitability insights.

### Crop Details Flow
- Use the crop lookup section to search by crop name.
- View agronomic requirements and crop-specific recommendations.

### Analytics Flow
- Navigate to the analytics page.
- Explore charts to compare crop characteristics and profitability trends.

### Chatbot Flow
- Click the floating green or yellow chat button.
- Ask questions in English or Kannada.
- Receive context-aware recommendations or advisory responses.

---

## API Endpoints

- `GET /` — Home page
- `POST /predict` — Crop recommendation
- `POST /crop-info` — Crop information lookup
- `POST /profit-predict` — Profit prediction
- `GET /analytics` — Analytics dashboard page
- `GET /api/analytics-data` — Analytics JSON data
- `POST /api/farmerhelp` — FarmerHelp chatbot logic
- `GET /api/season` — Current season information
- `GET /api/labs/<district>` — Soil testing labs information

---

## Technology Stack

- Backend: Flask (Python)
- Machine Learning: scikit-learn, Random Forest Classifier
- Data processing: pandas, numpy
- Visualizations: Plotly.js, matplotlib
- Frontend: HTML, CSS, JavaScript

---

## Data Sources and Model Notes

The system uses agronomic and market data based on Karnataka farming conditions. It incorporates crop needs, soil suitability, seasonal patterns, and price trends to generate recommendations and agricultural insights.

The main project data sources include:
- Karnataka agricultural guidance
- Crop board references
- Market pricing information
- Agricultural university references

---

## Troubleshooting

### Model or pickle errors
If you see dtype-related or pickle loading errors, delete the generated `.pkl` files and run:

```bash
python final_train.py
```

### Import errors
Ensure dependencies are installed first:

```bash
pip install -r requirements.txt
```

### Analytics not showing
Check the browser console and confirm Plotly scripts are loading correctly.

### Soil type or district fields missing
Ensure the latest app files are in place and the app has been restarted after changes.

---

## Summary

Krushi Mitra is a full-featured agricultural recommendation system for Karnataka farmers, combining predictive crop selection, data-driven insight, profitability analysis, and bilingual advisory support in one web application. It is designed to be simple to run locally, practical for real-world decisions, and helpful for farmers who need actionable crop guidance.

---

## Quick Command Reference

```bash
cd <project-folder>
pip install -r requirements.txt
python final_train.py
python app.py
```

Then open:
- http://127.0.0.1:5000
- http://127.0.0.1:5000/analytics

