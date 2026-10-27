"""
final_train.py
--------------
Run this ONCE before starting the Flask app:

    python final_train.py

It reads the CSV dataset, trains a RandomForest model, and saves
crop_model.pkl + location_encoder.pkl using YOUR installed scikit-learn
version — preventing the "incompatible dtype" pickle error.

Compatible with Python 3.7 and above.
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, 'karnataka_crop_recommendation_v5.csv')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'crop_model.pkl')
ENCODER_FILE = os.path.join(SCRIPT_DIR, 'location_encoder.pkl')
SOIL_ENCODER_FILE = os.path.join(SCRIPT_DIR, 'soil_type_encoder.pkl')

if not os.path.exists(CSV_FILE):
    print("ERROR: '{}' not found.".format(CSV_FILE))
    print("Make sure you're running this script from the correct directory.")
    exit(1)

print("Loading dataset from: {}".format(CSV_FILE))
df = pd.read_csv(CSV_FILE)
print("   {} rows | columns: {}".format(len(df), list(df.columns)))

# Encode district names -> integers
le_location = LabelEncoder()
df['Location_encoded'] = le_location.fit_transform(df['Location'])

# Encode soil types -> integers
le_soil = LabelEncoder()
df['Soil_Type_encoded'] = le_soil.fit_transform(df['Suitable Soil Type'])

# Features must match the order expected by app.py:
# N, P, K, temperature, ph, soil_moisture, Location_encoded, Soil_Type_encoded
X = df[['N', 'P', 'K', 'temperature', 'ph', 'soil_moisture', 'Location_encoded', 'Soil_Type_encoded']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training RandomForest (100 trees) ...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test) * 100
print("   Test accuracy: {:.2f}%".format(accuracy))
print("   Crop classes : {}".format(list(model.classes_)))
print("   Districts    : {}".format(list(le_location.classes_)))
print("   Soil Types   : {}".format(list(le_soil.classes_)))

# Save with protocol=2 for maximum backward compatibility (Python 3.7+)
print("\nSaving models to: {}".format(SCRIPT_DIR))
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(model, f, protocol=2)

with open(ENCODER_FILE, 'wb') as f:
    pickle.dump(le_location, f, protocol=2)

with open(SOIL_ENCODER_FILE, 'wb') as f:
    pickle.dump(le_soil, f, protocol=2)

print()
print("SUCCESS - files saved:")
print("   {}".format(MODEL_FILE))
print("   {}".format(ENCODER_FILE))
print("   {}".format(SOIL_ENCODER_FILE))
print()
print("Now run:  python app.py")

