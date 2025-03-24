from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model and encoders
model = joblib.load("house_price_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Load districts list
districts = list(label_encoders["District"].classes_)

@app.route('/')
def home():
    return render_template('index.html', districts=districts)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Encode categorical inputs
        district = label_encoders["District"].transform([data["district"]])[0]
        mainroad = label_encoders["Mainroad"].transform([data["mainroad"]])[0]
        guestroom = label_encoders["Guestroom"].transform([data["guestroom"]])[0]
        basement = label_encoders["Basement"].transform([data["basement"]])[0]
        hotwater = label_encoders["HotwaterHeating"].transform([data["hotwater"]])[0]
        aircon = label_encoders["Airconditioning"].transform([data["aircon"]])[0]
        prefarea = label_encoders["Prefarea"].transform([data["prefarea"]])[0]
        furnishing = label_encoders["FurnishingStatus"].transform([data["furnishing"]])[0]

        # Collect numerical inputs
        features = np.array([[district, int(data["area"]), int(data["bedrooms"]), int(data["bathrooms"]),
                              int(data["stories"]), mainroad, guestroom, basement, hotwater, aircon,
                              int(data["parking"]), prefarea, furnishing]])
        
        features_scaled = scaler.transform(features)
        price = model.predict(features_scaled)[0]

        return jsonify({"price": f"{price:.2f}"})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
