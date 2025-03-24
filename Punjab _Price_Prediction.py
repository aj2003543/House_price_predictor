import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("punjab_apartment_prices.csv")

# Encode categorical variables
label_encoders = {}
categorical_columns = ["District", "Mainroad", "Guestroom", "Basement", "HotwaterHeating", "Airconditioning", "Prefarea", "FurnishingStatus"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")

# Define features and target variable
X = df[["District", "Area_sqft", "Bedrooms", "Bathrooms", "Stories", "Mainroad", "Guestroom", "Basement", "HotwaterHeating", "Airconditioning", "Parking", "Prefarea", "FurnishingStatus"]]
y = df["Price"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "house_price_model.pkl")
print("Model trained and saved successfully!")

# Function to predict price
def predict_price(district, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwater, aircon, parking, prefarea, furnishing):
    model = joblib.load("house_price_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Validate district name
    if district not in label_encoders["District"].classes_:
        return f"Error: '{district}' is not a valid district. Please enter a valid district name from: {', '.join(label_encoders['District'].classes_)}"
    
    # Encode categorical values
    district_encoded = label_encoders["District"].transform([district])[0]
    mainroad_encoded = label_encoders["Mainroad"].transform([mainroad])[0]
    guestroom_encoded = label_encoders["Guestroom"].transform([guestroom])[0]
    basement_encoded = label_encoders["Basement"].transform([basement])[0]
    hotwater_encoded = label_encoders["HotwaterHeating"].transform([hotwater])[0]
    aircon_encoded = label_encoders["Airconditioning"].transform([aircon])[0]
    prefarea_encoded = label_encoders["Prefarea"].transform([prefarea])[0]
    furnishing_encoded = label_encoders["FurnishingStatus"].transform([furnishing])[0]
    
    # Prepare input features
    features = np.array([[district_encoded, area, bedrooms, bathrooms, stories, mainroad_encoded, guestroom_encoded, basement_encoded, hotwater_encoded, aircon_encoded, parking, prefarea_encoded, furnishing_encoded]])
    features_scaled = scaler.transform(features)
    
    # Predict price
    predicted_price = model.predict(features_scaled)[0]
    return f"Predicted Apartment Price: {predicted_price:.2f}"

# Function to get user input
def get_valid_input(prompt, valid_options=None, value_type=str):
    while True:
        user_input = input(prompt).strip()
        if valid_options and user_input not in valid_options:
            print(f"Invalid input. Please choose from {valid_options}")
        else:
            try:
                return value_type(user_input)
            except ValueError:
                print(f"Invalid input. Please enter a valid {value_type.__name__}.")

# Get user input
districts = list(joblib.load("label_encoders.pkl")["District"].classes_)  # Convert NumPy array to list
district = get_valid_input(f"Enter district name ({', '.join(districts)}): ", valid_options=districts)

area = get_valid_input("Enter area in sqft: ", value_type=int)
bedrooms = get_valid_input("Enter number of bedrooms: ", value_type=int)
bathrooms = get_valid_input("Enter number of bathrooms: ", value_type=int)
stories = get_valid_input("Enter number of stories: ", value_type=int)
mainroad = get_valid_input("Is it on the main road? (Yes/No): ", valid_options=["Yes", "No"])
guestroom = get_valid_input("Does it have a guestroom? (Yes/No): ", valid_options=["Yes", "No"])
basement = get_valid_input("Does it have a basement? (Yes/No): ", valid_options=["Yes", "No"])
hotwater = get_valid_input("Does it have hot water heating? (Yes/No): ", valid_options=["Yes", "No"])
aircon = get_valid_input("Does it have air conditioning? (Yes/No): ", valid_options=["Yes", "No"])
parking = get_valid_input("Enter number of parking spaces: ", value_type=int)
prefarea = get_valid_input("Is it in a preferred area? (Yes/No): ", valid_options=["Yes", "No"])
furnishing = get_valid_input("Furnishing Status (Furnished/Semi-Furnished/Unfurnished): ", valid_options=["Furnished", "Semi-Furnished", "Unfurnished"])

# Predict price
result = predict_price(district, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwater, aircon, parking, prefarea, furnishing)
print(result)
