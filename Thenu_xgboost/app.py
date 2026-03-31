import streamlit as st
import pandas as pd
import requests
import json

# Dropdown options (from CSV and model)
BRANDS = [
    "AUDI", "AUSTIN", "BAJAJ", "BMW", "CHERY", "CHEVROLET", "DAEWOO", "DAIHATSU", "DATSUN", "FIAT", "FORD", "FOTON", "HONDA", "HYUNDAI", "ISUZU", "JAGUAR", "JONWAY", "KIA", "LEXUS", "MAHINDRA", "MAZDA", "MERCEDES-BENZ", "MG", "MICRO", "MINI", "MITSUBISHI", "MORRIS", "NISSAN", "OPEL", "PERODUA", "PEUGEOT", "PROTON", "RENAULT", "SEAT", "SKODA", "SUBARU", "SUZUKI", "TATA", "TOYOTA", "VOLKSWAGEN", "VOLVO", "WILLYS", "ZOTYE"
]

GEARS = ["Automatic", "Manual"]
FUEL_TYPES = ["Diesel", "Electric", "Hybrid", "Petrol"]

# Read models for each brand from CSV
def get_models_for_brand(brand):
    df = pd.read_csv("car_price_dataset_final.csv")
    return sorted(df[df["Brand"] == brand]["Model"].unique())

def main():
    st.title("Car Price Prediction")
    st.write("Enter car details to predict the price.")


    brand = st.selectbox("Brand", BRANDS)
    models = get_models_for_brand(brand)
    model = st.selectbox("Model", models)
    gear = st.selectbox("Gear", GEARS)
    fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
    engine_cc = st.number_input("Engine (cc)", min_value=500, max_value=10000, value=1000)
    millage = st.number_input("Millage (KM)", min_value=0, max_value=1000000, value=50000)
    car_age = st.number_input("Car Age (years)", min_value=0, max_value=50, value=5)
    air_condition = st.selectbox("Air Condition", [0, 1])
    power_steering = st.selectbox("Power Steering", [0, 1])
    power_mirror = st.selectbox("Power Mirror", [0, 1])
    power_window = st.selectbox("Power Window", [0, 1])
    condition = st.selectbox("Condition", ["NEW", "USED"])
    leasing = st.selectbox("Leasing", ["0", "Ongoing Lease"])

    if st.button("Predict Price"):
        input_dict = {
            "Brand": brand,
            "Model": model,
            "Engine (cc)": engine_cc,
            "Gear": gear,
            "Fuel Type": fuel_type,
            "Millage(KM)": millage,
            "Car_Age": car_age,
            "AIR CONDITION": air_condition,
            "POWER STEERING": power_steering,
            "POWER MIRROR": power_mirror,
            "POWER WINDOW": power_window,
            "Condition": condition,
            "Leasing": leasing
        }
        # Local prediction (import predict.py)
        from predict import predict
        result = predict(input_dict)
        if result["status"] == "success":
            st.success(f"Predicted Price: {result['predicted_price']}")
        else:
            st.error(f"Error: {result['errors']}")

if __name__ == "__main__":
    main()
