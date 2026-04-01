import os
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from predict import predict

app = FastAPI(title="Car Price Prediction API")

# Setup static directory
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

# Expose static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")

BRANDS = sorted([
    "AUDI", "AUSTIN", "BAJAJ", "BMW", "CHERY", "CHEVROLET", "DAEWOO", "DAIHATSU", "DATSUN", "FIAT", "FORD", "FOTON", "HONDA", "HYUNDAI", "ISUZU", "JAGUAR", "JONWAY", "KIA", "LEXUS", "MAHINDRA", "MAZDA", "MERCEDES-BENZ", "MG", "MICRO", "MINI", "MITSUBISHI", "MORRIS", "NISSAN", "OPEL", "PERODUA", "PEUGEOT", "PROTON", "RENAULT", "SEAT", "SKODA", "SUBARU", "SUZUKI", "TATA", "TOYOTA", "VOLKSWAGEN", "VOLVO", "WILLYS", "ZOTYE"
])

# Read dataset once during startup to populate models
try:
    df = pd.read_csv(os.path.join(current_dir, "car_price_dataset_final.csv"))
    # Precompute models dictionary
    models_by_brand = {}
    for brand in BRANDS:
        models = df[df["Brand"] == brand]["Model"].unique()
        models_by_brand[brand] = sorted([str(m) for m in models])
except Exception as e:
    print(f"Warning: Could not load car dataset for metadata: {e}")
    models_by_brand = {brand: [] for brand in BRANDS}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open(os.path.join(static_dir, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/metadata")
async def get_metadata():
    return JSONResponse({
        "brands": BRANDS,
        "models_by_brand": models_by_brand
    })

@app.post("/api/predict")
async def make_prediction(request: Request):
    input_dict = await request.json()
    result = predict(input_dict)
    return JSONResponse(result)

if __name__ == "__main__":
    import uvicorn
    # Allows running exactly the same way (python app.py)
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
