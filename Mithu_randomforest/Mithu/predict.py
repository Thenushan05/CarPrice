
"""
Production Prediction Function for Car Price Model
==================================================
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

# Load model and metadata
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(MODEL_DIR, 'car_price_model.joblib'))

with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

REQUIRED_FEATURES = [
    'Brand', 'Engine (cc)', 'Gear', 'Fuel Type', 'Millage(KM)',
    'Leasing', 'AIR CONDITION', 'POWER STEERING', 'POWER MIRROR',
    'POWER WINDOW', 'Car_Age'
]


def validate_input(input_dict: dict) -> tuple:
    """Validate input dictionary and return errors if any."""
    errors = []
    
    # Check for required features
    for feature in REQUIRED_FEATURES:
        if feature not in input_dict:
            errors.append(f"Missing required feature: '{feature}'")
    
    # Type validation
    numeric_features = ['Engine (cc)', 'Millage(KM)', 'Car_Age']
    for feat in numeric_features:
        if feat in input_dict:
            try:
                float(input_dict[feat])
            except (ValueError, TypeError):
                errors.append(f"'{feat}' must be numeric, got: {type(input_dict[feat]).__name__}")
    
    # Value range validation
    if 'Car_Age' in input_dict:
        try:
            age = float(input_dict['Car_Age'])
            if age < 0 or age > 100:
                errors.append(f"'Car_Age' should be between 0-100, got: {age}")
        except:
            pass
    
    if 'Millage(KM)' in input_dict:
        try:
            mileage = float(input_dict['Millage(KM)'])
            if mileage < 0:
                errors.append(f"'Millage(KM)' cannot be negative, got: {mileage}")
        except:
            pass
    
    return len(errors) == 0, errors


def predict(input_dict: dict) -> dict:
    """
    Predict car price from raw input dictionary.
    
    Parameters:
    -----------
    input_dict : dict
        Dictionary containing car features. Required keys:
        - Brand: str (e.g., 'TOYOTA', 'BMW')
        - Engine (cc): float (e.g., 1500.0)
        - Gear: str ('Automatic' or 'Manual')
        - Fuel Type: str ('Petrol', 'Diesel', 'Hybrid', 'Electric')
        - Millage(KM): float (e.g., 50000.0)
        - Leasing: str or int ('0', '1', or 'Ongoing Lease')
        - AIR CONDITION: int (0 or 1)
        - POWER STEERING: int (0 or 1)
        - POWER MIRROR: int (0 or 1)
        - POWER WINDOW: int (0 or 1)
        - Car_Age: int (e.g., 5)
    
    Returns:
    --------
    dict: {'predicted_price': float, 'status': 'success'} or
          {'error': str, 'status': 'error'}
    
    Example:
    --------
    >>> result = predict({
    ...     'Brand': 'TOYOTA',
    ...     'Engine (cc)': 1500.0,
    ...     'Gear': 'Automatic',
    ...     'Fuel Type': 'Petrol',
    ...     'Millage(KM)': 50000.0,
    ...     'Leasing': 0,
    ...     'AIR CONDITION': 1,
    ...     'POWER STEERING': 1,
    ...     'POWER MIRROR': 1,
    ...     'POWER WINDOW': 1,
    ...     'Car_Age': 5
    ... })
    >>> print(result)
    {'predicted_price': 85.5, 'status': 'success'}
    """
    
    # Validate input
    is_valid, errors = validate_input(input_dict)
    if not is_valid:
        return {
            'status': 'error',
            'errors': errors
        }
    
    try:
        # Create DataFrame from input
        df = pd.DataFrame([input_dict])
        
        # Engineer features (same as training)
        df['Mileage_Per_Year'] = df['Millage(KM)'] / (df['Car_Age'] + 1)
        df['Engine_Category'] = pd.cut(
            df['Engine (cc)'],
            bins=[0, 1000, 1500, 2000, 3000, np.inf],
            labels=['Small', 'Medium', 'Large', 'XLarge', 'Luxury']
        )
        df['Comfort_Score'] = (
            df['AIR CONDITION'].astype(float) + 
            df['POWER STEERING'].astype(float) + 
            df['POWER MIRROR'].astype(float) + 
            df['POWER WINDOW'].astype(float)
        )
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # If log-transformed during training, convert back
        if metadata.get('use_log_transform', False):
            prediction = np.expm1(prediction)
        
        return {
            'status': 'success',
            'predicted_price': round(float(prediction), 2)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'errors': [str(e)]
        }


# Example usage
if __name__ == '__main__':
    # Test prediction
    test_input = {
        'Brand': 'TOYOTA',
        'Engine (cc)': 1500.0,
        'Gear': 'Automatic',
        'Fuel Type': 'Petrol',
        'Millage(KM)': 50000.0,
        'Leasing': 0,
        'AIR CONDITION': 1,
        'POWER STEERING': 1,
        'POWER MIRROR': 1,
        'POWER WINDOW': 1,
        'Car_Age': 5
    }
    
    result = predict(test_input)
    print(f"Prediction Result: {result}")
