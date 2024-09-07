import joblib
import os

# Check if the file exists
file_path = 'Models/home_price_model.pickle'
if os.path.exists(file_path):
    with open(file_path, 'rb') as f:
        model = joblib.load(f)
    print(model.predict([[3000, 3, 40]]))
else:
    print(f"File not found: {file_path}")
    print(os.getcwd())