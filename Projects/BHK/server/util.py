import json
import pickle
import pandas as pd
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
	try:
		loc_index = __data_columns.index(location.lower())
	except:
		loc_index = -1
	
	x=np.zeros(len(__data_columns))
	x[0] = sqft
	x[1] = bath
	x[2] = bhk
	if loc_index >= 0:
		x[loc_index] = 1
	
	return round(__model.predict([x])[0], 2)

def get_location_names():
	return __locations

def load_saved_artifacts():
	print("loading saved artifacts...start")
	global __data_columns
	global __locations
	global __model

	with open(r"Projects\BHK\server\artifacts\columns.json", "r") as f:
		__data_columns = json.load(f)['data_columns']
		__locations = __data_columns[3:]

	with open(r"Projects\BHK\server\artifacts\bangalore_house_price_prediction_model.pkl", "rb") as f:
		__model = pickle.load(f)

	print("loading saved artifacts...done")

if __name__ == "__main__":
	load_saved_artifacts()
	print(get_location_names())
	x = get_estimated_price('Rajaji Nagar', 1000, 3, 3)
	print(x)
	print(type(x))