import requests

URL = "http://localhost:9696/predict"

customer = { "city": "Sao Paolo", "area": 95,"rooms": 2,"bathroom":1 , "animal": 1 ,"furniture": 0,
			"parking_spaces": 1,"hoa_brl": 560,"property_tax_brl":40 ,"fire_insurance_brl":37 }

response = requests.post(URL, json=customer).json()

print("The rent prediction based on XGBoost: ", response['rent_xgb'] , "BRL")

print("The rent prediction based on Random Forest: ", response['rent_rf'], "BRL")