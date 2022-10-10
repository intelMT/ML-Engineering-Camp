import requests

URL = "http://localhost:9696/predict"

client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

response = requests.post(url=URL, json=client).json()

print("The probability of customer to churn: ", round(response['probability'],3))

if response['churn'] == True:
	print("sending 25% discount.")
else:
	print("not sending 25% discount")