import bentoml
import numpy as np

from bentoml.io import JSON

model_ref = bentoml.xgboost.get("xgboost_brazil_house:latest")

dv = model_ref.custom_objects["dictVec"]
model_runner = model_ref.to_runner()

svc = bentoml.Service("brazil_rent_predictor", runners = [model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(customer_data):
	features = dv.transform(customer_data)
	prediction = model_runner.predict.run(features)
	total = np.expm1(prediction)
	return {"predicted_rent_brl": int(total)}