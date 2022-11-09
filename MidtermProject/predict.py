from flask import Flask, request, jsonify
import pickle
import numpy as np
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer


app = Flask('predict_rent')

# Load the XGBoost model
with open("bin/xgboost_brazil_house.bin", 'rb') as f:
    model = pickle.load(f)

# Load the RandomForest model
with open("bin/sklearn_rf_brazil_house.bin", 'rb') as f:
    model_rf = pickle.load(f)

# Load the Dictionary Vectorizer   
with open("bin/dictvec.bin",'rb') as dv:
    dict_vectorizer = pickle.load(dv)


@app.route('/predict', methods=['POST'])
def predict():
	customer = request.get_json() # convert JSON to Python dictionary.
	features = dict_vectorizer.transform([customer])
	dfeats = xgb.DMatrix(features)
	y_pred = model.predict(dfeats)
	act_brl = np.expm1(y_pred)

	y_pred2 = model_rf.predict(features)
	act_brl2 = np.expm1(y_pred2)

	# add rent amounts in BRL to a python dictionary
	result = {'rent_xgb' : int(act_brl), 'rent_rf': int(act_brl2)}
	# return json 
	return jsonify(result)


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port='9696')