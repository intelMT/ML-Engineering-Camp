from flask import Flask, request, jsonify
import pickle


app = Flask('predict_churn')

with open("data/model1.bin", 'rb') as f_model:
    model = pickle.load(f_model)
    
with open("data/dv.bin",'rb') as dv:
    dict_vectorizer = pickle.load(dv)


@app.route('/predict', methods=['POST'])
def predict():
	customer = request.get_json() # convert JSON to Python dictionary.
	client = dict_vectorizer.transform([customer])
	y_pred = model.predict_proba(client)[:,1]
	churn = y_pred >= 0.5

	result = {
		"probability" : float(y_pred),
		"churn": bool(churn)
	}
	return jsonify(result)


if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0', port='9696')

