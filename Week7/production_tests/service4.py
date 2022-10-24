import bentoml
import numpy as np
from bentoml.io import NumpyNdarray


model_ref = bentoml.sklearn.get('mlzoomcamp_homework:jsi67fslz6txydu5')

model_runner = model_ref.to_runner()

svc = bentoml.Service("mlzoomcamp_sklearn_classifier", runners=[model_runner])

@svc.api(input=NumpyNdarray(shape=(-1, 4), enforce_shape=True, dtype=np.float64, enforce_dtype=True), output=NumpyNdarray())
async def classify(app_data):
    array = np.array(app_data)
    prediction = await model_runner.predict.async_run(array)
    print('\n')
    print(prediction)
    print('\n')
    return prediction
