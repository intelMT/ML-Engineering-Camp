import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

class UserProfile(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int


model_ref = bentoml.xgboost.get('credit_risk_model:latest')

model_runner = model_ref.to_runner()
dv = model_ref.custom_objects['dictVectorizer']

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=UserProfile), output=JSON())
def classify(user_profile):
    vector = dv.transform(user_profile.dict())
    prediction = model_runner.predict.run(vector)
    print(prediction)
    if prediction > 0.5:
        return {"status": "Approved"}
    elif prediction >0.23:
        return {"status": "Maybe"}
    else:
        return {"status": "Declined"}
