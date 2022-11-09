
FROM python:3.9.12-slim

RUN pip install pipenv==2022.10.4

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["dictvec.bin", "xgboost_brazil_house.bin", "./"]

COPY ["sklearn_rf_brazil_house.bin", "predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]