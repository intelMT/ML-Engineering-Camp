# Brazilian Houses Rent Prediction

## Problem Description

In this ML project, Brazilian Houses data is used to predict rent prices. Being able to predict a house based on its descriptors such as number of rooms, area etc. is useful for both renters, landlords and realtors. This repo shows the training of two ML models (XGBoost and Random Forest) using a clean OpenML dataset, which sourced from Kaggle.

## Files

### Model and data files:
Model files and embedding binaries are stored in `bin` folder. <br>
Dataset csv file is stored in `data` folder. More information about dataset in `notebook.ipynb` 

### Training files:

`notebook.ipynb` is the Jupyter file used to train the data. <br>
A `train.py` file is also provided for those who don't like notebooks. <br>
So, you can also train the same models with `pipenv run python train.py`

## Dependency Management

To train and run all the scripts, you should first install pipenv as: <br>
`pip install pipenv` <br>

Then, in the directory for the project, run:
`pipenv install`

This will create a virtual environment and install all needed libaries from `Pipfile` and `Pipfile.lock` provided in this repo. <br>
Also, `bentofile.yaml` and `requirements.txt` file is provided if you want to use different dependency management software.

## Containerization and Docker Image
`Dockerfile` is provided to create a docker image to both train and serve the ML service using gunicorn on a linux/amd64 image. <br>
Also, the image is pushed to the [Docker Hub](https://hub.docker.com/repository/docker/intelmt/zoomcamp-midterm)

## Deployment to pythonanywhere
The Random Forest prediction model is deployed to [https://drmtan.pythonanywhere.com/predict](https://drmtan.pythonanywhere.com/predict) <br>
To use interact with this webapp, use `test_deploy_predict.ipynb` notebook. <br>
For this deployment, only dependencies you need are Flask, Scikit-learn and numpy to get the actual rent values in Brazilian Real (BRL).
You need to install these dependencies on your cloud web-app serving environment <br>
Here, I'll link [a video](#) about how I've deployed my Flask webapp to Pythonanywhere.com.
