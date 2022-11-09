# Brazilian Houses Rent Prediction

## Problem Description
In this Machine Learning project, Brazilian Houses data is used to predict rent prices in 2020 in 5 Brazilian cities. Being able to predict a house based on its descriptors such as number of rooms, area etc. is useful for both renters, landlords and realtors. The renters or property-buyers can have an idea what should be an average rent value based on house descriptors and calculate their budget or ROI. This repo shows the training of two ML models (XGBoost and Random Forest) using a clean OpenML dataset, which sourced from Kaggle.

Dataset provides information such as the city and the area of house/apartment; the number of rooms and bathrooms; Is house furnitured?; Is house pet-friendly? <br>
Also, there are monetary descriptors such as tax information (property, home owners association), fire insurance etc. available. <br>
More info available at [Kaggle repo](https://www.kaggle.com/datasets/rubenssjr/brasilian-houses-to-rent?select=houses_to_rent_v2.csv)

As inflation soars, the prices keep changing, this applies to the rent values. This is part of what is called "concept drift." The dataset is from 2020, so the rent values is expected to be off real market values now. I think obtained predictions can further be adjusted for inflation, or specific inflation for civil building market, and any changes in taxes.

## Files
### Model and data files:
Model files and embedding binaries are stored in `bin` folder. <br>
Dataset csv file is stored in `data` folder. More information about dataset in `notebook.ipynb` <br>
It is taken from [OpenML ID 44062 Brazilian_houses v.8](https://www.openml.org/search?type=data&status=active&sort=runs&id=44062)

### Training files:
`notebook.ipynb` is the Jupyter file used to train the data. <br>
A `train.py` file is also provided for those who don't like notebooks. <br>
So, you can also train the same models with `pipenv run python train.py`

## Testing Locally
In your `pipenv shell` run `python predict.py` and your Flask application runs in debug mode. <br>
You can then run `python test.py` in another cli OR run `test_predict.ipynb` to see that the Flask application runs without any problem.

## Dependency Management
To train and run all the scripts, you should first install pipenv as: <br>
`pip install pipenv` <br>

Then, in the directory for the project, run:
`pipenv install`

This will create a virtual environment and install all needed libaries from `Pipfile` and `Pipfile.lock` provided in this repo. <br>
Also, `bentofile.yaml` and `requirements.txt` files are constructed and provided if you want to use different dependency management software.

## Containerization and Docker Image
`Dockerfile` is provided to create a docker image to both train and serve the ML service using gunicorn on a linux/amd64 image. <br>
Also, the image is pushed to the [Docker Hub](https://hub.docker.com/repository/docker/intelmt/zoomcamp-midterm)

## Model Deployment
`docker build -t zoomcamp-midterm .` is used to build the docker image with tag "zoomcamp-midterm". <br>
Alternatively, you can pull the model from dockerhub link provided above. <br>
Then, Start running the docker image by: <br>
`docker run -it --rm -p 9696:9696 zoomcamp-midterm:latest` on the cli.
Now, the Docker is running. Use `test_predict.ipynb` notebook to see that it works and gives the same result for the input. The same notebook works because we use the same port number and either flask on host machine or docker container listens to requests at that port number, that is the second one in -p tag `xxxx:9696`.

## Cloud Deployment to Pythonanywhere.com
The Random Forest prediction model is deployed to [https://drmtan.pythonanywhere.com/predict](https://drmtan.pythonanywhere.com/predict) <br>
To interact with this webapp, use `test_deploy_predict.ipynb` notebook. <br>
For this deployment, only dependencies you need are Flask, Scikit-learn and numpy to get the actual rent values in Brazilian Real (BRL).
You need to install these dependencies on your cloud web-app serving environment either using pip or pipenv. <br>
Here, I explain in a [YouTube video](https://www.youtube.com/watch?v=yeefD9OaxHM) about how I've deployed my Flask webapp to Pythonanywhere.com. <br>
The application on the web is still running and will run for a while, all that is needed to test it out is to run `test_deploy_predict.ipynb` notebook. <br>
Thanks for reading!
