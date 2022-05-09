import os
import pickle

import pandas as pd
from fastapi import FastAPI

from pydantic import BaseModel, Field

from starter.starter.ml.model import  train_model, inference, compute_model_metrics
from starter.starter.ml.data import process_data


# Load models and encoder
with open("./starter/model/model.pkl", "rb") as file:
    model = pickle.load(file)

with open("./starter/model/encoder.pkl", "rb") as file:
    encoder = pickle.load(file)

with open("./starter/model/label_binarizer.pkl", "rb") as file:
    lb = pickle.load(file)

# Enable dvc on eroku
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Run FastAPI
app = FastAPI()

class RequestData(BaseModel):
    age: int = Field(..., example=60)
    workclass: str = Field(..., example="Local-gov")
    fnlgt: int = Field(..., example=98350)
    education: str = Field(..., example="Some-college")
    education_num: int = Field(..., alias="education-num", example=10)
    marital_status: str = Field(
        ..., alias="marital-status", example="Married-civ-spouse"
    )
    occupation: str = Field(..., example="Other-service")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="Asian-Pac-Islander")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=0)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=60)
    native_country: str = Field(
        ..., alias="native-country", example="Philippines"
    )

@app.get("/")
async def get_test():
    return {"message": "Hello world!"}


@app.post("/predict")
async def predict(request_data: RequestData):
    df_data = pd.DataFrame.from_dict([request_data.dict(by_alias=True)])

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Prepare data for inference
    X, _, _, _ = process_data(
        df_data,
        categorical_features=categorical_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    # Run model
    predictions = inference(model, X)

    # Get prediction string response
    if pred[0] == 1:
        predictions = ">50K"
    else:
        predictions = "<=50K"

    return {"predictions": predictions}