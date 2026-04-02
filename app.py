import os, sys
from networksecurity.constants.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
import pymongo
import networksecurity.pipeline.training_pipeline as training_pipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import load_object

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run as app_run
from fastapi.responses import Response
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

import pandas as pd

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)


database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", tags=["Authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train", tags=["Train"])
async def train_route(request: Request):
    try:
        training_pipeline_obj = training_pipeline.TrainingPipeline()
        training_pipeline_obj.run_pipeline()
        return Response(content="Training successful !!", media_type="text/plain")
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
@app.post("/predict", tags=["Predict"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        preprocessor = load_object(file_path="final_models/preprocessor.pkl")
        final_model = load_object(file_path="final_models/final_model.pkl")
        network_model = NetworkModel(preprocessor=preprocessor, model=final_model)
        print(df.iloc[0])
        prediction = network_model.predict(df)
        print(prediction)
        df['predicted_column'] = prediction
        print(df["predicted_column"])
        df.to_csv("predicted_output/output.csv", index=False)
        table_html = df.to_html(classes="table table-striped")
        return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
    except Exception as e:
        raise NetworkSecurityException(e, sys)

if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)