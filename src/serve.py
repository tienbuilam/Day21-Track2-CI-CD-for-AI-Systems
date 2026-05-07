from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Model da duoc tai xuong tu S3: s3://{S3_BUCKET}/{S3_MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


LABELS = {0: "thap", 1: "trung_binh", 2: "cao"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    pred = int(model.predict([req.features])[0])
    return {"prediction": pred, "label": LABELS.get(pred, "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
