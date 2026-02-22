# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from PIL import UnidentifiedImageError

from .predictor import predictor

app = FastAPI(title="BMW_2 Bike Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/api/bmw/predict")
async def predict_bmw(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        pred = predictor.predict_bytes(data)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Upload a valid JPG/PNG/WebP.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(e).__name__}: {e}")

    payload = {
        "status": pred.status,
        "label": pred.label,
        "confidence": pred.confidence,
        "margin": pred.margin,
        "reason": pred.reason,
        "per_model": pred.per_model,
    }
    return JSONResponse(content=jsonable_encoder(payload))