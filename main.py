from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import os
from typing import Optional

app = FastAPI(title="Ticket Classify Service", version="1.0.0")

model = None


class TicketRequest(BaseModel):
    day_of_week_num: float
    company_size_cat: float
    industry_cat: float
    customer_tier_cat: float
    org_users: float
    region_cat: float
    past_30d_tickets: float
    past_90d_incidents: float
    product_area_cat: float
    booking_channel_cat: float
    reported_by_role_cat: float
    customers_affected: float
    error_rate_pct: float
    downtime_min: float
    payment_impact_flag: float
    security_incident_flag: float
    data_loss_flag: float
    has_runbook: float
    customer_sentiment_cat: float
    description_length: float


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: Optional[float] = None


@app.on_event("startup")
async def load_model():
    global model
    model_path = "pipeline.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise Exception(f"Model file {model_path} not found")


@app.get("/")
async def root():
    return {"message": "Welcome to Ticket Classify Service"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: TicketRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert request to list format expected by model
        feature_vector = [
            request.day_of_week_num, request.company_size_cat, request.industry_cat,
            request.customer_tier_cat, request.org_users, request.region_cat,
            request.past_30d_tickets, request.past_90d_incidents, request.product_area_cat,
            request.booking_channel_cat, request.reported_by_role_cat, request.customers_affected,
            request.error_rate_pct, request.downtime_min, request.payment_impact_flag,
            request.security_incident_flag, request.data_loss_flag, request.has_runbook,
            request.customer_sentiment_cat, request.description_length
        ]
        
        prediction = model.predict([feature_vector])[0]
        
        # If model has predict_proba method, get confidence
        confidence = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba([feature_vector])[0]
            confidence = float(max(probabilities))

        prediction_labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        return PredictionResponse(prediction=prediction, confidence=confidence, prediction_label=prediction_labels[prediction])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


