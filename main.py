from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib
import os


model = None


@asynccontextmanager
async def lifespan(app):
    global model
    model_path = "pipeline.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise Exception(f"Model file {model_path} not found")

    yield

app = FastAPI(title="Ticket Classify Service", version="1.0.0",lifespan=lifespan)

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

class ProbabilityResponse(BaseModel):
    probabilities: dict[str, float]





@app.get("/")
async def root():
    return {"message": "Welcome to Ticket Classify Service"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def _get_feature_vector(request: TicketRequest):
    return [
        request.day_of_week_num, request.company_size_cat, request.industry_cat,
        request.customer_tier_cat, request.org_users, request.region_cat,
        request.past_30d_tickets, request.past_90d_incidents, request.product_area_cat,
        request.booking_channel_cat, request.reported_by_role_cat, request.customers_affected,
        request.error_rate_pct, request.downtime_min, request.payment_impact_flag,
        request.security_incident_flag, request.data_loss_flag, request.has_runbook,
        request.customer_sentiment_cat, request.description_length
    ]


# noinspection PyUnreachableCode
@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(request: TicketRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        feature_vector = _get_feature_vector(request)
        prediction = model.predict([feature_vector])[0]
        
        prediction_labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        
        return PredictionResponse(
            prediction=int(prediction), 
            prediction_label=prediction_labels[prediction]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# noinspection PyUnreachableCode
@app.post("/api/v1/predict_proba", response_model=ProbabilityResponse)
async def predict_proba(request: TicketRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        if not hasattr(model, 'predict_proba'):
            raise HTTPException(status_code=400, detail="Model does not support probability prediction")
        
        feature_vector = _get_feature_vector(request)
        probabilities = model.predict_proba([feature_vector])[0]
        
        prediction_labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }
        
        prob_dict = {
            prediction_labels[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        }
        
        return ProbabilityResponse(probabilities=prob_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Probability prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# DATA DESCRIPTION:
# Feature columns for ticket classification model:
# - day_of_week_num: Day of week (1-7)
# - company_size_cat: Company size category (1-3)
# - industry_cat: Industry category (1-7)
# - customer_tier_cat: Customer tier category (1-3)
# - org_users: Number of organization users (71-5757)
# - region_cat: Region category (1-3)
# - past_30d_tickets: Number of tickets in past 30 days (0-19)
# - past_90d_incidents: Number of incidents in past 90 days (0-12)
# - product_area_cat: Product area category (1-6)
# - booking_channel_cat: Booking channel category (1-4)
# - reported_by_role_cat: Reporter role category (1-5)
# - customers_affected: Number of customers affected (0-5757)
# - error_rate_pct: Error rate percentage (0.0-79.8)
# - downtime_min: Downtime in minutes (0.0-196.0)
# - payment_impact_flag: Payment impact flag (0/1)
# - security_incident_flag: Security incident flag (0/1)
# - data_loss_flag: Data loss flag (0/1)
# - has_runbook: Has runbook flag (0/1)
# - customer_sentiment_cat: Customer sentiment category (0-3)
# - description_length: Description length in characters (20-815)
# - priority_cat: Priority category (target variable)
