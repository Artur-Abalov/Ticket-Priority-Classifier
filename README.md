# Ticket Classification Service

A FastAPI-based machine learning service that predicts ticket priority levels (Low, Medium, High) using XGBoost. The service analyzes various ticket features including company information, impact metrics, and historical data to provide accurate priority classifications.

## Features

- **REST API**: FastAPI-based endpoints for ticket classification
- **Machine Learning**: XGBoost classifier with hyperparameter optimization
- **Priority Prediction**: Classifies tickets into Low (0), Medium (1), or High (2) priority
- **Probability Scores**: Get prediction confidence scores for all priority levels
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Health Monitoring**: Built-in health check endpoints
- **Model Training**: Automated model training with validation and model comparison

## API Endpoints

### Core Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check endpoint
- `POST /api/v1/predict` - Get ticket priority prediction
- `POST /api/v1/predict_proba` - Get prediction probabilities

### Prediction Request Format

```json
{
  "day_of_week_num": 1.0,
  "company_size_cat": 2.0,
  "industry_cat": 3.0,
  "customer_tier_cat": 1.0,
  "org_users": 1500.0,
  "region_cat": 2.0,
  "past_30d_tickets": 5.0,
  "past_90d_incidents": 2.0,
  "product_area_cat": 1.0,
  "booking_channel_cat": 2.0,
  "reported_by_role_cat": 3.0,
  "customers_affected": 100.0,
  "error_rate_pct": 15.5,
  "downtime_min": 30.0,
  "payment_impact_flag": 1.0,
  "security_incident_flag": 0.0,
  "data_loss_flag": 0.0,
  "has_runbook": 1.0,
  "customer_sentiment_cat": 2.0,
  "description_length": 250.0
}
```

## Quick Start

### Using Docker (Recommended)

1. **Clone and build**:
   ```bash
   git clone https://github.com/Artur-Abalov/Ticket-Priority-Classifier.git
   cd TicketClassifyService
   docker-compose up --build
   ```

2. **Access the service**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the service**:
   ```bash
   python main.py
   ```

3. **Train a new model** (optional):
   ```bash
   python train.py
   ```

## Feature Descriptions

The model uses the following features for prediction:

| Feature | Description | Range |
|---------|-------------|-------|
| `day_of_week_num` | Day of week | 1-7 |
| `company_size_cat` | Company size category | 1-3 |
| `industry_cat` | Industry category | 1-7 |
| `customer_tier_cat` | Customer tier category | 1-3 |
| `org_users` | Number of organization users | 71-5757 |
| `region_cat` | Region category | 1-3 |
| `past_30d_tickets` | Tickets in past 30 days | 0-19 |
| `past_90d_incidents` | Incidents in past 90 days | 0-12 |
| `product_area_cat` | Product area category | 1-6 |
| `booking_channel_cat` | Booking channel category | 1-4 |
| `reported_by_role_cat` | Reporter role category | 1-5 |
| `customers_affected` | Number of customers affected | 0-5757 |
| `error_rate_pct` | Error rate percentage | 0.0-79.8 |
| `downtime_min` | Downtime in minutes | 0.0-196.0 |
| `payment_impact_flag` | Payment impact flag | 0/1 |
| `security_incident_flag` | Security incident flag | 0/1 |
| `data_loss_flag` | Data loss flag | 0/1 |
| `has_runbook` | Has runbook flag | 0/1 |
| `customer_sentiment_cat` | Customer sentiment category | 0-3 |
| `description_length` | Description length in characters | 20-815 |

## Model Training

The training pipeline includes:

- **Data Loading**: Loads data from SQLite database (`tickets.sqlite`)
- **Feature Engineering**: Removes noisy features and encodes labels
- **Hyperparameter Optimization**: Uses RandomizedSearchCV with 20 iterations
- **Cross-Validation**: 5-fold stratified cross-validation
- **Model Validation**: Compares new model performance with existing model
- **Automatic Saving**: Only saves improved models

### Training Configuration

- **Algorithm**: XGBoost Classifier
- **Scoring**: F1-macro score
- **Test Size**: 30%
- **CV Folds**: 5 (stratified)
- **Random State**: 42 (for reproducibility)

## Technology Stack

- **Python 3.11**
- **FastAPI**: Modern, fast web framework
- **XGBoost**: Gradient boosting framework
- **Scikit-learn**: Machine learning library
- **Pandas**: Data manipulation
- **SQLite**: Database for training data
- **Docker**: Containerization
- **Uvicorn**: ASGI server

## Development

### Project Structure

```
TicketClassifyService/
├── main.py              # FastAPI application
├── train.py             # Model training script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
├── tickets.sqlite      # Training data
├── pipeline.joblib     # Trained model
└── README.md          # This file
```

### Response Examples

**Prediction Response**:
```json
{
  "prediction": 2,
  "prediction_label": "High"
}
```

**Probability Response**:
```json
{
  "probabilities": {
    "Low": 0.1,
    "Medium": 0.2,
    "High": 0.7
  }
}
```

## License

[Add your license information here]