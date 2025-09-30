# Complete ML Pipeline Project

A comprehensive, production-ready machine learning pipeline that demonstrates the entire ML lifecycle from data preprocessing to deployment and monitoring.

## 🎯 Project Overview

This project implements a complete ML pipeline for customer churn prediction, showcasing best practices in:
- Data preprocessing and feature engineering
- Model training and hyperparameter optimization
- Model evaluation and selection
- API development and deployment
- Monitoring and maintenance

### Key Features
- **Automated Data Pipeline**: ETL processes with validation
- **Ensemble Models**: Combining multiple algorithms
- **Real-time Predictions**: Low-latency API endpoints
- **Model Monitoring**: Performance tracking and drift detection
- **Scalable Architecture**: Containerized microservices

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Source   │    │  Preprocessing  │    │  Model Training │
│   (Database/    │───▶│   & Feature     │───▶│  & Optimization │
│    Files/API)   │    │   Engineering   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Model Storage  │    │  API Gateway    │    │  Monitoring &   │
│   (MLflow/S3)   │◀───│   & Serving     │◀───│  Alerting       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Docker
- PostgreSQL
- Redis (optional, for caching)

### Setup

1. **Clone and navigate**:
```bash
cd 01_complete_ml_pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment**:
```bash
cp .env.example .env
# Edit .env with your database and API keys
```

4. **Initialize database**:
```bash
python scripts/setup_database.py
```

5. **Run data preprocessing**:
```bash
python scripts/preprocess_data.py
```

6. **Train models**:
```bash
python scripts/train_models.py
```

7. **Start API server**:
```bash
python src/inference/api.py
```

## 📊 Project Structure

```
01_complete_ml_pipeline/
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── config/
│   ├── model_config.yaml      # Model parameters
│   ├── preprocessing_config.yaml # Data processing config
│   └── api_config.yaml        # API configuration
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # Data preprocessing pipeline
│   │   ├── feature_engineering.py # Feature engineering
│   │   ├── data_loader.py     # Data loading utilities
│   │   └── data_validation.py # Data validation
│   ├── models/
│   │   ├── ensemble_model.py  # Ensemble model implementation
│   │   ├── model_registry.py  # Model management
│   │   ├── base_model.py      # Base model class
│   │   └── hyperparameter_tuning.py # Hyperparameter optimization
│   ├── training/
│   │   ├── train.py           # Main training script
│   │   ├── cross_validation.py # Cross-validation
│   │   └── model_selection.py # Model selection
│   ├── inference/
│   │   ├── api.py             # FastAPI application
│   │   ├── predict.py         # Prediction logic
│   │   ├── batch_processor.py # Batch processing
│   │   └── model_loader.py    # Model loading utilities
│   └── utils/
│       ├── monitoring.py      # Monitoring utilities
│       ├── logging.py         # Logging configuration
│       ├── metrics.py         # Performance metrics
│       └── database.py        # Database utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── tests/
│   ├── test_data_processing.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_integration.py
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   └── Dockerfile.training
│   ├── kubernetes/
│   │   ├── api-deployment.yaml
│   │   ├── training-job.yaml
│   │   └── service.yaml
│   └── cloud/
│       ├── aws/
│       │   ├── ecs-task-definition.json
│       │   └── cloudformation-template.yaml
│       └── gcp/
│           └── cloud-run.yaml
├── data/
│   ├── raw/
│   │   └── customer_churn.csv  # Sample dataset
│   ├── processed/
│   └── sample/
├── scripts/
│   ├── setup_database.py
│   ├── preprocess_data.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── deploy.sh
└── docs/
    ├── api_documentation.md
    ├── deployment_guide.md
    └── troubleshooting.md
```

## 🔧 Configuration

### Model Configuration (`config/model_config.yaml`)
```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5

  xgboost:
    n_estimators: 200
    learning_rate: 0.1
    max_depth: 6

  logistic_regression:
    C: 1.0
    solver: 'liblinear'

ensemble:
  method: 'weighted_average'
  weights: [0.3, 0.5, 0.2]

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  cross_validation_folds: 5
```

### API Configuration (`config/api_config.yaml`)
```yaml
api:
  host: '0.0.0.0'
  port: 8000
  workers: 4
  reload: true

monitoring:
  enable_metrics: true
  metrics_port: 8001

cache:
  enable_redis: false
  redis_url: 'redis://localhost:6379'
  cache_ttl: 3600
```

## 📈 Usage Examples

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "customer_id": "12345",
       "age": 35,
       "tenure": 24,
       "monthly_charges": 75.50,
       "total_charges": 1812.00,
       "contract_type": "Month-to-month",
       "payment_method": "Electronic check"
     }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "customers": [
         {
           "customer_id": "12345",
           "age": 35,
           "tenure": 24,
           "monthly_charges": 75.50,
           "total_charges": 1812.00
         },
         {
           "customer_id": "67890",
           "age": 28,
           "tenure": 12,
           "monthly_charges": 95.00,
           "total_charges": 1140.00
         }
       ]
     }'
```

### Python SDK Usage

```python
from src.inference.predict import ChurnPredictor

# Initialize predictor
predictor = ChurnPredictor(model_path="models/ensemble_model.pkl")

# Single prediction
customer_data = {
    "age": 35,
    "tenure": 24,
    "monthly_charges": 75.50,
    "total_charges": 1812.00
}

result = predictor.predict(customer_data)
print(f"Churn probability: {result['probability']:.2f}")
print(f"Prediction: {result['prediction']}")

# Batch prediction
customers = [customer_data1, customer_data2, ...]
results = predictor.predict_batch(customers)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_models.py
pytest tests/test_api.py

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🚀 Deployment

### Docker Deployment

1. **Build Docker image**:
```bash
docker build -f deployment/docker/Dockerfile.api -t churn-predictor-api .
```

2. **Run with Docker Compose**:
```bash
docker-compose up -d
```

### Kubernetes Deployment

1. **Apply Kubernetes manifests**:
```bash
kubectl apply -f deployment/kubernetes/
```

2. **Check deployment status**:
```bash
kubectl get pods -l app=churn-predictor
```

### Cloud Deployment

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com
docker build -t churn-predictor .
docker tag churn-predictor:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-predictor:latest
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/churn-predictor:latest
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy churn-predictor \
  --image gcr.io/your-project/churn-predictor \
  --platform managed \
  --region us-central1
```

## 📊 Monitoring

### Performance Metrics
- **Prediction Latency**: < 100ms p99
- **Throughput**: 1000+ requests/second
- **Accuracy**: 85-90% depending on dataset
- **Model Drift Detection**: Automated alerts
- **Data Quality Monitoring**: Real-time validation

### Monitoring Dashboard
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **MLflow**: Experiment tracking
- **Custom Alerts**: Email/Slack notifications

## 🛠️ Development

### Adding New Models

1. **Create model class**:
```python
# src/models/new_model.py
from .base_model import BaseModel

class NewModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model

    def train(self, X, y):
        # Training logic
        pass

    def predict(self, X):
        # Prediction logic
        pass
```

2. **Register model**:
```python
# src/models/model_registry.py
from .new_model import NewModel

MODEL_REGISTRY = {
    'new_model': NewModel,
    # ... other models
}
```

3. **Update configuration**:
```yaml
# config/model_config.yaml
models:
  new_model:
    param1: value1
    param2: value2
```

### Adding New Features

1. **Update preprocessing**:
```python
# src/data/feature_engineering.py
def create_new_feature(df):
    # Feature engineering logic
    return df
```

2. **Update validation**:
```python
# src/data/data_validation.py
def validate_new_feature(feature_data):
    # Validation logic
    pass
```

## 📚 Documentation

- [API Documentation](docs/api_documentation.md)
- [Deployment Guide](docs/deployment_guide.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Model Documentation](docs/model_documentation.md)

## 🔗 Related Projects

- [Ensemble Methods Project](../02_ensemble_methods/)
- [Real-time Prediction System](../03_realtime_predictions/)
- [Model Monitoring Project](../04_model_monitoring/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Update documentation
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset: Customer Churn Prediction Dataset
- Libraries: scikit-learn, XGBoost, FastAPI, MLflow
- Inspiration: Best practices from ML engineering community

---

*Last Updated: September 2025*