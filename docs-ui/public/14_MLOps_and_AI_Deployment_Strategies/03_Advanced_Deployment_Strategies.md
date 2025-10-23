---
title: "Mlops And Ai Deployment Strategies - Module 3: Advanced"
description: "## Navigation. Comprehensive guide covering algorithms, optimization, model training, data preprocessing. Part of AI documentation system with 1500+ topics."
keywords: "optimization, algorithms, optimization, model training, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Module 3: Advanced Deployment Strategies

## Navigation
- **← Previous**: [02_LLMops_and_Generative_AI_Operations.md](02_LLMops_and_Generative_AI_Operations.md)
- **→ Next**: [04_Infrastructure_and_Orchestration.md](04_Infrastructure_and_Orchestration.md)
- **↑ Up**: [README.md](README.md)

## Overview

Advanced deployment strategies are crucial for maintaining reliable, scalable, and efficient machine learning systems in production. This module covers various deployment patterns, from batch processing to real-time inference, and provides comprehensive implementations for each approach.

## 1. Batch Prediction

### Scheduled Batch Processing

#### Apache Beam Implementation
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.metrics import Metrics
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class BatchPredictionPipeline:
    def __init__(self, model_path, input_path, output_path, config=None):
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path
        self.config = config or {}
        self.model = self.load_model()
        self.prediction_counter = Metrics.counter(self.__class__, 'predictions')

    def load_model(self):
        """Load model with error handling and versioning"""
        try:
            model = joblib.load(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def run_pipeline(self):
        """Execute the complete batch prediction pipeline"""
        pipeline_options = PipelineOptions(
            runner=self.config.get('runner', 'DirectRunner'),
            project=self.config.get('project', 'your-project'),
            region=self.config.get('region', 'us-central1'),
            temp_location=self.config.get('temp_location', 'gs://your-bucket/temp/')
        )

        with beam.Pipeline(options=pipeline_options) as pipeline:
            # Read and parse input data
            (pipeline
             | 'Read Data' >> beam.io.ReadFromText(self.input_path)
             | 'Parse JSON' >> beam.Map(self.parse_json)
             | 'Filter Valid Records' >> beam.Filter(self.validate_record)
             | 'Make Predictions' >> beam.Map(self.predict_with_error_handling)
             | 'Filter Valid Predictions' >> beam.Filter(lambda x: x['valid'])
             | 'Format Output' >> beam.Map(self.format_prediction)
             | 'Write Results' >> beam.io.WriteToText(
                 self.output_path,
                 file_name_suffix='.json'
             ))

    def parse_json(self, line):
        """Parse JSON line with error handling"""
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            return None

    def validate_record(self, record):
        """Validate input record structure"""
        if not record:
            return False

        required_fields = self.config.get('required_fields', ['customer_id', 'features'])
        return all(field in record for field in required_fields)

    def predict_with_error_handling(self, record):
        """Make prediction with comprehensive error handling"""
        try:
            # Extract features
            features = self.extract_features(record)

            # Make prediction
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0].max()

            # Increment prediction counter
            self.prediction_counter.inc()

            return {
                'customer_id': record['customer_id'],
                'prediction': int(prediction),
                'probability': float(probability),
                'features': features.tolist(),
                'timestamp': datetime.now().isoformat(),
                'model_version': self.config.get('model_version', '1.0.0'),
                'valid': True
            }

        except Exception as e:
            logging.error(f"Prediction error for record {record.get('customer_id', 'unknown')}: {e}")
            return {
                'customer_id': record.get('customer_id', 'unknown'),
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'valid': False
            }

    def extract_features(self, record):
        """Extract and preprocess features"""
        # Convert features to numpy array
        features_dict = record['features']

        # Order features consistently
        feature_order = self.config.get('feature_order', sorted(features_dict.keys()))
        features = [features_dict[feature] for feature in feature_order]

        # Convert to numpy array
        features_array = np.array(features, dtype=np.float32)

        # Apply preprocessing if configured
        if self.config.get('preprocessing_enabled', False):
            features_array = self.apply_preprocessing(features_array)

        return features_array

    def apply_preprocessing(self, features):
        """Apply preprocessing transformations"""
        # Example: Standardization
        if 'scaler' in self.config:
            from sklearn.preprocessing import StandardScaler
            scaler = joblib.load(self.config['scaler'])
            features = scaler.transform(features.reshape(1, -1)).flatten()

        return features

    def format_prediction(self, prediction_result):
        """Format prediction result for output"""
        if prediction_result['valid']:
            return json.dumps({
                'customer_id': prediction_result['customer_id'],
                'prediction': prediction_result['prediction'],
                'probability': prediction_result['probability'],
                'model_version': prediction_result['model_version'],
                'timestamp': prediction_result['timestamp']
            })
        else:
            return json.dumps({
                'customer_id': prediction_result['customer_id'],
                'error': prediction_result['error'],
                'timestamp': prediction_result['timestamp']
            })
```

#### Advanced Airflow DAG for Batch Processing
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago
from airflow.hooks.base_hook import BaseHook
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
import pandas as pd
import joblib
import logging
from datetime import datetime, timedelta
import os
import json

# DAG configuration
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'sla': timedelta(hours=2),
}

dag = DAG(
    'advanced_batch_prediction_pipeline',
    default_args=default_args,
    description='Advanced daily batch prediction pipeline with monitoring',
    schedule_interval='@daily',
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'batch', 'prediction'],
)

# Configuration paths
DATA_PATH = '/data/daily/'
MODEL_PATH = '/models/latest/'
OUTPUT_PATH = '/output/predictions/'
LOG_PATH = '/logs/batch_pipeline/'

def extract_and_validate_data(**context):
    """Extract daily data and perform validation"""
    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    # Input file path
    input_file = os.path.join(DATA_PATH, f'customers_{date_str}.csv')

    try:
        # Load data
        df = pd.read_csv(input_file)
        logging.info(f"Loaded {len(df)} records from {input_file}")

        # Data validation
        validation_results = validate_dataset(df)

        if not validation_results['valid']:
            raise ValueError(f"Data validation failed: {validation_results['errors']}")

        # Store validation results
        validation_file = os.path.join(LOG_PATH, f'validation_{date_str}.json')
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2)

        # Push file path to XCom for next task
        context['ti'].xcom_push(key='input_file', value=input_file)
        context['ti'].xcom_push(key='record_count', value=len(df))

        return {'status': 'success', 'records': len(df)}

    except Exception as e:
        logging.error(f"Data extraction failed: {e}")
        raise

def validate_dataset(df):
    """Validate dataset quality and completeness"""
    errors = []
    warnings = []

    # Check required columns
    required_columns = ['customer_id', 'age', 'income', 'purchase_history', 'last_purchase_date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    # Check for null values
    null_counts = df.isnull().sum()
    high_null_columns = null_counts[null_counts > len(df) * 0.1].index.tolist()
    if high_null_columns:
        warnings.append(f"High null values in columns: {high_null_columns}")

    # Check data types
    if not pd.api.types.is_numeric_dtype(df['age']):
        errors.append("Age column should be numeric")

    if not pd.api.types.is_numeric_dtype(df['income']):
        errors.append("Income column should be numeric")

    # Check for duplicates
    duplicate_count = df.duplicated(subset=['customer_id']).sum()
    if duplicate_count > 0:
        warnings.append(f"Found {duplicate_count} duplicate customer IDs")

    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'statistics': {
            'total_records': len(df),
            'null_counts': null_counts.to_dict(),
            'duplicate_count': duplicate_count
        }
    }

def run_batch_predictions(**context):
    """Run batch predictions with monitoring"""
    # Pull data from XCom
    input_file = context['ti'].xcom_pull(key='input_file')
    record_count = context['ti'].xcom_pull(key='record_count')

    execution_date = context['execution_date']
    date_str = execution_date.strftime('%Y-%m-%d')

    try:
        # Load data
        df = pd.read_csv(input_file)

        # Load model
        model = joblib.load(os.path.join(MODEL_PATH, 'churn_model.pkl'))

        # Preprocess features
        features = preprocess_features(df)

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)

        # Create results DataFrame
        results = pd.DataFrame({
            'customer_id': df['customer_id'],
            'prediction': predictions,
            'probability': probabilities.max(axis=1),
            'timestamp': datetime.now()
        })

        # Save results
        output_file = os.path.join(OUTPUT_PATH, f'predictions_{date_str}.csv')
        results.to_csv(output_file, index=False)

        # Calculate metrics
        metrics = calculate_prediction_metrics(results)

        # Save metrics
        metrics_file = os.path.join(LOG_PATH, f'metrics_{date_str}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Push metrics to XCom
        context['ti'].xcom_push(key='prediction_metrics', value=metrics)
        context['ti'].xcom_push(key='output_file', value=output_file)

        logging.info(f"Completed predictions for {len(results)} customers")
        return {'status': 'success', 'predictions': len(results)}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise

def preprocess_features(df):
    """Preprocess features for model prediction"""
    # Feature engineering
    features = df.copy()

    # Age groups
    features['age_group'] = pd.cut(features['age'], bins=[0, 25, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])

    # Income categories
    features['income_category'] = pd.qcut(features['income'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

    # Time since last purchase
    features['days_since_purchase'] = (pd.to_datetime('today') - pd.to_datetime(features['last_purchase_date'])).dt.days

    # Purchase frequency
    features['purchase_frequency'] = features['purchase_history'] / (features['days_since_purchase'] + 1)

    # Select features for model
    feature_columns = ['age', 'income', 'days_since_purchase', 'purchase_frequency']
    return features[feature_columns]

def calculate_prediction_metrics(results):
    """Calculate prediction metrics"""
    metrics = {
        'total_predictions': len(results),
        'churn_predictions': (results['prediction'] == 1).sum(),
        'non_churn_predictions': (results['prediction'] == 0).sum(),
        'churn_rate': (results['prediction'] == 1).mean(),
        'avg_confidence': results['probability'].mean(),
        'high_confidence_predictions': (results['probability'] > 0.8).sum(),
        'low_confidence_predictions': (results['probability'] < 0.5).sum()
    }
    return metrics

def validate_results(**context):
    """Validate prediction results and generate reports"""
    # Pull metrics from XCom
    metrics = context['ti'].xcom_pull(key='prediction_metrics')
    output_file = context['ti'].xcom_pull(key='output_file')

    try:
        # Load results
        results = pd.read_csv(output_file)

        # Additional validation
        validation_results = {
            'file_exists': os.path.exists(output_file),
            'records_match': len(results) == metrics['total_predictions'],
            'prediction_values_valid': results['prediction'].isin([0, 1]).all(),
            'probability_range_valid': results['probability'].between(0, 1).all(),
            'no_null_predictions': results['prediction'].notna().all(),
            'no_null_probabilities': results['probability'].notna().all()
        }

        # Generate summary report
        report = generate_summary_report(metrics, validation_results)

        # Save report
        execution_date = context['execution_date']
        date_str = execution_date.strftime('%Y-%m-%d')
        report_file = os.path.join(LOG_PATH, f'report_{date_str}.html')

        with open(report_file, 'w') as f:
            f.write(report)

        # Check for anomalies
        anomalies = detect_anomalies(metrics)
        if anomalies:
            context['ti'].xcom_push(key='anomalies', value=anomalies)

        return {'status': 'success', 'validation_passed': all(validation_results.values())}

    except Exception as e:
        logging.error(f"Result validation failed: {e}")
        raise

def generate_summary_report(metrics, validation_results):
    """Generate HTML summary report"""
    html_template = f"""
    <html>
    <head><title>Batch Prediction Report</title></head>
    <body>
        <h1>Batch Prediction Report</h1>
        <h2>Summary Metrics</h2>
        <ul>
            <li>Total Predictions: {metrics['total_predictions']}</li>
            <li>Churn Predictions: {metrics['churn_predictions']}</li>
            <li>Churn Rate: {metrics['churn_rate']:.2%}</li>
            <li>Average Confidence: {metrics['avg_confidence']:.2f}</li>
        </ul>

        <h2>Validation Results</h2>
        <ul>
            {''.join([f'<li>{k}: {"✓" if v else "✗"}</li>' for k, v in validation_results.items()])}
        </ul>
    </body>
    </html>
    """
    return html_template

def detect_anomalies(metrics):
    """Detect anomalies in prediction metrics"""
    anomalies = []

    # Unusual churn rate
    if metrics['churn_rate'] > 0.5 or metrics['churn_rate'] < 0.01:
        anomalies.append(f"Unusual churn rate: {metrics['churn_rate']:.2%}")

    # Low average confidence
    if metrics['avg_confidence'] < 0.6:
        anomalies.append(f"Low average confidence: {metrics['avg_confidence']:.2f}")

    # High percentage of low confidence predictions
    low_conf_ratio = metrics['low_confidence_predictions'] / metrics['total_predictions']
    if low_conf_ratio > 0.3:
        anomalies.append(f"High percentage of low confidence predictions: {low_conf_ratio:.2%}")

    return anomalies

# Task definitions
wait_for_data = FileSensor(
    task_id='wait_for_data',
    filepath='/data/daily/customers_{{{{ ds_nodash }}}}.csv',
    poke_interval=300,  # 5 minutes
    timeout=3600,  # 1 hour
    mode='poke',
    dag=dag,
)

extract_task = PythonOperator(
    task_id='extract_and_validate_data',
    python_callable=extract_and_validate_data,
    provide_context=True,
    dag=dag,
)

predict_task = PythonOperator(
    task_id='run_batch_predictions',
    python_callable=run_batch_predictions,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_results',
    python_callable=validate_results,
    provide_context=True,
    dag=dag,
)

# Success notification
success_notification = SlackWebhookOperator(
    task_id='success_notification',
    http_conn_id='slack_webhook',
    message="✅ Batch prediction pipeline completed successfully",
    channel='#ml-ops',
    dag=dag,
)

# Error notification
error_notification = EmailOperator(
    task_id='error_notification',
    to='ml-team@company.com',
    subject='❌ Batch Prediction Pipeline Failed',
    html_content="Batch prediction pipeline failed. Please check logs.",
    dag=dag,
)

# Task dependencies
wait_for_data >> extract_task >> predict_task >> validate_task

# Success and error paths
validate_task >> success_notification

# Error handling for all tasks
for task in [extract_task, predict_task, validate_task]:
    task.on_failure_callback = error_notification.execute
```

## 2. Real-Time Inference

### Advanced FastAPI Implementation
```python
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import asyncio
import redis
import json
from contextlib import asynccontextmanager
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge
import time
import uvicorn
from concurrent.futures import ThreadPoolExecutor

# Initialize Prometheus metrics
REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total number of prediction requests',
    ['endpoint', 'model_version', 'status']
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Request latency in seconds',
    ['endpoint', 'model_version']
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total number of predictions',
    ['model_version', 'prediction_class']
)

ACTIVE_CONNECTIONS = Gauge(
    'model_active_connections',
    'Number of active connections',
    ['model_version']
)

# Request/Response models
class PredictionRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=100)
    features: Dict[str, Any]
    request_id: Optional[str] = None
    priority: Optional[str] = Field("normal", regex="^(low|normal|high)$")

    @validator('features')
    def validate_features(cls, v):
        required_features = {'age', 'income', 'purchase_history', 'last_purchase_date'}
        missing_features = required_features - set(v.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        return v

class PredictionResponse(BaseModel):
    customer_id: str
    prediction: int
    probability: float
    confidence_level: str
    model_version: str
    request_id: str
    processing_time: float
    timestamp: str
    features_used: List[str]

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]
    batch_id: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    batch_id: str
    predictions: List[PredictionResponse]
    total_processed: int
    success_count: int
    error_count: int
    processing_time: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    last_updated: str
    uptime_seconds: float
    active_connections: int

# Model service class
class ModelService:
    def __init__(self, model_path: str, redis_host: str = 'localhost', redis_port: int = 6379):
        self.model_path = model_path
        self.model = None
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.start_time = datetime.now()
        self.feature_order = ['age', 'income', 'purchase_history', 'days_since_purchase']
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def load_model(self):
        """Load model with error handling"""
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Make single prediction with caching"""
        start_time = time.time()

        # Check cache first
        cache_key = f"prediction:{request.customer_id}:{hash(str(request.features))}"
        cached_result = self.redis_client.get(cache_key)

        if cached_result:
            cached_data = json.loads(cached_result)
            return PredictionResponse(**cached_data)

        # Make prediction
        prediction_data = await self._make_prediction(request)

        # Cache result (TTL: 1 hour)
        self.redis_client.setex(cache_key, 3600, json.dumps(prediction_data.dict()))

        # Update metrics
        REQUEST_LATENCY.labels(endpoint='predict', model_version=prediction_data.model_version).observe(time.time() - start_time)
        PREDICTION_COUNT.labels(
            model_version=prediction_data.model_version,
            prediction_class=prediction_data.prediction
        ).inc()

        return prediction_data

    async def _make_prediction(self, request: PredictionRequest) -> PredictionResponse:
        """Make actual prediction"""
        # Extract and validate features
        features = self.extract_features(request.features)

        # Make prediction
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0].max()

        # Determine confidence level
        confidence_level = self.get_confidence_level(probability)

        return PredictionResponse(
            customer_id=request.customer_id,
            prediction=int(prediction),
            probability=float(probability),
            confidence_level=confidence_level,
            model_version="1.0.0",
            request_id=request.request_id or str(uuid.uuid4()),
            processing_time=time.time() - start_time,
            timestamp=datetime.now().isoformat(),
            features_used=self.feature_order
        )

    def extract_features(self, raw_features: Dict[str, Any]) -> np.ndarray:
        """Extract and preprocess features"""
        features = []

        # Age
        features.append(float(raw_features['age']))

        # Income
        features.append(float(raw_features['income']))

        # Purchase history
        features.append(float(raw_features['purchase_history']))

        # Days since last purchase
        last_purchase = pd.to_datetime(raw_features['last_purchase_date'])
        days_since = (pd.to_datetime('today') - last_purchase).days
        features.append(float(days_since))

        return np.array(features)

    def get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.9:
            return "high"
        elif probability >= 0.7:
            return "medium"
        else:
            return "low"

    async def batch_predict(self, batch_request: BatchPredictionRequest) -> BatchPredictionResponse:
        """Handle batch predictions efficiently"""
        start_time = time.time()
        batch_id = batch_request.batch_id or str(uuid.uuid4())

        # Process predictions in parallel
        tasks = []
        for request in batch_request.requests:
            task = asyncio.create_task(self.predict(request))
            tasks.append(task)

        # Wait for all predictions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        predictions = []
        success_count = 0
        error_count = 0

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_count += 1
                logging.error(f"Error processing request {i}: {result}")
            else:
                predictions.append(result)
                success_count += 1

        return BatchPredictionResponse(
            batch_id=batch_id,
            predictions=predictions,
            total_processed=len(batch_request.requests),
            success_count=success_count,
            error_count=error_count,
            processing_time=time.time() - start_time
        )

    def get_health_status(self) -> dict:
        """Get service health status"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'model_version': '1.0.0',
            'last_updated': self.start_time.isoformat(),
            'uptime_seconds': uptime,
            'active_connections': ACTIVE_CONNECTIONS.labels(model_version='1.0.0')._value.get()
        }

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    model_service = ModelService('models/churn_model.pkl')
    await model_service.load_model()
    app.state.model_service = model_service

    # Increment active connections
    ACTIVE_CONNECTIONS.labels(model_version='1.0.0').inc()

    yield

    # Shutdown
    ACTIVE_CONNECTIONS.labels(model_version='1.0.0').dec()

app = FastAPI(
    title="ML Model API",
    description="Advanced ML model serving API with monitoring and caching",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication for demonstration"""
    # In production, implement proper JWT validation
    return credentials.credentials

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(lambda: app.state.model_service)
):
    """Single prediction endpoint with caching"""
    try:
        # Track request
        REQUEST_COUNT.labels(
            endpoint='predict',
            model_version='1.0.0',
            status='success'
        ).inc()

        # Make prediction
        response = await model_service.predict(request)

        # Log prediction in background
        background_tasks.add_task(log_prediction, request, response)

        return response

    except Exception as e:
        REQUEST_COUNT.labels(
            endpoint='predict',
            model_version='1.0.0',
            status='error'
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    batch_request: BatchPredictionRequest,
    background_tasks: BackgroundTasks,
    model_service: ModelService = Depends(lambda: app.state.model_service)
):
    """Batch prediction endpoint"""
    try:
        # Validate batch size
        if len(batch_request.requests) > 1000:
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 1000 requests")

        # Process batch
        response = await model_service.batch_predict(batch_request)

        # Log batch processing
        background_tasks.add_task(log_batch_prediction, batch_request, response)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check(model_service: ModelService = Depends(lambda: app.state.model_service)):
    """Health check endpoint"""
    return HealthResponse(**model_service.get_health_status())

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return prom.generate_latest()

@app.get("/models/versions")
async def list_model_versions():
    """List available model versions"""
    return {
        "versions": ["1.0.0"],
        "current": "1.0.0",
        "last_updated": "2024-01-01T00:00:00Z"
    }

# Background tasks
async def log_prediction(request: PredictionRequest, response: PredictionResponse):
    """Log prediction to database"""
    # In production, log to database or data warehouse
    log_data = {
        'customer_id': request.customer_id,
        'prediction': response.prediction,
        'probability': response.probability,
        'timestamp': response.timestamp,
        'processing_time': response.processing_time
    }
    logging.info(f"Prediction logged: {log_data}")

async def log_batch_prediction(batch_request: BatchPredictionRequest, response: BatchPredictionResponse):
    """Log batch prediction summary"""
    log_data = {
        'batch_id': response.batch_id,
        'total_processed': response.total_processed,
        'success_count': response.success_count,
        'error_count': response.error_count,
        'processing_time': response.processing_time,
        'timestamp': datetime.now().isoformat()
    }
    logging.info(f"Batch prediction logged: {log_data}")

# Run the app
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        workers=4,
        log_level="info"
    )
```

### Model Serving with BentoML
```python
import bentoml
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput, DataframeInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.service.artifacts import PickleArtifact
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Any
import json
import redis
from prometheus_client import Counter, Histogram, start_http_server
import time

# Initialize metrics
PREDICTION_COUNTER = Counter('bento_predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('bento_prediction_latency_seconds', 'Prediction latency')

@env(infer_pip_packages=True)
@artifacts([
    SklearnModelArtifact('model'),
    PickleArtifact('feature_preprocessor'),
    PickleArtifact('scaler')
])
class AdvancedChurnPredictionService(BentoService):
    """
    Advanced churn prediction service with monitoring, caching, and batch processing
    """

    def __init__(self):
        super().__init__()
        self.redis_client = None
        self.feature_columns = ['age', 'income', 'purchase_history', 'days_since_purchase']

    def setup_redis(self, host='localhost', port=6379):
        """Setup Redis connection for caching"""
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            logging.info("Redis connection established")
        except Exception as e:
            logging.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None

    def _get_cache_key(self, features: Dict[str, Any]) -> str:
        """Generate cache key for features"""
        feature_str = json.dumps(features, sort_keys=True)
        return f"churn_prediction:{hash(feature_str)}"

    def _get_cached_prediction(self, cache_key: str) -> Dict[str, Any]:
        """Get cached prediction if available"""
        if not self.redis_client:
            return None

        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logging.error(f"Cache retrieval error: {e}")

        return None

    def _cache_prediction(self, cache_key: str, prediction: Dict[str, Any], ttl: int = 3600):
        """Cache prediction result"""
        if not self.redis_client:
            return

        try:
            self.redis_client.setex(cache_key, ttl, json.dumps(prediction))
        except Exception as e:
            logging.error(f"Cache storage error: {e}")

    def preprocess_features(self, raw_data: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw features"""
        features = []

        # Basic features
        features.append(float(raw_data['age']))
        features.append(float(raw_data['income']))
        features.append(float(raw_data['purchase_history']))

        # Derived features
        last_purchase = pd.to_datetime(raw_data['last_purchase_date'])
        days_since = (pd.to_datetime('today') - last_purchase).days
        features.append(float(days_since))

        # Convert to numpy array
        features_array = np.array(features).reshape(1, -1)

        # Apply preprocessing
        if hasattr(self.artifacts, 'scaler'):
            features_array = self.artifacts.scaler.transform(features_array)

        return features_array

    @api(input=JsonInput(), batch=False)
    def predict(self, parsed_json: Dict[str, Any]) -> Dict[str, Any]:
        """Single prediction with caching"""
        start_time = time.time()

        try:
            # Setup Redis if not done
            if not self.redis_client:
                self.setup_redis()

            # Generate cache key
            cache_key = self._get_cache_key(parsed_json['features'])

            # Check cache
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                return cached_result

            # Preprocess features
            features = self.preprocess_features(parsed_json['features'])

            # Make prediction
            prediction = int(self.artifacts.model.predict(features)[0])
            probabilities = self.artifacts.model.predict_proba(features)[0]
            probability = float(probabilities.max())
            confidence_level = self._get_confidence_level(probability)

            # Prepare response
            result = {
                'customer_id': parsed_json['customer_id'],
                'prediction': prediction,
                'probability': probability,
                'confidence_level': confidence_level,
                'model_version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time
            }

            # Cache result
            self._cache_prediction(cache_key, result)

            # Update metrics
            PREDICTION_COUNTER.inc()
            PREDICTION_LATENCY.observe(time.time() - start_time)

            return result

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    @api(input=JsonInput(), batch=True)
    def batch_predict(self, parsed_json_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch prediction processing"""
        start_time = time.time()
        results = []

        try:
            # Setup Redis if not done
            if not self.redis_client:
                self.setup_redis()

            # Process each request
            for json_data in parsed_json_list:
                result = self.predict(json_data)
                results.append(result)

            # Update metrics
            PREDICTION_COUNTER.inc(len(parsed_json_list))
            PREDICTION_LATENCY.observe(time.time() - start_time)

            return results

        except Exception as e:
            logging.error(f"Batch prediction error: {e}")
            return [{'error': str(e), 'timestamp': datetime.now().isoformat()}]

    @api(input=DataframeInput(), batch=True)
    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict from pandas DataFrame"""
        try:
            # Preprocess features
            features = df[self.feature_columns].values

            # Apply scaling if available
            if hasattr(self.artifacts, 'scaler'):
                features = self.artifacts.scaler.transform(features)

            # Make predictions
            predictions = self.artifacts.model.predict(features)
            probabilities = self.artifacts.model.predict_proba(features)

            # Create results DataFrame
            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities.max(axis=1),
                'confidence_level': [self._get_confidence_level(p) for p in probabilities.max(axis=1)],
                'timestamp': datetime.now()
            })

            return results

        except Exception as e:
            logging.error(f"DataFrame prediction error: {e}")
            return pd.DataFrame({'error': [str(e)]})

    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.9:
            return "high"
        elif probability >= 0.7:
            return "medium"
        else:
            return "low"

    @api(input=JsonInput(), batch=False)
    def health_check(self, parsed_json: Dict[str, Any]) -> Dict[str, Any]:
        """Health check endpoint"""
        return {
            'status': 'healthy',
            'model_loaded': hasattr(self.artifacts, 'model'),
            'model_version': '1.0.0',
            'redis_connected': self.redis_client is not None,
            'timestamp': datetime.now().isoformat()
        }

# Create and save service
if __name__ == "__main__":
    # Load your trained model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib

    # Example model loading (replace with your actual model)
    trained_model = joblib.load('models/churn_model.pkl')
    scaler = joblib.load('models/scaler.pkl')

    # Create service
    service = AdvancedChurnPredictionService()
    service.pack('model', trained_model)
    service.pack('scaler', scaler)

    # Save service
    saved_path = service.save()

    print(f"Service saved to: {saved_path}")

    # Start metrics server
    start_http_server(8000)
```

## 3. Stream Processing

### Advanced Kafka Streaming Implementation
```python
from kafka import KafkaConsumer, KafkaProducer, KafkaAdminClient
from kafka.admin import NewTopic
from kafka.errors import KafkaError
import json
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
import threading
import queue
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import signal
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Initialize metrics
STREAM_PREDICTIONS = Counter('stream_predictions_total', 'Total stream predictions')
STREAM_LATENCY = Histogram('stream_prediction_latency_seconds', 'Stream prediction latency')
STREAM_ERRORS = Counter('stream_errors_total', 'Total stream errors', ['error_type'])
ACTIVE_STREAMS = Gauge('active_streams', 'Number of active streams')
CONSUMER_LAG = Gauge('consumer_lag', 'Consumer lag for each partition')

@dataclass
class StreamConfig:
    bootstrap_servers: List[str]
    input_topic: str
    output_topic: str
    error_topic: str
    model_path: str
    redis_host: str = 'localhost'
    redis_port: int = 6379
    consumer_group: str = 'ml-stream-consumer'
    max_workers: int = 4
    batch_size: int = 100
    flush_interval: float = 1.0
    metrics_port: int = 8000

class AdvancedStreamingPredictor:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.model = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=True
        )
        self.producer = None
        self.consumer = None
        self.message_queue = queue.Queue(maxsize=10000)
        self.stats = {
            'messages_processed': 0,
            'predictions_made': 0,
            'errors': 0,
            'start_time': None
        }

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"Received signal {signum}, shutting down...")
        self.running = False

    def initialize(self):
        """Initialize Kafka connections and load model"""
        try:
            # Load model
            self.model = joblib.load(self.config.model_path)
            logging.info(f"Model loaded from {self.config.model_path}")

            # Create topics if they don't exist
            self._ensure_topics_exist()

            # Initialize producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda x: json.dumps(x).encode('utf-8'),
                key_serializer=lambda x: str(x).encode('utf-8') if x else None,
                acks='all',
                retries=3,
                max_in_flight_requests_per_connection=1,
                enable_idempotence=True
            )

            # Initialize consumer
            self.consumer = KafkaConsumer(
                self.config.input_topic,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.consumer_group,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                key_deserializer=lambda x: x.decode('utf-8') if x else None,
                auto_offset_reset='latest',
                enable_auto_commit=False,
                max_poll_records=self.config.batch_size,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000
            )

            # Start metrics server
            start_http_server(self.config.metrics_port)

            logging.info("Streaming predictor initialized successfully")
            return True

        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            return False

    def _ensure_topics_exist(self):
        """Ensure Kafka topics exist"""
        try:
            admin_client = KafkaAdminClient(bootstrap_servers=self.config.bootstrap_servers)

            topics = [
                NewTopic(self.config.input_topic, num_partitions=3, replication_factor=2),
                NewTopic(self.config.output_topic, num_partitions=3, replication_factor=2),
                NewTopic(self.config.error_topic, num_partitions=1, replication_factor=2)
            ]

            # Create topics
            admin_client.create_topics(topics, validate_only=False)
            logging.info("Topics created/verified successfully")

        except Exception as e:
            logging.warning(f"Topic creation warning: {e}")

    def start(self):
        """Start the streaming predictor"""
        if not self.initialize():
            return False

        self.running = True
        self.stats['start_time'] = datetime.now()

        # Start worker threads
        threads = [
            threading.Thread(target=self._consume_messages, daemon=True),
            threading.Thread(target=self._process_messages, daemon=True),
            threading.Thread(target=self._monitor_performance, daemon=True),
            threading.Thread(target=self._periodic_flush, daemon=True)
        ]

        for thread in threads:
            thread.start()

        # Update metrics
        ACTIVE_STREAMS.inc()

        logging.info("Streaming predictor started")
        return True

    def _consume_messages(self):
        """Consume messages from Kafka"""
        logging.info("Starting message consumer")

        while self.running:
            try:
                # Poll for messages
                batch = self.consumer.poll(timeout_ms=1000)

                if batch:
                    for topic_partition, messages in batch.items():
                        for message in messages:
                            try:
                                self.message_queue.put(message, timeout=1)
                                self.stats['messages_processed'] += 1
                            except queue.Full:
                                logging.error("Message queue full, dropping message")
                                STREAM_ERRORS.labels(error_type='queue_full').inc()

                    # Commit offsets
                    self.consumer.commit_async()

                # Update consumer lag
                self._update_consumer_lag()

            except Exception as e:
                logging.error(f"Consumer error: {e}")
                STREAM_ERRORS.labels(error_type='consumer_error').inc()
                time.sleep(1)

    def _process_messages(self):
        """Process messages from the queue"""
        logging.info("Starting message processor")

        while self.running:
            try:
                message = self.message_queue.get(timeout=1)

                # Process message in executor
                future = self.executor.submit(self._process_single_message, message)

                # Add callback for result handling
                future.add_done_callback(self._handle_processing_result)

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processor error: {e}")
                STREAM_ERRORS.labels(error_type='processor_error').inc()

    def _process_single_message(self, message):
        """Process a single message"""
        start_time = time.time()

        try:
            # Extract data
            data = message.value
            customer_id = data.get('customer_id')

            if not customer_id:
                raise ValueError("Missing customer_id")

            # Check cache
            cache_key = f"stream_prediction:{customer_id}"
            cached_result = self.redis_client.get(cache_key)

            if cached_result:
                return json.loads(cached_result)

            # Extract features
            features = self.extract_features(data)

            # Make prediction
            prediction = self.model.predict([features])[0]
            probability = self.model.predict_proba([features])[0].max()

            # Create result
            result = {
                'customer_id': customer_id,
                'prediction': int(prediction),
                'probability': float(probability),
                'confidence_level': self.get_confidence_level(probability),
                'model_version': '1.0.0',
                'timestamp': datetime.now().isoformat(),
                'processing_time': time.time() - start_time,
                'original_timestamp': data.get('timestamp')
            }

            # Cache result
            self.redis_client.setex(cache_key, 3600, json.dumps(result))

            # Update metrics
            STREAM_PREDICTIONS.inc()
            STREAM_LATENCY.observe(time.time() - start_time)
            self.stats['predictions_made'] += 1

            return result

        except Exception as e:
            logging.error(f"Message processing error: {e}")
            STREAM_ERRORS.labels(error_type='processing_error').inc()

            # Create error result
            return {
                'customer_id': data.get('customer_id', 'unknown'),
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'original_data': data
            }

    def _handle_processing_result(self, future):
        """Handle processing result from executor"""
        try:
            result = future.result()

            # Send to appropriate topic
            if 'error' in result:
                self.producer.send(self.config.error_topic, value=result)
            else:
                self.producer.send(self.config.output_topic, value=result)

        except Exception as e:
            logging.error(f"Result handling error: {e}")
            STREAM_ERRORS.labels(error_type='result_handler_error').inc()

    def _monitor_performance(self):
        """Monitor system performance"""
        while self.running:
            try:
                # Calculate statistics
                uptime = (datetime.now() - self.stats['start_time']).total_seconds()
                messages_per_second = self.stats['messages_processed'] / uptime if uptime > 0 else 0
                predictions_per_second = self.stats['predictions_made'] / uptime if uptime > 0 else 0

                logging.info(f"""
                Performance Stats:
                - Uptime: {uptime:.2f}s
                - Messages processed: {self.stats['messages_processed']}
                - Predictions made: {self.stats['predictions_made']}
                - Errors: {self.stats['errors']}
                - Messages/sec: {messages_per_second:.2f}
                - Predictions/sec: {predictions_per_second:.2f}
                - Queue size: {self.message_queue.qsize()}
                """)

                # Reset some counters
                self.stats['errors'] = 0

                time.sleep(60)  # Report every minute

            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                time.sleep(60)

    def _periodic_flush(self):
        """Periodic flush of producer"""
        while self.running:
            try:
                if self.producer:
                    self.producer.flush()
                time.sleep(self.config.flush_interval)
            except Exception as e:
                logging.error(f"Flush error: {e}")

    def _update_consumer_lag(self):
        """Update consumer lag metrics"""
        try:
            if self.consumer:
                offsets = self.consumer.committed(*self.consumer.assignment())
                for topic_partition, committed_offset in offsets.items():
                    lag = self.consumer.highwater(topic_partition) - committed_offset
                    CONSUMER_LAG.labels(
                        topic=topic_partition.topic,
                        partition=topic_partition.partition
                    ).set(lag)
        except Exception as e:
            logging.error(f"Lag monitoring error: {e}")

    def extract_features(self, data: Dict[str, Any]) -> np.ndarray:
        """Extract features from raw data"""
        features = []

        # Basic features
        features.append(float(data.get('age', 0)))
        features.append(float(data.get('income', 0)))
        features.append(float(data.get('purchase_history', 0)))

        # Derived features
        last_purchase = pd.to_datetime(data.get('last_purchase_date', datetime.now()))
        days_since = (pd.to_datetime('today') - last_purchase).days
        features.append(float(days_since))

        return np.array(features)

    def get_confidence_level(self, probability: float) -> str:
        """Determine confidence level"""
        if probability >= 0.9:
            return "high"
        elif probability >= 0.7:
            return "medium"
        else:
            return "low"

    def stop(self):
        """Stop the streaming predictor"""
        logging.info("Stopping streaming predictor...")
        self.running = False

        # Stop consumer
        if self.consumer:
            self.consumer.close()

        # Stop producer
        if self.producer:
            self.producer.flush()
            self.producer.close()

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)

        # Update metrics
        ACTIVE_STREAMS.dec()

        logging.info("Streaming predictor stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        if self.stats['start_time']:
            uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        else:
            uptime = 0

        return {
            'uptime': uptime,
            'messages_processed': self.stats['messages_processed'],
            'predictions_made': self.stats['predictions_made'],
            'errors': self.stats['errors'],
            'queue_size': self.message_queue.qsize(),
            'running': self.running
        }

# Main execution
if __name__ == "__main__":
    # Configuration
    config = StreamConfig(
        bootstrap_servers=['localhost:9092'],
        input_topic='customer-events',
        output_topic='predictions',
        error_topic='prediction-errors',
        model_path='models/churn_model.pkl',
        consumer_group='ml-stream-predictor',
        max_workers=4,
        batch_size=100,
        flush_interval=1.0,
        metrics_port=8000
    )

    # Create and start predictor
    predictor = AdvancedStreamingPredictor(config)

    if predictor.start():
        try:
            # Keep running until interrupted
            while predictor.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Received interrupt signal")
        finally:
            predictor.stop()
    else:
        logging.error("Failed to start streaming predictor")
        sys.exit(1)
```

## Key Takeaways

### Deployment Strategies
1. **Batch Processing**: Scheduled predictions for large datasets
2. **Real-Time Inference**: Low-latency predictions for individual requests
3. **Stream Processing**: Continuous predictions for streaming data
4. **Hybrid Approaches**: Combining multiple strategies as needed

### Best Practices
- **Monitoring**: Comprehensive metrics and logging
- **Error Handling**: Robust error handling and recovery
- **Caching**: Intelligent caching for performance optimization
- **Scalability**: Horizontal scaling and load balancing
- **Reliability**: High availability and fault tolerance

### Common Challenges
- **Performance**: Maintaining low latency and high throughput
- **Resource Management**: Efficient resource utilization
- **Data Consistency**: Ensuring data consistency across systems
- **Monitoring**: Comprehensive monitoring and alerting
- **Deployment**: Smooth deployment and rollback processes

---

## Next Steps

Continue to [Module 4: Infrastructure and Orchestration](04_Infrastructure_and_Orchestration.md) to learn about containerization, Kubernetes, and orchestration strategies for ML systems.

## Quick Reference

### Key Concepts
- **Batch Processing**: Scheduled predictions for large datasets
- **Real-Time Inference**: Low-latency predictions for individual requests
- **Stream Processing**: Continuous predictions for streaming data
- **Caching**: Intelligent caching for performance optimization
- **Monitoring**: Comprehensive metrics and logging

### Essential Tools
- **Apache Beam**: Unified batch and stream processing
- **Airflow**: Workflow orchestration
- **FastAPI**: High-performance API framework
- **BentoML**: Model serving framework
- **Kafka**: Distributed streaming platform

### Common Patterns
- **Batch Pipeline**: Scheduled data processing and prediction
- **Real-Time API**: Low-latency prediction service
- **Stream Processor**: Continuous prediction on streaming data
- **Hybrid Architecture**: Combining multiple deployment strategies
- **Monitoring Stack**: Comprehensive monitoring and alerting