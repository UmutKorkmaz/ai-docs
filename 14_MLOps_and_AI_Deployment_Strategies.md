# MLOps and AI Deployment Strategies (2024-2025 Edition)

## Introduction

Machine Learning Operations (MLOps) has evolved dramatically in 2024-2025, now encompassing Large Language Model Operations (LLMOps), AI Operations (AIOps), and advanced deployment strategies for next-generation AI systems. This comprehensive guide covers the cutting-edge practices, tools, and architectures that define modern AI operations.

### Major Advances in 2024-2025

- **LLMOps**: Specialized operations for large language models and generative AI
- **AIOps Integration**: AI-powered operations and automation
- **Edge AI Deployment**: Optimized deployment for edge devices and IoT
- **Federated Learning MLOps**: Privacy-preserving distributed training
- **Mixture of Experts (MoE) Operations**: Managing complex multi-model systems
- **Multi-modal AI Operations**: Coordinating diverse AI model types
- **AI Agent Orchestration**: Managing complex agentic workflows
- **Sustainable AI Operations**: Green computing and resource optimization

## Table of Contents

1. [MLOps Fundamentals](#mlops-fundamentals)
2. [LLMOps and Generative AI Operations](#llmops-and-generative-ai-operations)
3. [Advanced Deployment Strategies](#advanced-deployment-strategies)
4. [Infrastructure and Orchestration](#infrastructure-and-orchestration)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Model Management and Versioning](#model-management-and-versioning)
7. [CI/CD for Machine Learning](#cicd-for-machine-learning)
8. [Edge AI and Federated Learning](#edge-ai-and-federated-learning)
9. [AIOps and Automation](#aiops-and-automation)
10. [Production Best Practices](#production-best-practices)
11. [Security and Compliance](#security-and-compliance)
12. [Future Trends](#future-trends)

---

## MLOps Fundamentals

### What is MLOps?

MLOps is a set of practices that aims to deploy and maintain machine learning models in production reliably and efficiently. It combines Machine Learning, DevOps, and Data Engineering principles to create a unified workflow for ML systems.

### Core Principles

#### 1. Automation
- **Automated Training**: Scheduled model retraining
- **Automated Testing**: Data validation, model validation, integration testing
- **Automated Deployment**: Continuous deployment pipelines
- **Automated Monitoring**: Performance tracking and alerting

#### 2. Reproducibility
- **Environment Management**: Consistent development and production environments
- **Data Versioning**: Track data changes and lineage
- **Model Versioning**: Version control for models and code
- **Experiment Tracking**: Record and compare experiments

#### 3. Collaboration
- **Cross-functional Teams**: Data scientists, ML engineers, DevOps engineers
- **Shared Tools**: Common platforms and workflows
- **Documentation**: Comprehensive model and pipeline documentation
- **Knowledge Sharing**: Best practices and lessons learned

### MLOps vs DevOps

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| **Code** | Application code | ML code + models |
| **Data** | Configuration files | Training/validation datasets |
| **Testing** | Unit/integration tests | Data validation + model validation |
| **Deployment** | Code deployment | Model + code deployment |
| **Monitoring** | Application metrics | Model performance + drift |
| **Rollback** | Previous code version | Previous model version |

---

## LLMOps and Generative AI Operations

### Introduction to LLMOps

Large Language Model Operations (LLMOps) has emerged as a specialized discipline within MLOps, focusing on the unique challenges of deploying, managing, and optimizing large language models and generative AI systems.

### Key Challenges in LLMOps

#### 1. Model Scale and Complexity
- **Parameter Scale**: Models with 100B+ parameters require specialized infrastructure
- **Computational Requirements**: Massive GPU/TPU resources for training and inference
- **Memory Management**: Efficient handling of large model weights and contexts
- **Latency Requirements**: Real-time inference despite model complexity

#### 2. Prompt Engineering and Management
- **Prompt Templates**: Managing and versioning prompt libraries
- **Prompt Optimization**: Automated prompt improvement and testing
- **Context Management**: Efficient handling of long contexts and memory
- **Output Quality Control**: Ensuring consistent, high-quality outputs

#### 3. Safety and Alignment
- **Content Filtering**: Implementing robust safety mechanisms
- **Bias Detection**: Identifying and mitigating harmful biases
- **Alignment Monitoring**: Ensuring models stay aligned with human values
- **Red Teaming**: Continuous adversarial testing

### LLMOps Architecture

```python

class LLMOperations:
    def __init__(self, config):
        self.config = config
        self.model_registry = ModelRegistry()
        self.prompt_manager = PromptManager()
        self.safety_system = SafetySystem()
        self.monitoring_system = LLMMonitoringSystem()
        self.deployment_engine = LLMDeploymentEngine()

    def deploy_llm_application(self, app_config: dict) -> dict:
        """Deploy complete LLM application with LLMOps"""

        # Step 1: Model selection and preparation
        model = self._select_and_prepare_model(app_config)

        # Step 2: Prompt template setup
        prompt_templates = self._setup_prompt_templates(app_config)

        # Step 3: Safety system configuration
        safety_config = self._configure_safety_system(app_config)

        # Step 4: Deployment infrastructure setup
        deployment = self._setup_deployment_infrastructure(app_config)

        # Step 5: Monitoring and observability
        monitoring = self._setup_monitoring(deployment)

        return {
            'model': model,
            'prompts': prompt_templates,
            'safety': safety_config,
            'deployment': deployment,
            'monitoring': monitoring
        }
```

### Prompt Management System

```python

class PromptManager:
    def __init__(self):
        self.template_registry = TemplateRegistry()
        self.version_control = PromptVersionControl()
        self.optimization_engine = PromptOptimizationEngine()
        self.testing_framework = PromptTestingFramework()

    def register_prompt_template(self, template: PromptTemplate) -> str:
        """Register new prompt template with versioning"""

        # Validate template
        validation_result = self._validate_template(template)
        if not validation_result['valid']:
            raise ValueError(f"Invalid template: {validation_result['errors']}")

        # Assign version
        version = self.version_control.create_version(template)

        # Register in registry
        template_id = self.template_registry.register(template, version)

        # Run optimization
        optimization_result = self.optimization_engine.optimize(template)

        # Run tests
        test_results = self.testing_framework.test_template(template)

        return template_id

    def optimize_prompt(self, template_id: str, optimization_goals: list) -> PromptTemplate:
        """Optimize existing prompt template"""

        template = self.template_registry.get(template_id)

        # Generate optimized variants
        variants = self.optimization_engine.generate_variants(
            template,
            optimization_goals
        )

        # Evaluate variants
        evaluation_results = self.testing_framework.evaluate_variants(variants)

        # Select best variant
        best_variant = max(evaluation_results.items(), key=lambda x: x[1]['score'])

        # Create new version
        optimized_template = self.version_control.create_version(best_variant[0])

        return optimized_template
```

### LLM Safety and Alignment System

```python

class SafetySystem:
    def __init__(self):
        self.content_filters = ContentFilters()
        self.bias_detectors = BiasDetectors()
        self.alignment_monitor = AlignmentMonitor()
        self.red_team_engine = RedTeamEngine()
        self.emergency_controls = EmergencyControls()

    def validate_output(self, input_text: str, output_text: str) -> dict:
        """Validate LLM output against safety constraints"""

        validation_results = {
            'content_filter': self.content_filters.check(output_text),
            'bias_detection': self.bias_detectors.analyze(input_text, output_text),
            'alignment_check': self.alignment_monitor.check_alignment(input_text, output_text),
            'quality_assessment': self._assess_output_quality(input_text, output_text)
        }

        # Determine overall safety status
        safety_score = self._calculate_safety_score(validation_results)

        return {
            'safe': safety_score >= self.config.safety_threshold,
            'score': safety_score,
            'details': validation_results,
            'recommendations': self._generate_recommendations(validation_results)
        }

    def red_team_assessment(self, model_interface) -> dict:
        """Conduct red team assessment of LLM system"""

        # Generate adversarial prompts
        adversarial_prompts = self.red_team_engine.generate_prompts()

        # Test system against adversarial prompts
        test_results = []
        for prompt in adversarial_prompts:
            response = model_interface.generate(prompt)
            safety_check = self.validate_output(prompt, response)

            test_results.append({
                'prompt': prompt,
                'response': response,
                'safety_check': safety_check
            })

        # Analyze vulnerabilities
        vulnerability_analysis = self._analyze_vulnerabilities(test_results)

        return {
            'test_results': test_results,
            'vulnerabilities': vulnerability_analysis,
            'remediation_suggestions': self._generate_remediation_suggestions(vulnerability_analysis)
        }
```

### LLM Deployment Strategies

```python

class LLMDeploymentEngine:
    def __init__(self):
        self.quantization_engine = QuantizationEngine()
        self.cache_manager = CacheManager()
        self.load_balancer = LoadBalancer()
        self.scaling_manager = ScalingManager()

    def deploy_model(self, model_config: dict, deployment_config: dict) -> dict:
        """Deploy LLM with optimization strategies"""

        # Model optimization
        optimized_model = self._optimize_model(model_config)

        # Setup serving infrastructure
        serving_config = self._setup_serving_infrastructure(deployment_config)

        # Configure caching
        cache_config = self._configure_cache(deployment_config)

        # Setup load balancing
        load_balancer_config = self._setup_load_balancer(deployment_config)

        # Configure auto-scaling
        scaling_config = self._configure_scaling(deployment_config)

        return {
            'model': optimized_model,
            'serving': serving_config,
            'cache': cache_config,
            'load_balancer': load_balancer_config,
            'scaling': scaling_config
        }

    def _optimize_model(self, model_config: dict):
        """Apply optimization techniques to LLM"""

        optimization_strategies = []

        # Quantization
        if model_config.get('quantization', True):
            quantized_model = self.quantization_engine.quantize(
                model_config['model'],
                precision=model_config.get('precision', 'int8')
            )
            optimization_strategies.append('quantization')

        # Knowledge distillation (for smaller models)
        if model_config.get('distillation', False):
            distilled_model = self._distill_model(model_config['model'])
            optimization_strategies.append('distillation')

        # Pruning
        if model_config.get('pruning', False):
            pruned_model = self._prune_model(model_config['model'])
            optimization_strategies.append('pruning')

        return {
            'model': quantized_model if 'quantization' in optimization_strategies else model_config['model'],
            'optimizations': optimization_strategies,
            'performance_metrics': self._benchmark_optimized_model(model_config['model'])
        }
```

### LLM Monitoring and Observability

```python

class LLMMonitoringSystem:
    def __init__(self):
        self.metrics_collector = LLMetricsCollector()
        self.drift_detector = LLMDriftDetector()
        self.quality_analyzer = OutputQualityAnalyzer()
        self.cost_tracker = CostTracker()
        self.alert_manager = AlertManager()

    def monitor_deployment(self, deployment_id: str) -> dict:
        """Comprehensive monitoring of LLM deployment"""

        monitoring_data = {
            'performance_metrics': self.metrics_collector.collect(deployment_id),
            'drift_analysis': self.drift_detector.analyze(deployment_id),
            'quality_metrics': self.quality_analyzer.analyze(deployment_id),
            'cost_analysis': self.cost_tracker.analyze(deployment_id),
            'user_feedback': self._collect_user_feedback(deployment_id)
        }

        # Generate insights and alerts
        insights = self._generate_insights(monitoring_data)
        alerts = self.alert_manager.generate_alerts(monitoring_data)

        return {
            'monitoring_data': monitoring_data,
            'insights': insights,
            'alerts': alerts,
            'recommendations': self._generate_recommendations(insights)
        }

    def detect_performance_degradation(self, deployment_id: str) -> dict:
        """Detect performance degradation in LLM deployment"""

        # Collect recent performance data
        recent_metrics = self.metrics_collector.get_recent_metrics(deployment_id, hours=24)

        # Compare with baseline
        baseline_metrics = self.metrics_collector.get_baseline_metrics(deployment_id)

        # Analyze degradation patterns
        degradation_analysis = self._analyze_degradation(recent_metrics, baseline_metrics)

        # Root cause analysis
        root_causes = self._identify_root_causes(degradation_analysis)

        # Generate remediation suggestions
        remediation = self._generate_remediation_plan(root_causes)

        return {
            'degradation_detected': degradation_analysis['degradation_detected'],
            'severity': degradation_analysis['severity'],
            'root_causes': root_causes,
            'remediation_plan': remediation,
            'impact_assessment': self._assess_impact(degradation_analysis)
        }
```

---

## Model Development Lifecycle

### 1. Problem Definition and Requirements

#### Business Requirements
```yaml
project:
  name: "Customer Churn Prediction"
  objective: "Predict customer churn with 85% accuracy"
  business_impact: "Reduce churn by 15% through targeted interventions"
  success_metrics:
    - "Model accuracy > 85%"
    - "Prediction latency < 100ms"
    - "Monthly model drift < 5%"

constraints:
  - "GDPR compliance required"
  - "Explainable predictions needed"
  - "Real-time inference required"
```

#### Technical Requirements
```python
class ModelRequirements:
    def __init__(self):
        self.performance_requirements = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.85,
            'f1_score': 0.82
        }

        self.operational_requirements = {
            'inference_latency': '< 100ms',
            'throughput': '> 1000 requests/second',
            'availability': '99.9%',
            'scalability': 'auto-scaling 1-10 instances'
        }

        self.compliance_requirements = [
            'GDPR',
            'SOC2',
            'PCI-DSS'
        ]
```

### 2. Data Management

#### Data Pipeline Architecture
```python
class DataPipeline:
    def __init__(self):
        self.stages = [
            'data_ingestion',
            'data_validation',
            'data_preprocessing',
            'feature_engineering',
            'data_splitting'
        ]

    def data_ingestion(self):
        """Collect data from various sources"""
        sources = {
            'database': self.extract_from_db(),
            'api': self.fetch_from_apis(),
            'files': self.load_files(),
            'streaming': self.consume_streams()
        }
        return self.merge_sources(sources)

    def data_validation(self, data):
        """Validate data quality and schema"""
        validations = [
            self.validate_schema(data),
            self.check_data_quality(data),
            self.detect_anomalies(data),
            self.verify_completeness(data)
        ]
        return all(validations)

    def feature_engineering(self, data):
        """Create and transform features"""
        features = FeatureStore()

        # Apply transformations
        transformed_data = features.apply_transformations(data)

        # Create new features
        engineered_features = features.create_features(transformed_data)

        return engineered_features
```

#### Data Versioning with DVC
```bash
# Initialize DVC
dvc init

# Add data to version control
dvc add data/raw/customers.csv
dvc add data/processed/features.parquet

# Create data pipeline
dvc run -f prepare.dvc \
    -d data/raw/customers.csv \
    -o data/processed/features.parquet \
    python prepare_data.py

# Commit changes
git add prepare.dvc data/raw/customers.csv.dvc
git commit -m "Add data preparation pipeline"

# Push data to remote storage
dvc push
```

### 3. Experiment Tracking

#### MLflow Experiment Management
```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

class ExperimentTracker:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def log_experiment(self, model, X_train, X_test, y_train, y_test, params):
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred)
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log artifacts
            self.log_feature_importance(model)
            self.log_confusion_matrix(y_test, y_pred)

            return mlflow.active_run().info.run_id

    def calculate_metrics(self, y_true, y_pred):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
```

#### Weights & Biases Integration
```python
import wandb
from sklearn.ensemble import RandomForestClassifier

# Initialize wandb
wandb.init(project="customer-churn", entity="your-team")

# Define hyperparameters
config = wandb.config
config.n_estimators = 100
config.max_depth = 10
config.random_state = 42

# Train model
model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    random_state=config.random_state
)

model.fit(X_train, y_train)

# Log metrics
wandb.log({
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average='weighted'),
    "recall": recall_score(y_test, y_pred, average='weighted')
})

# Log model
wandb.save("model.pkl")
```

---

## Deployment Strategies

### 1. Batch Prediction

#### Scheduled Batch Processing
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

class BatchPredictionPipeline:
    def __init__(self, model_path, input_path, output_path):
        self.model_path = model_path
        self.input_path = input_path
        self.output_path = output_path
        self.model = self.load_model()

    def load_model(self):
        import joblib
        return joblib.load(self.model_path)

    def run_pipeline(self):
        pipeline_options = PipelineOptions()

        with beam.Pipeline(options=pipeline_options) as pipeline:
            (pipeline
             | 'Read Data' >> beam.io.ReadFromText(self.input_path)
             | 'Parse JSON' >> beam.Map(json.loads)
             | 'Make Predictions' >> beam.Map(self.predict)
             | 'Format Output' >> beam.Map(self.format_prediction)
             | 'Write Results' >> beam.io.WriteToText(self.output_path))

    def predict(self, data):
        features = self.extract_features(data)
        prediction = self.model.predict([features])[0]
        probability = self.model.predict_proba([features])[0].max()

        return {
            'customer_id': data['customer_id'],
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.now().isoformat()
        }
```

#### Airflow DAG for Batch Processing
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'batch_prediction_pipeline',
    default_args=default_args,
    description='Daily batch prediction pipeline',
    schedule_interval='@daily',
    catchup=False
)

def extract_data(**context):
    """Extract data for prediction"""
    # Implementation here
    pass

def run_predictions(**context):
    """Run batch predictions"""
    # Implementation here
    pass

def validate_results(**context):
    """Validate prediction results"""
    # Implementation here
    pass

# Define tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag
)

predict_task = PythonOperator(
    task_id='run_predictions',
    python_callable=run_predictions,
    dag=dag
)

validate_task = PythonOperator(
    task_id='validate_results',
    python_callable=validate_results,
    dag=dag
)

# Set dependencies
extract_task >> predict_task >> validate_task
```

### 2. Real-Time Inference

#### REST API with FastAPI
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List, Dict, Any
import logging

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
model = None

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load("models/churn_model.pkl")
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

class PredictionRequest(BaseModel):
    features: Dict[str, Any]
    customer_id: str

class PredictionResponse(BaseModel):
    customer_id: str
    prediction: int
    probability: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Extract and validate features
        features = extract_features(request.features)

        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0].max()

        return PredictionResponse(
            customer_id=request.customer_id,
            prediction=int(prediction),
            probability=float(probability),
            model_version="1.0.0"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def extract_features(raw_features: Dict[str, Any]) -> List[float]:
    """Extract and transform features for prediction"""
    # Implementation here
    pass
```

#### Model Serving with BentoML
```python
import bentoml
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import JsonInput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])
class ChurnPredictionService(BentoService):

    @api(input=JsonInput(), batch=True)
    def predict(self, parsed_json_list):
        results = []

        for json_data in parsed_json_list:
            features = self.extract_features(json_data)
            prediction = self.artifacts.model.predict([features])[0]
            probability = self.artifacts.model.predict_proba([features])[0].max()

            results.append({
                'prediction': int(prediction),
                'probability': float(probability)
            })

        return results

    def extract_features(self, data):
        # Feature extraction logic
        pass

# Save the service
service = ChurnPredictionService()
service.pack('model', trained_model)
saved_path = service.save()
```

### 3. Stream Processing

#### Kafka Streaming with Kafka Streams
```python
from kafka import KafkaConsumer, KafkaProducer
import json
import joblib
from datetime import datetime

class StreamingPredictor:
    def __init__(self, model_path, input_topic, output_topic):
        self.model = joblib.load(model_path)
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        self.output_topic = output_topic

    def process_stream(self):
        for message in self.consumer:
            try:
                # Extract features
                data = message.value
                features = self.extract_features(data)

                # Make prediction
                prediction = self.model.predict([features])[0]
                probability = self.model.predict_proba([features])[0].max()
                
                # Create result
                result = {
                    'customer_id': data.get('customer_id'),
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': '1.0.0'
                }
                
                # Send to output topic
                self.producer.send(self.output_topic, result)
                
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                # Send to error topic
                self.producer.send('prediction-errors', {
                    'error': str(e),
                    'message': message.value,
                    'timestamp': datetime.now().isoformat()
                })
    
    def extract_features(self, data):
        """Extract features from raw data"""
        # Implementation here
        pass

# Run streaming predictor
if __name__ == "__main__":
    predictor = StreamingPredictor(
        model_path='models/churn_model.pkl',
        input_topic='customer-events',
        output_topic='predictions'
    )
    predictor.process_stream()
```

---

## Infrastructure and Orchestration

### Container-Based Deployment

#### Docker Configuration
```dockerfile
# Dockerfile for ML Model Service
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/latest_model.pkl
ENV PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### Docker Compose for Development
```yaml
# docker-compose.yml
version: '3.8'

services:
  model-service:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MODEL_PATH=/app/models/latest_model.pkl
      - LOG_LEVEL=INFO
      - REDIS_HOST=redis
      - POSTGRES_HOST=postgres
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - ml-network

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_USER=mlops
      - POSTGRES_PASSWORD=mlops123
      - POSTGRES_DB=mlops_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - ml-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - ml-network

volumes:
  postgres_data:
  grafana_data:

networks:
  ml-network:
    driver: bridge
```

### Kubernetes Deployment

#### Complete Kubernetes Manifests
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-platform
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-model-config
  namespace: ml-platform
data:
  MODEL_VERSION: "1.0.0"
  LOG_LEVEL: "INFO"
  MAX_BATCH_SIZE: "32"
  INFERENCE_TIMEOUT: "100"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-model-secrets
  namespace: ml-platform
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0BkYi9tbG9wcw==
  API_KEY: YWJjZGVmZ2hpams=
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-service
  namespace: ml-platform
  labels:
    app: ml-model
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: model-server
        image: your-registry/ml-model:1.0.0
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        envFrom:
        - configMapRef:
            name: ml-model-config
        - secretRef:
            name: ml-model-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
  namespace: ml-platform
spec:
  selector:
    app: ml-model
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
  namespace: ml-platform
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_latency
      target:
        type: AverageValue
        averageValue: "100"
```

### Helm Chart for ML Platform

```yaml
# Chart.yaml
apiVersion: v2
name: ml-platform
description: A Helm chart for ML model deployment
type: application
version: 1.0.0
appVersion: "1.0.0"

# values.yaml
replicaCount: 3

image:
  repository: your-registry/ml-model
  pullPolicy: IfNotPresent
  tag: "1.0.0"

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

monitoring:
  enabled: true
  prometheus:
    enabled: true
    scrapeInterval: 30s
  grafana:
    enabled: true
    dashboards:
      - model-performance
      - infrastructure-metrics

modelConfig:
  version: "1.0.0"
  batchSize: 32
  inferenceTimeout: 100
  logLevel: INFO

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 10Gi
```

---

## Monitoring and Observability

### Comprehensive Monitoring Stack

```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
import functools
import logging
from typing import Callable, Any
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

class ModelMonitoring:
    """
    Comprehensive monitoring for ML models in production.
    """
    
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        self.registry = CollectorRegistry()
        
        # Initialize metrics
        self.setup_metrics()
        
        # Initialize drift detector
        self.drift_detector = DriftDetector()
        
        # Initialize performance tracker
        self.performance_tracker = PerformanceTracker()
        
    def setup_metrics(self):
        """Setup Prometheus metrics"""
        
        # Request metrics
        self.request_count = Counter(
            'model_request_count',
            'Total number of requests',
            ['model_name', 'model_version', 'endpoint'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'model_request_latency_seconds',
            'Request latency in seconds',
            ['model_name', 'model_version', 'endpoint'],
            registry=self.registry
        )
        
        # Prediction metrics
        self.prediction_count = Counter(
            'model_prediction_count',
            'Total number of predictions',
            ['model_name', 'model_version', 'prediction_class'],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'model_prediction_confidence',
            'Prediction confidence distribution',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'model_error_count',
            'Total number of errors',
            ['model_name', 'model_version', 'error_type'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Current model accuracy',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.model_drift_score = Gauge(
            'model_drift_score',
            'Current drift score',
            ['model_name', 'model_version', 'drift_type'],
            registry=self.registry
        )
        
        # Resource metrics
        self.memory_usage = Gauge(
            'model_memory_usage_bytes',
            'Memory usage in bytes',
            ['model_name', 'model_version'],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'model_cpu_usage_percent',
            'CPU usage percentage',
            ['model_name', 'model_version'],
            registry=self.registry
        )
    
    def track_request(self, endpoint: str):
        """Decorator to track request metrics"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Increment request count
                self.request_count.labels(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    endpoint=endpoint
                ).inc()
                
                # Track latency
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_count.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        error_type=type(e).__name__
                    ).inc()
                    raise
                finally:
                    latency = time.time() - start_time
                    self.request_latency.labels(
                        model_name=self.model_name,
                        model_version=self.model_version,
                        endpoint=endpoint
                    ).observe(latency)
            
            return wrapper
        return decorator
    
    def track_prediction(self, prediction: int, confidence: float):
        """Track prediction metrics"""
        self.prediction_count.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            prediction_class=str(prediction)
        ).inc()
        
        self.prediction_confidence.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(confidence)
    
    def check_drift(self, current_features: np.ndarray, reference_features: np.ndarray):
        """Check for data drift"""
        drift_scores = self.drift_detector.detect_drift(
            current_features,
            reference_features
        )
        
        for drift_type, score in drift_scores.items():
            self.model_drift_score.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                drift_type=drift_type
            ).set(score)
            
            if score > self.drift_detector.threshold:
                self.trigger_drift_alert(drift_type, score)
    
    def trigger_drift_alert(self, drift_type: str, score: float):
        """Trigger alert for drift detection"""
        alert = {
            'alert_type': 'drift_detected',
            'model_name': self.model_name,
            'model_version': self.model_version,
            'drift_type': drift_type,
            'drift_score': score,
            'timestamp': datetime.now().isoformat(),
            'severity': self.get_severity(score)
        }
        
        # Send alert (implement actual alerting mechanism)
        self.send_alert(alert)
    
    def get_severity(self, score: float) -> str:
        """Determine alert severity based on score"""
        if score > 0.8:
            return 'critical'
        elif score > 0.6:
            return 'warning'
        else:
            return 'info'
    
    def send_alert(self, alert: dict):
        """Send alert to monitoring system"""
        # Implement actual alerting (e.g., PagerDuty, Slack, email)
        logging.warning(f"Alert: {alert}")


class DriftDetector:
    """Detect various types of drift in ML models"""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.drift_methods = {
            'kolmogorov_smirnov': self.ks_test,
            'population_stability_index': self.psi,
            'jensen_shannon': self.jensen_shannon_distance,
            'wasserstein': self.wasserstein_distance
        }
    
    def detect_drift(self, current: np.ndarray, reference: np.ndarray) -> dict:
        """Detect drift using multiple methods"""
        drift_scores = {}
        
        for method_name, method_func in self.drift_methods.items():
            try:
                score = method_func(current, reference)
                drift_scores[method_name] = score
            except Exception as e:
                logging.error(f"Error in {method_name}: {e}")
                drift_scores[method_name] = None
        
        return drift_scores
    
    def ks_test(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Kolmogorov-Smirnov test for drift detection"""
        from scipy import stats
        
        # Aggregate KS statistics across features
        ks_statistics = []
        
        for i in range(current.shape[1]):
            statistic, _ = stats.ks_2samp(
                current[:, i],
                reference[:, i]
            )
            ks_statistics.append(statistic)
        
        return np.mean(ks_statistics)
    
    def psi(self, current: np.ndarray, reference: np.ndarray, buckets: int = 10) -> float:
        """Population Stability Index calculation"""
        psi_values = []
        
        for i in range(current.shape[1]):
            # Create bins
            min_val = min(reference[:, i].min(), current[:, i].min())
            max_val = max(reference[:, i].max(), current[:, i].max())
            bins = np.linspace(min_val, max_val, buckets + 1)
            
            # Calculate distributions
            ref_counts, _ = np.histogram(reference[:, i], bins=bins)
            curr_counts, _ = np.histogram(current[:, i], bins=bins)
            
            # Normalize
            ref_prop = (ref_counts + 1) / (len(reference) + buckets)
            curr_prop = (curr_counts + 1) / (len(current) + buckets)
            
            # Calculate PSI
            psi = np.sum((curr_prop - ref_prop) * np.log(curr_prop / ref_prop))
            psi_values.append(psi)
        
        return np.mean(psi_values)
    
    def jensen_shannon_distance(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Jensen-Shannon distance for distribution comparison"""
        from scipy.spatial.distance import jensenshannon
        
        js_distances = []
        
        for i in range(current.shape[1]):
            # Create probability distributions
            curr_hist, bins = np.histogram(current[:, i], bins=50, density=True)
            ref_hist, _ = np.histogram(reference[:, i], bins=bins, density=True)
            
            # Normalize
            curr_hist = curr_hist / curr_hist.sum()
            ref_hist = ref_hist / ref_hist.sum()
            
            # Calculate JS distance
            js_dist = jensenshannon(curr_hist, ref_hist)
            js_distances.append(js_dist)
        
        return np.mean(js_distances)
    
    def wasserstein_distance(self, current: np.ndarray, reference: np.ndarray) -> float:
        """Wasserstein distance for distribution comparison"""
        from scipy.stats import wasserstein_distance
        
        w_distances = []
        
        for i in range(current.shape[1]):
            w_dist = wasserstein_distance(
                current[:, i],
                reference[:, i]
            )
            w_distances.append(w_dist)
        
        return np.mean(w_distances)


class PerformanceTracker:
    """Track model performance over time"""
    
    def __init__(self):
        self.performance_history = []
        self.alert_thresholds = {
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.75,
            'f1_score': 0.77
        }
    
    def track_performance(self, y_true: np.ndarray, y_pred: np.ndarray, timestamp: datetime = None):
        """Track model performance metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if timestamp is None:
            timestamp = datetime.now()
        
        metrics = {
            'timestamp': timestamp,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        self.performance_history.append(metrics)
        
        # Check for performance degradation
        self.check_performance_degradation(metrics)
        
        return metrics
    
    def check_performance_degradation(self, current_metrics: dict):
        """Check if performance has degraded below thresholds"""
        alerts = []
        
        for metric, threshold in self.alert_thresholds.items():
            if current_metrics[metric] < threshold:
                alerts.append({
                    'metric': metric,
                    'current_value': current_metrics[metric],
                    'threshold': threshold,
                    'severity': 'high' if current_metrics[metric] < threshold * 0.9 else 'medium'
                })
        
        if alerts:
            self.trigger_performance_alert(alerts)
    
    def trigger_performance_alert(self, alerts: list):
        """Trigger alerts for performance degradation"""
        for alert in alerts:
            logging.warning(
                f"Performance degradation detected: {alert['metric']} = {alert['current_value']:.3f} "
                f"(threshold: {alert['threshold']:.3f})"
            )
    
    def get_performance_trend(self, window_size: int = 24) -> dict:
        """Analyze performance trends over time"""
        if len(self.performance_history) < window_size:
            return {}
        
        recent_history = self.performance_history[-window_size:]
        
        trends = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            values = [h[metric] for h in recent_history]
            
            # Calculate trend statistics
            trends[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'trend': self.calculate_trend(values)
            }
        
        return trends
    
    def calculate_trend(self, values: list) -> str:
        """Calculate trend direction"""
        from scipy import stats
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
```

### Logging and Observability

```python
import logging
import json
import traceback
from datetime import datetime
from typing import Any, Dict, Optional
import contextvars
import uuid

# Context variable for request tracking
request_id_var = contextvars.ContextVar('request_id', default=None)

class StructuredLogger:
    """Structured logging for ML systems"""
    
    def __init__(self, logger_name: str, service_name: str):
        self.logger = logging.getLogger(logger_name)
        self.service_name = service_name
        
        # Configure structured logging
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter(service_name))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_prediction(self, 
                      customer_id: str,
                      features: Dict[str, Any],
                      prediction: int,
                      confidence: float,
                      latency: float):
        """Log prediction details"""
        self.logger.info(
            "prediction_made",
            extra={
                'event_type': 'prediction',
                'customer_id': customer_id,
                'features': features,
                'prediction': prediction,
                'confidence': confidence,
                'latency_ms': latency * 1000,
                'request_id': request_id_var.get()
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with context"""
        self.logger.error(
            "error_occurred",
            extra={
                'event_type': 'error',
                'error_type': type(error).__name__,
                'error_message': str(error),
                'stack_trace': traceback.format_exc(),
                'context': context or {},
                'request_id': request_id_var.get()
            }
        )
    
    def log_model_update(self, 
                        old_version: str,
                        new_version: str,
                        metrics: Dict[str, float]):
        """Log model update event"""
        self.logger.info(
            "model_updated",
            extra={
                'event_type': 'model_update',
                'old_version': old_version,
                'new_version': new_version,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def log_drift_detected(self,
                          drift_type: str,
                          drift_score: float,
                          threshold: float):
        """Log drift detection event"""
        self.logger.warning(
            "drift_detected",
            extra={
                'event_type': 'drift',
                'drift_type': drift_type,
                'drift_score': drift_score,
                'threshold': threshold,
                'severity': 'high' if drift_score > threshold * 1.5 else 'medium',
                'timestamp': datetime.now().isoformat()
            }
        )


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'service': self.service_name,
            'message': record.getMessage(),
            'logger': record.name,
            'request_id': request_id_var.get()
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_obj.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_obj)


class RequestTracker:
    """Track requests throughout their lifecycle"""
    
    def __init__(self, logger: StructuredLogger):
        self.logger = logger
    
    def track_request(self):
        """Context manager for request tracking"""
        class RequestContext:
            def __enter__(self_):
                request_id = str(uuid.uuid4())
                request_id_var.set(request_id)
                self.logger.logger.info(
                    "request_started",
                    extra={'request_id': request_id}
                )
                return request_id
            
            def __exit__(self_, exc_type, exc_val, exc_tb):
                if exc_type:
                    self.logger.log_error(exc_val)
                else:
                    self.logger.logger.info(
                        "request_completed",
                        extra={'request_id': request_id_var.get()}
                    )
                request_id_var.set(None)
        
        return RequestContext()
```

---

## Model Management and Versioning

### Complete Model Registry System

```python
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import pickle
import joblib
from dataclasses import dataclass, asdict
import boto3
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    model_name: str
    version: str
    algorithm: str
    framework: str
    created_at: datetime
    created_by: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    description: str
    training_data_version: str
    feature_schema: Dict[str, str]
    model_size_bytes: int
    model_hash: str


class ModelRecord(Base):
    """Database model for model registry"""
    __tablename__ = 'models'
    
    model_id = Column(String, primary_key=True)
    model_name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    algorithm = Column(String)
    framework = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String)
    metrics = Column(JSON)
    parameters = Column(JSON)
    tags = Column(JSON)
    description = Column(String)
    training_data_version = Column(String)
    feature_schema = Column(JSON)
    model_size_bytes = Column(Float)
    model_hash = Column(String)
    status = Column(String, default='development')
    deployed_at = Column(DateTime)
    retired_at = Column(DateTime)


class ModelRegistry:
    """
    Complete model registry for managing ML models.
    """
    
    def __init__(self, 
                 storage_backend: str = 'local',
                 storage_path: str = './model_registry',
                 db_url: str = 'sqlite:///model_registry.db'):
        
        self.storage_backend = storage_backend
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize storage backend
        if storage_backend == 's3':
            self.s3_client = boto3.client('s3')
            self.s3_bucket = 'ml-model-registry'
    
    def register_model(self,
                      model: Any,
                      model_name: str,
                      version: str,
                      metadata: ModelMetadata) -> str:
        """Register a new model"""
        
        # Generate model ID
        model_id = self.generate_model_id(model_name, version)
        metadata.model_id = model_id
        
        # Calculate model hash
        model_bytes = pickle.dumps(model)
        metadata.model_hash = hashlib.sha256(model_bytes).hexdigest()
        metadata.model_size_bytes = len(model_bytes)
        
        # Save model artifact
        artifact_path = self.save_model_artifact(model, model_id)
        
        # Save metadata to database
        session = self.Session()
        try:
            model_record = ModelRecord(**asdict(metadata))
            session.add(model_record)
            session.commit()
            
            logging.info(f"Model registered: {model_id}")
            return model_id
            
        except Exception as e:
            session.rollback()
            # Clean up artifact
            self.delete_model_artifact(model_id)
            raise Exception(f"Failed to register model: {e}")
        finally:
            session.close()
    
    def get_model(self, model_id: str = None, model_name: str = None, version: str = None) -> Any:
        """Retrieve a model from registry"""
        
        if model_id is None:
            model_id = self.generate_model_id(model_name, version)
        
        # Get metadata from database
        session = self.Session()
        try:
            model_record = session.query(ModelRecord).filter_by(model_id=model_id).first()
            
            if not model_record:
                raise ValueError(f"Model not found: {model_id}")
            
            # Load model artifact
            model = self.load_model_artifact(model_id)
            
            return model, model_record
            
        finally:
            session.close()
    
    def promote_model(self, model_id: str, target_stage: str):
        """Promote model to different stage"""
        
        valid_stages = ['development', 'staging', 'production', 'archived']
        if target_stage not in valid_stages:
            raise ValueError(f"Invalid stage: {target_stage}")
        
        session = self.Session()
        try:
            model_record = session.query(ModelRecord).filter_by(model_id=model_id).first()
            
            if not model_record:
                raise ValueError(f"Model not found: {model_id}")
            
            # Update status
            old_status = model_record.status
            model_record.status = target_stage
            
            if target_stage == 'production':
                model_record.deployed_at = datetime.utcnow()
            elif target_stage == 'archived':
                model_record.retired_at = datetime.utcnow()
            
            session.commit()
            
            logging.info(f"Model {model_id} promoted from {old_status} to {target_stage}")
            
            # Trigger deployment if moving to production
            if target_stage == 'production':
                self.trigger_deployment(model_id)
            
        finally:
            session.close()
    
    def list_models(self, 
                   model_name: str = None,
                   status: str = None,
                   tags: List[str] = None) -> List[ModelRecord]:
        """List models based on filters"""
        
        session = self.Session()
        try:
            query = session.query(ModelRecord)
            
            if model_name:
                query = query.filter_by(model_name=model_name)
            
            if status:
                query = query.filter_by(status=status)
            
            if tags:
                # Filter by tags (assuming tags stored as JSON array)
                for tag in tags:
                    query = query.filter(ModelRecord.tags.contains(tag))
            
            return query.all()
            
        finally:
            session.close()
    
    def compare_models(self, model_ids: List[str]) -> Dict:
        """Compare multiple models"""
        
        session = self.Session()
        try:
            models = session.query(ModelRecord).filter(
                ModelRecord.model_id.in_(model_ids)
            ).all()
            
            comparison = {
                'models': [],
                'metrics_comparison': {},
                'best_performing': {}
            }
            
            # Collect model information
            for model in models:
                comparison['models'].append({
                    'model_id': model.model_id,
                    'version': model.version,
                    'metrics': model.metrics,
                    'created_at': model.created_at.isoformat()
                })
            
            # Compare metrics
            if models:
                metric_names = set()
                for model in models:
                    if model.metrics:
                        metric_names.update(model.metrics.keys())
                
                for metric in metric_names:
                    comparison['metrics_comparison'][metric] = []
                    best_value = None
                    best_model = None
                    
                    for model in models:
                        value = model.metrics.get(metric) if model.metrics else None
                        comparison['metrics_comparison'][metric].append({
                            'model_id': model.model_id,
                            'value': value
                        })
                        
                        if value is not None:
                            if best_value is None or value > best_value:
                                best_value = value
                                best_model = model.model_id
                    
                    comparison['best_performing'][metric] = {
                        'model_id': best_model,
                        'value': best_value
                    }
            
            return comparison
            
        finally:
            session.close()
    
    def save_model_artifact(self, model: Any, model_id: str) -> str:
        """Save model artifact to storage"""
        
        if self.storage_backend == 'local':
            artifact_path = self.storage_path / f"{model_id}.pkl"
            joblib.dump(model, artifact_path)
            return str(artifact_path)
            
        elif self.storage_backend == 's3':
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Upload to S3
            s3_key = f"models/{model_id}.pkl"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=model_bytes
            )
            return f"s3://{self.s3_bucket}/{s3_key}"
    
    def load_model_artifact(self, model_id: str) -> Any:
        """Load model artifact from storage"""
        
        if self.storage_backend == 'local':
            artifact_path = self.storage_path / f"{model_id}.pkl"
            return joblib.load(artifact_path)
            
        elif self.storage_backend == 's3':
            s3_key = f"models/{model_id}.pkl"
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=s3_key
            )
            model_bytes = response['Body'].read()
            return pickle.loads(model_bytes)
    
    def generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID"""
        return f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    def trigger_deployment(self, model_id: str):
        """Trigger model deployment to production"""
        # Implement deployment logic
        logging.info(f"Triggering deployment for model: {model_id}")
        # Could trigger CI/CD pipeline, Kubernetes deployment, etc.
```

---

## CI/CD for Machine Learning

### Complete MLOps Pipeline

```yaml
# .gitlab-ci.yml - GitLab CI/CD Pipeline for ML
stages:
  - test
  - train
  - evaluate
  - register
  - deploy
  - monitor

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  MODEL_REGISTRY: "your-registry.com"
  MLFLOW_TRACKING_URI: "http://mlflow.your-domain.com"

# Test Stage
unit-tests:
  stage: test
  image: python:3.9
  before_script:
    - pip install -r requirements-test.txt
  script:
    - pytest tests/unit -v --cov=src --cov-report=xml
    - python -m pylint src/
    - python -m mypy src/
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

data-validation:
  stage: test
  image: python:3.9
  script:
    - python scripts/validate_data.py --data-path=data/raw
    - python scripts/check_data_quality.py
  artifacts:
    paths:
      - data/validation_report.html
    expire_in: 1 week

# Train Stage
train-model:
  stage: train
  image: python:3.9
  script:
    - |
      python train.py \
        --data-path=data/processed \
        --model-type=${MODEL_TYPE} \
        --hyperparameters=config/hyperparameters.yaml \
        --output-path=models/candidate
  artifacts:
    paths:
      - models/candidate/
      - logs/training/
    expire_in: 1 week
  only:
    - main
    - develop

# Evaluate Stage
evaluate-model:
  stage: evaluate
  image: python:3.9
  dependencies:
    - train-model
  script:
    - |
      python evaluate.py \
        --model-path=models/candidate \
        --test-data=data/test \
        --threshold-config=config/thresholds.yaml
  artifacts:
    paths:
      - evaluation/report.html
      - evaluation/metrics.json
    expire_in: 1 week

compare-models:
  stage: evaluate
  image: python:3.9
  dependencies:
    - train-model
  script:
    - |
      python compare_models.py \
        --candidate-model=models/candidate \
        --production-model=${PRODUCTION_MODEL_ID} \
        --comparison-metrics=config/comparison_metrics.yaml
  artifacts:
    paths:
      - comparison/report.html
    expire_in: 1 week

# Register Stage
register-model:
  stage: register
  image: python:3.9
  dependencies:
    - evaluate-model
  script:
    - |
      python register_model.py \
        --model-path=models/candidate \
        --model-name=${CI_PROJECT_NAME} \
        --version=${CI_COMMIT_SHA} \
        --metrics-path=evaluation/metrics.json
  only:
    - main
  when: manual

# Deploy Stage
deploy-staging:
  stage: deploy
  image: google/cloud-sdk:latest
  dependencies:
    - register-model
  environment:
    name: staging
    url: https://staging.your-domain.com
  script:
    - |
      # Deploy to staging environment
      kubectl set image deployment/ml-model \
        ml-model=${MODEL_REGISTRY}/${CI_PROJECT_NAME}:${CI_COMMIT_SHA} \
        -n staging
      
      # Wait for rollout
      kubectl rollout status deployment/ml-model -n staging
      
      # Run smoke tests
      python scripts/smoke_tests.py --env=staging
  only:
    - main

deploy-production:
  stage: deploy
  image: google/cloud-sdk:latest
  dependencies:
    - deploy-staging
  environment:
    name: production
    url: https://api.your-domain.com
  script:
    - |
      # Blue-green deployment
      python scripts/blue_green_deploy.py \
        --model-id=${CI_COMMIT_SHA} \
        --environment=production \
        --strategy=blue-green
      
      # Run health checks
      python scripts/health_checks.py --env=production
      
      # Update monitoring dashboards
      python scripts/update_dashboards.py
  only:
    - main
  when: manual

# Monitor Stage
monitor-deployment:
  stage: monitor
  image: python:3.9
  dependencies:
    - deploy-production
  script:
    - |
      # Monitor model performance
      python scripts/monitor_performance.py \
        --model-id=${CI_COMMIT_SHA} \
        --duration=1h \
        --metrics=config/monitoring_metrics.yaml
      
      # Check for drift
      python scripts/check_drift.py \
        --model-id=${CI_COMMIT_SHA} \
        --reference-data=data/reference
  only:
    - main
```

### GitHub Actions Workflow

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * *'  # Daily retraining

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml

  train:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Download training data
        run: |
          aws s3 sync s3://ml-data/training data/training
      
      - name: Train model
        run: |
          python train.py \
            --data-path=data/training \
            --config=config/training_config.yaml
      
      - name: Upload model artifact
        uses: actions/upload-artifact@v3
        with:
          name: trained-model
          path: models/

  evaluate:
    needs: train
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model artifact
        uses: actions/download-artifact@v3
        with:
          name: trained-model
          path: models/
      
      - name: Evaluate model
        run: |
          python evaluate.py \
            --model-path=models/ \
            --test-data=data/test
      
      - name: Check thresholds
        run: |
          python scripts/check_thresholds.py \
            --metrics-file=evaluation/metrics.json \
            --thresholds-file=config/thresholds.yaml

  deploy:
    needs: evaluate
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to production
        run: |
          echo "Deploying model to production"
          # Deployment logic here
```

---

## Production Best Practices

### A/B Testing Framework

```python
class ABTestingFramework:
    """
    A/B testing framework for ML models in production.
    """
    
    def __init__(self, 
                 control_model,
                 treatment_model,
                 traffic_split: float = 0.5):
        self.control_model = control_model
        self.treatment_model = treatment_model
        self.traffic_split = traffic_split
        self.results = {
            'control': [],
            'treatment': []
        }
    
    def route_traffic(self, request_id: str) -> str:
        """Route traffic between models"""
        # Use hash of request ID for consistent routing
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        
        if (hash_value % 100) / 100 < self.traffic_split:
            return 'treatment'
        else:
            return 'control'
    
    def predict(self, features, request_id: str):
        """Make prediction with appropriate model"""
        variant = self.route_traffic(request_id)
        
        if variant == 'control':
            prediction = self.control_model.predict(features)
            model_version = 'control'
        else:
            prediction = self.treatment_model.predict(features)
            model_version = 'treatment'
        
        # Log for analysis
        self.log_prediction(request_id, variant, prediction)
        
        return prediction, model_version
    
    def analyze_results(self, metric: str = 'accuracy') -> Dict:
        """Analyze A/B test results"""
        from scipy import stats
        
        control_metrics = [r[metric] for r in self.results['control']]
        treatment_metrics = [r[metric] for r in self.results['treatment']]
        
        # Perform statistical test
        t_stat, p_value = stats.ttest_ind(
            treatment_metrics,
            control_metrics
        )
        
        # Calculate effect size
        control_mean = np.mean(control_metrics)
        treatment_mean = np.mean(treatment_metrics)
        pooled_std = np.sqrt(
            (np.std(control_metrics)**2 + np.std(treatment_metrics)**2) / 2
        )
        effect_size = (treatment_mean - control_mean) / pooled_std
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'lift': (treatment_mean - control_mean) / control_mean,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': effect_size,
            'sample_size': {
                'control': len(control_metrics),
                'treatment': len(treatment_metrics)
            }
        }
```

### Feature Store Integration

```python
class FeatureStore:
    """
    Feature store for consistent feature engineering.
    """
    
    def __init__(self, 
                 online_store: str = 'redis',
                 offline_store: str = 'postgres'):
        self.online_store = self.init_online_store(online_store)
        self.offline_store = self.init_offline_store(offline_store)
        self.feature_definitions = {}
    
    def register_feature(self,
                        name: str,
                        description: str,
                        computation: callable,
                        dependencies: List[str] = None):
        """Register a feature definition"""
        self.feature_definitions[name] = {
            'description': description,
            'computation': computation,
            'dependencies': dependencies or [],
            'version': self.generate_version(computation)
        }
    
    def compute_features(self, entity_id: str, features: List[str]) -> Dict:
        """Compute features for an entity"""
        computed_features = {}
        
        # Check online store first
        cached = self.get_from_online_store(entity_id, features)
        computed_features.update(cached)
        
        # Compute missing features
        missing = set(features) - set(cached.keys())
        for feature_name in missing:
            if feature_name in self.feature_definitions:
                value = self.compute_feature(entity_id, feature_name)
                computed_features[feature_name] = value
                
                # Cache in online store
                self.store_online(entity_id, feature_name, value)
        
        return computed_features
    
    def compute_feature(self, entity_id: str, feature_name: str) -> Any:
        """Compute a single feature"""
        feature_def = self.feature_definitions[feature_name]
        
        # Get dependencies
        dependencies = {}
        for dep in feature_def['dependencies']:
            dependencies[dep] = self.compute_feature(entity_id, dep)
        
        # Compute feature
        return feature_def['computation'](entity_id, dependencies)
    
    def materialize_features(self, 
                           features: List[str],
                           entity_ids: List[str] = None,
                           start_date: datetime = None,
                           end_date: datetime = None):
        """Materialize features to offline store"""
        # Implementation for batch materialization
        pass
```

### Model Serving Optimization

```python
class OptimizedModelServer:
    """
    Optimized model server with caching, batching, and pooling.
    """
    
    def __init__(self,
                 model,
                 batch_size: int = 32,
                 batch_timeout: float = 0.1,
                 cache_size: int = 10000):
        self.model = model
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Initialize cache
        self.cache = LRUCache(cache_size)
        
        # Initialize batching
        self.batch_queue = asyncio.Queue()
        self.pending_requests = {}
        
        # Start batch processor
        asyncio.create_task(self.batch_processor())
    
    async def predict(self, features: np.ndarray) -> np.ndarray:
        """Make prediction with caching and batching"""
        
        # Check cache
        cache_key = self.get_cache_key(features)
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Add to batch queue
        request_id = str(uuid.uuid4())
        future = asyncio.Future()
        
        await self.batch_queue.put({
            'id': request_id,
            'features': features,
            'future': future
        })
        
        # Wait for result
        result = await future
        
        # Cache result
        self.cache.put(cache_key, result)
        
        return result
    
    async def batch_processor(self):
        """Process predictions in batches"""
        while True:
            batch = []
            futures = []
            
            # Collect batch
            try:
                # Wait for first item
                item = await self.batch_queue.get()
                batch.append(item['features'])
                futures.append(item['future'])
                
                # Collect more items up to batch size
                deadline = time.time() + self.batch_timeout
                while len(batch) < self.batch_size and time.time() < deadline:
                    try:
                        item = await asyncio.wait_for(
                            self.batch_queue.get(),
                            timeout=deadline - time.time()
                        )
                        batch.append(item['features'])
                        futures.append(item['future'])
                    except asyncio.TimeoutError:
                        break
                
                # Process batch
                if batch:
                    batch_array = np.array(batch)
                    predictions = await self.run_inference(batch_array)
                    
                    # Return results
                    for future, prediction in zip(futures, predictions):
                        future.set_result(prediction)
                        
            except Exception as e:
                # Handle errors
                for future in futures:
                    future.set_exception(e)
    
    async def run_inference(self, batch: np.ndarray) -> np.ndarray:
        """Run model inference"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.model.predict,
            batch
        )
    
    def get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key for features"""
        return hashlib.md5(features.tobytes()).hexdigest()
```

---

## Conclusion

This comprehensive MLOps guide provides:

1. **Complete Infrastructure**: Docker, Kubernetes, and cloud deployment strategies
2. **Production Monitoring**: Comprehensive metrics, drift detection, and alerting
3. **Model Registry**: Full versioning and lifecycle management
4. **CI/CD Pipelines**: Complete automation for training, evaluation, and deployment
5. **Best Practices**: A/B testing, feature stores, and optimization techniques

Key takeaways:
- MLOps requires automation at every stage of the ML lifecycle
- Monitoring and observability are critical for production ML systems
- Version control extends beyond code to data and models
- CI/CD pipelines must handle the unique requirements of ML workflows
- Production deployment requires careful consideration of performance, reliability, and maintainability

The practices and code examples provided here form a solid foundation for building production-ready ML systems that are scalable, maintainable, and reliable.
                probability = self.model.predict_proba([features])[0].max()

                # Send result
                result = {
                    'customer_id': data['customer_id'],
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'timestamp': datetime.now().isoformat()
                }

                self.producer.send(self.output_topic, result)

            except Exception as e:
                print(f"Error processing message: {e}")

    def extract_features(self, data):
        # Feature extraction logic
        pass

# Usage
predictor = StreamingPredictor(
    'models/churn_model.pkl',
    'customer_events',
    'churn_predictions'
)
predictor.process_stream()
```

### 4. Edge Deployment

#### TensorFlow Lite for Mobile
```python
import tensorflow as tf

class ModelOptimizer:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def convert_to_tflite(self, output_path, quantize=True):
        """Convert model to TensorFlow Lite format"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        return output_path

    def convert_to_tflite_quantized(self, output_path, representative_dataset):
        """Convert with post-training quantization"""
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_quantized_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_quantized_model)

        return output_path

# Usage
optimizer = ModelOptimizer('models/tensorflow_model.h5')
optimizer.convert_to_tflite('models/model.tflite', quantize=True)
```

#### ONNX for Cross-Platform Deployment
```python
import torch
import torch.onnx
import onnxruntime as ort
import numpy as np

class ONNXConverter:
    def __init__(self, pytorch_model):
        self.model = pytorch_model
        self.model.eval()

    def convert_to_onnx(self, output_path, input_shape):
        """Convert PyTorch model to ONNX format"""
        dummy_input = torch.randn(1, *input_shape)

        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

    def optimize_onnx_model(self, input_path, output_path):
        """Optimize ONNX model for inference"""
        from onnxruntime.tools import optimizer

        optimizer.optimize_model(
            input_path,
            output_path,
            ['eliminate_dropout', 'eliminate_identity']
        )

class ONNXInference:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_data):
        """Run inference with ONNX model"""
        result = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return result[0]
```

---

## Infrastructure and Orchestration

### 1. Kubernetes Deployment

#### Model Deployment Manifest
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-deployment
  labels:
    app: ml-model
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ml-model
  template:
    metadata:
      labels:
        app: ml-model
    spec:
      containers:
      - name: ml-model
        image: your-registry/ml-model:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/models/churn_model.pkl"
        - name: MODEL_VERSION
          value: "v1.0.0"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
spec:
  selector:
    app: ml-model
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-deployment
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Istio Service Mesh Configuration
```yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model-vs
spec:
  http:
  - match:
    - headers:
        model-version:
          exact: v1.0.0
    route:
    - destination:
        host: ml-model-service
        subset: v1
      weight: 90
    - destination:
        host: ml-model-service
        subset: v2
      weight: 10
  - route:
    - destination:
        host: ml-model-service
        subset: v1
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ml-model-dr
spec:
  host: ml-model-service
  subsets:
  - name: v1
    labels:
      version: v1.0.0
  - name: v2
    labels:
      version: v2.0.0
```

### 2. Kubeflow Pipelines

#### ML Pipeline Definition
```python
import kfp
from kfp import dsl
from kfp.components import func_to_container_op

@func_to_container_op
def data_preprocessing(input_path: str, output_path: str):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    import pickle

    # Load data
    df = pd.read_csv(input_path)

    # Preprocessing steps
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.drop('target', axis=1))

    # Save processed data and scaler
    processed_df = pd.DataFrame(scaled_features, columns=df.columns[:-1])
    processed_df['target'] = df['target']
    processed_df.to_csv(output_path, index=False)

    with open(f"{output_path}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

@func_to_container_op
def model_training(input_path: str, model_output_path: str):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import pickle

    # Load data
    df = pd.read_csv(input_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate model
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Model accuracy: {accuracy}")

    # Save model
    with open(model_output_path, 'wb') as f:
        pickle.dump(model, f)

@func_to_container_op
def model_validation(model_path: str, test_data_path: str) -> bool:
    import pandas as pd
    import pickle
    from sklearn.metrics import accuracy_score

    # Load model and test data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_csv(test_data_path)
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']

    # Validate model
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Return validation result
    return accuracy > 0.85

@dsl.pipeline(
    name='ML Training Pipeline',
    description='A pipeline that trains and validates an ML model'
)
def ml_training_pipeline(
    input_data_path: str = '/data/raw/train.csv',
    test_data_path: str = '/data/raw/test.csv',
    processed_data_path: str = '/data/processed/train.csv',
    model_output_path: str = '/models/model.pkl'
):
    # Data preprocessing step
    preprocess_task = data_preprocessing(input_data_path, processed_data_path)

    # Model training step
    training_task = model_training(processed_data_path, model_output_path)
    training_task.after(preprocess_task)

    # Model validation step
    validation_task = model_validation(model_output_path, test_data_path)
    validation_task.after(training_task)

# Compile and submit pipeline
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(ml_training_pipeline, 'ml_pipeline.yaml')
```

### 3. Docker and Container Management

#### Multi-stage Dockerfile
```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "src.main"]
```

#### Docker Compose for Development
```yaml
version: '3.8'

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/churn_model.pkl
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://user:password@postgres:5432/mldb
    depends_on:
      - redis
      - postgres
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    networks:
      - ml-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - ml-network

  postgres:
    image: postgres:14
    environment:
      - POSTGRES_DB=mldb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - ml-network

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    networks:
      - ml-network

volumes:
  postgres_data:
  grafana_data:

networks:
  ml-network:
    driver: bridge
```

---

## Monitoring and Observability

### 1. Model Performance Monitoring

#### Performance Metrics Tracking
```python
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time
import numpy as np
from datetime import datetime

class ModelMonitor:
    def __init__(self):
        # Prometheus metrics
        self.prediction_counter = Counter(
            'ml_predictions_total',
            'Total number of predictions made',
            ['model_version', 'endpoint']
        )

        self.prediction_latency = Histogram(
            'ml_prediction_latency_seconds',
            'Time spent making predictions',
            ['model_version', 'endpoint']
        )

        self.model_accuracy = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_version']
        )

        self.error_counter = Counter(
            'ml_prediction_errors_total',
            'Total number of prediction errors',
            ['model_version', 'endpoint', 'error_type']
        )

        # Initialize drift detection
        self.drift_detector = DriftDetector()

    def track_prediction(self, model_version, endpoint, latency, error=None):
        """Track a single prediction"""
        self.prediction_counter.labels(
            model_version=model_version,
            endpoint=endpoint
        ).inc()

        self.prediction_latency.labels(
            model_version=model_version,
            endpoint=endpoint
        ).observe(latency)

        if error:
            self.error_counter.labels(
                model_version=model_version,
                endpoint=endpoint,
                error_type=type(error).__name__
            ).inc()

    def update_model_accuracy(self, model_version, accuracy):
        """Update model accuracy metric"""
        self.model_accuracy.labels(model_version=model_version).set(accuracy)

    def check_drift(self, reference_data, current_data):
        """Check for data drift"""
        return self.drift_detector.detect_drift(reference_data, current_data)

class DriftDetector:
    def __init__(self, threshold=0.05):
        self.threshold = threshold

    def detect_drift(self, reference_data, current_data):
        """Detect statistical drift using KS test"""
        from scipy import stats

        drift_results = {}

        for column in reference_data.columns:
            if reference_data[column].dtype in ['float64', 'int64']:
                statistic, p_value = stats.ks_2samp(
                    reference_data[column],
                    current_data[column]
                )

                drift_results[column] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < self.threshold
                }

        return drift_results
```

### 2. Data Quality Monitoring

#### Data Validation Pipeline
```python
import pandas as pd
from typing import Dict, Any, List
import logging

class DataQualityMonitor:
    def __init__(self):
        self.quality_checks = [
            self.check_schema,
            self.check_completeness,
            self.check_validity,
            self.check_consistency,
            self.check_anomalies
        ]

    def validate_data(self, data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive data quality checks"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(data),
            'checks': {}
        }

        for check in self.quality_checks:
            try:
                check_result = check(data, schema)
                results['checks'][check.__name__] = check_result
            except Exception as e:
                logging.error(f"Error in {check.__name__}: {e}")
                results['checks'][check.__name__] = {'status': 'error', 'message': str(e)}

        # Overall quality score
        passed_checks = sum(1 for check in results['checks'].values()
                           if check.get('status') == 'passed')
        results['quality_score'] = passed_checks / len(self.quality_checks)

        return results

    def check_schema(self, data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data schema"""
        issues = []

        # Check required columns
        required_columns = schema.get('required_columns', [])
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            issues.append(f"Missing columns: {missing_columns}")

        # Check data types
        expected_types = schema.get('column_types', {})
        for column, expected_type in expected_types.items():
            if column in data.columns:
                actual_type = str(data[column].dtype)
                if actual_type != expected_type:
                    issues.append(f"Column {column}: expected {expected_type}, got {actual_type}")

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues,
            'columns_checked': len(expected_types)
        }

    def check_completeness(self, data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check data completeness"""
        completeness_threshold = schema.get('completeness_threshold', 0.95)
        issues = []

        for column in data.columns:
            completeness = 1 - (data[column].isnull().sum() / len(data))
            if completeness < completeness_threshold:
                issues.append(f"Column {column}: {completeness:.2%} complete (threshold: {completeness_threshold:.2%})")

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues,
            'overall_completeness': 1 - (data.isnull().sum().sum() / data.size)
        }

    def check_validity(self, data: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Check data validity against business rules"""
        validity_rules = schema.get('validity_rules', {})
        issues = []

        for column, rules in validity_rules.items():
            if column not in data.columns:
                continue

            # Range checks
            if 'min_value' in rules:
                invalid_count = (data[column] < rules['min_value']).sum()
                if invalid_count > 0:
                    issues.append(f"Column {column}: {invalid_count} values below minimum {rules['min_value']}")

            if 'max_value' in rules:
                invalid_count = (data[column] > rules['max_value']).sum()
                if invalid_count > 0:
                    issues.append(f"Column {column}: {invalid_count} values above maximum {rules['max_value']}")

            # Pattern checks
            if 'pattern' in rules:
                import re
                pattern = re.compile(rules['pattern'])
                invalid_count = ~data[column].astype(str).str.match(pattern).sum()
                if invalid_count > 0:
                    issues.append(f"Column {column}: {invalid_count} values don't match pattern")

        return {
            'status': 'passed' if not issues else 'failed',
            'issues': issues,
            'rules_checked': len(validity_rules)
        }

# Great Expectations Integration
import great_expectations as ge

class GreatExpectationsValidator:
    def __init__(self, context_path):
        self.context = ge.data_context.DataContext(context_path)

    def create_expectation_suite(self, suite_name, data_source):
        """Create expectation suite for data validation"""
        suite = self.context.create_expectation_suite(suite_name)

        # Define expectations
        expectations = [
            {
                "expectation_type": "expect_table_row_count_to_be_between",
                "kwargs": {"min_value": 1000, "max_value": 100000}
            },
            {
                "expectation_type": "expect_column_values_to_not_be_null",
                "kwargs": {"column": "customer_id"}
            },
            {
                "expectation_type": "expect_column_values_to_be_between",
                "kwargs": {"column": "age", "min_value": 18, "max_value": 100}
            }
        ]

        for expectation in expectations:
            suite.add_expectation(expectation)

        self.context.save_expectation_suite(suite, suite_name)
        return suite

    def validate_data(self, data_asset, expectation_suite_name):
        """Validate data against expectation suite"""
        return self.context.run_validation_operator(
            "action_list_operator",
            assets_to_validate=[data_asset],
            run_id=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            expectation_suite_name=expectation_suite_name
        )
```

---

## Model Management and Versioning

### 1. Model Registry

#### MLflow Model Registry
```python
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

class ModelRegistry:
    def __init__(self, registry_uri):
        mlflow.set_registry_uri(registry_uri)
        self.client = MlflowClient()

    def register_model(self, model_uri, model_name, description=None):
        """Register a new model version"""
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            description=description
        )
        return model_version

    def promote_model(self, model_name, version, stage):
        """Promote model to different stage"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=False
        )

    def get_model_version(self, model_name, stage):
        """Get latest model version for stage"""
        model_versions = self.client.get_latest_versions(
            name=model_name,
            stages=[stage]
        )
        return model_versions[0] if model_versions else None

    def compare_models(self, model_name, version1, version2):
        """Compare two model versions"""
        mv1 = self.client.get_model_version(model_name, version1)
        mv2 = self.client.get_model_version(model_name, version2)

        # Get run information for comparison
        run1 = self.client.get_run(mv1.run_id)
        run2 = self.client.get_run(mv2.run_id)

        comparison = {
            'version1': {
                'version': version1,
                'metrics': run1.data.metrics,
                'parameters': run1.data.params,
                'tags': run1.data.tags
            },
            'version2': {
                'version': version2,
                'metrics': run2.data.metrics,
                'parameters': run2.data.params,
                'tags': run2.data.tags
            }
        }

        return comparison

# Usage example
registry = ModelRegistry("sqlite:///mlflow_registry.db")

# Register model
model_version = registry.register_model(
    model_uri="runs:/abc123/model",
    model_name="customer_churn_model",
    description="Random Forest model for customer churn prediction"
)

# Promote to staging
registry.promote_model("customer_churn_model", model_version.version, "Staging")
```

### 2. A/B Testing Framework

#### Model A/B Testing
```python
import random
import hashlib
from typing import Dict, Any, Optional

class ABTestingFramework:
    def __init__(self):
        self.experiments = {}

    def create_experiment(self, experiment_id: str, model_configs: Dict[str, Any],
                         traffic_split: Dict[str, float], success_metric: str):
        """Create a new A/B testing experiment"""

        # Validate traffic split
        if abs(sum(traffic_split.values()) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")

        self.experiments[experiment_id] = {
            'model_configs': model_configs,
            'traffic_split': traffic_split,
            'success_metric': success_metric,
            'results': {variant: [] for variant in model_configs.keys()},
            'status': 'active'
        }

    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Assign user to experiment variant using consistent hashing"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]

        # Use consistent hashing for stable assignments
        hash_input = f"{experiment_id}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        normalized_hash = (hash_value % 10000) / 10000.0

        # Assign based on traffic split
        cumulative_probability = 0
        for variant, probability in experiment['traffic_split'].items():
            cumulative_probability += probability
            if normalized_hash <= cumulative_probability:
                return variant

        # Fallback to first variant
        return list(experiment['model_configs'].keys())[0]

    def record_result(self, experiment_id: str, variant: str, result: float):
        """Record experiment result"""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]['results'][variant].append(result)

    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment = self.experiments[experiment_id]
        results = experiment['results']

        analysis = {
            'experiment_id': experiment_id,
            'variants': {}
        }

        for variant, values in results.items():
            if not values:
                continue

            analysis['variants'][variant] = {
                'sample_size': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'confidence_interval': self.calculate_confidence_interval(values)
            }

        # Statistical significance testing
        if len(results) == 2:
            variants = list(results.keys())
            control_results = results[variants[0]]
            treatment_results = results[variants[1]]

            if control_results and treatment_results:
                analysis['significance_test'] = self.t_test(
                    control_results, treatment_results
                )

        return analysis

    def calculate_confidence_interval(self, values, confidence=0.95):
        """Calculate confidence interval"""
        import scipy.stats as stats

        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)
        interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

        return {
            'lower': mean - interval,
            'upper': mean + interval,
            'confidence': confidence
        }

    def t_test(self, control, treatment):
        """Perform t-test for statistical significance"""
        from scipy import stats

        statistic, p_value = stats.ttest_ind(control, treatment)

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'confidence_level': 0.95
        }

# Usage example
ab_test = ABTestingFramework()

# Create experiment
ab_test.create_experiment(
    experiment_id="churn_model_v2_test",
    model_configs={
        'control': {'model_version': 'v1.0', 'model_path': 'models/v1.0/model.pkl'},
        'treatment': {'model_version': 'v2.0', 'model_path': 'models/v2.0/model.pkl'}
    },
    traffic_split={'control': 0.5, 'treatment': 0.5},
    success_metric='prediction_accuracy'
)

# Assign user to variant
user_variant = ab_test.assign_variant("churn_model_v2_test", "user_12345")
```

---

## CI/CD for Machine Learning

### 1. GitHub Actions Workflow

#### ML CI/CD Pipeline
```yaml
name: ML Model CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  MODEL_REGISTRY_URI: ${{ secrets.MODEL_REGISTRY_URI }}
  AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install great-expectations dvc

    - name: Validate data schema
      run: |
        python scripts/validate_data.py --data-path data/raw/

    - name: Run Great Expectations
      run: |
        great_expectations checkpoint run data_validation_checkpoint

  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install mlflow boto3

    - name: Pull data with DVC
      run: |
        dvc pull

    - name: Train model
      run: |
        python scripts/train_model.py \
          --experiment-name "github-actions-training" \
          --data-path data/processed/train.csv

    - name: Register model
      run: |
        python scripts/register_model.py \
          --model-name "customer_churn_model" \
          --stage "Staging"

  model-testing:
    needs: model-training
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest

    - name: Run model tests
      run: |
        pytest tests/model_tests/ -v

    - name: Model performance validation
      run: |
        python scripts/validate_model_performance.py \
          --model-name "customer_churn_model" \
          --stage "Staging" \
          --min-accuracy 0.85

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Run security scan
      uses: snyk/actions/python@master
      with:
        args: --severity-threshold=medium
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

  build-and-push:
    needs: [model-testing, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v3

    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: us-east-1

    - name: Login to Amazon ECR
      id: login-ecr
      uses: aws-actions/amazon-ecr-login@v1

    - name: Build and push Docker image
      run: |
        docker build -t ml-model .
        docker tag ml-model:latest $ECR_REGISTRY/ml-model:$GITHUB_SHA
        docker push $ECR_REGISTRY/ml-model:$GITHUB_SHA

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    steps:
    - name: Deploy to staging
      run: |
        kubectl set image deployment/ml-model-deployment \
          ml-model=$ECR_REGISTRY/ml-model:$GITHUB_SHA \
          --namespace=staging

  integration-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Run integration tests
      run: |
        python tests/integration_tests/test_api.py \
          --base-url https://staging-api.example.com

  deploy-production:
    needs: integration-tests
    runs-on: ubuntu-latest
    environment: production
    if: github.ref == 'refs/heads/main'
    steps:
    - name: Promote model to production
      run: |
        python scripts/promote_model.py \
          --model-name "customer_churn_model" \
          --from-stage "Staging" \
          --to-stage "Production"

    - name: Deploy to production
      run: |
        kubectl set image deployment/ml-model-deployment \
          ml-model=$ECR_REGISTRY/ml-model:$GITHUB_SHA \
          --namespace=production
```

### 2. Model Testing Framework

#### Comprehensive Model Testing
```python
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ModelTestSuite:
    def __init__(self, model_path, test_data_path):
        self.model = joblib.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
        self.X_test = self.test_data.drop('target', axis=1)
        self.y_test = self.test_data['target']
        self.y_pred = self.model.predict(self.X_test)
        self.y_proba = self.model.predict_proba(self.X_test)

    def test_model_accuracy(self, min_accuracy=0.85):
        """Test model meets minimum accuracy requirement"""
        accuracy = accuracy_score(self.y_test, self.y_pred)
        assert accuracy >= min_accuracy, f"Model accuracy {accuracy:.3f} below threshold {min_accuracy}"

    def test_model_precision(self, min_precision=0.80):
        """Test model meets minimum precision requirement"""
        precision = precision_score(self.y_test, self.y_pred, average='weighted')
        assert precision >= min_precision, f"Model precision {precision:.3f} below threshold {min_precision}"

    def test_model_recall(self, min_recall=0.80):
        """Test model meets minimum recall requirement"""
        recall = recall_score(self.y_test, self.y_pred, average='weighted')
        assert recall >= min_recall, f"Model recall {recall:.3f} below threshold {min_recall}"

    def test_prediction_range(self):
        """Test predictions are within expected range"""
        unique_predictions = np.unique(self.y_pred)
        expected_classes = np.unique(self.y_test)

        assert set(unique_predictions).issubset(set(expected_classes)), \
            f"Model predictions {unique_predictions} outside expected classes {expected_classes}"

    def test_probability_range(self):
        """Test probability predictions are in valid range [0, 1]"""
        assert np.all(self.y_proba >= 0), "Found negative probabilities"
        assert np.all(self.y_proba <= 1), "Found probabilities > 1"
        assert np.allclose(np.sum(self.y_proba, axis=1), 1), "Probabilities don't sum to 1"

    def test_feature_importance(self):
        """Test feature importance makes business sense"""
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.X_test.columns
            importances = self.model.feature_importances_

            # Check no single feature dominates
            max_importance = np.max(importances)
            assert max_importance < 0.8, f"Single feature has {max_importance:.3f} importance (too high)"

            # Check important features are reasonable
            top_features = feature_names[np.argsort(importances)[-5:]]
            print(f"Top 5 features: {list(top_features)}")

    def test_model_stability(self, n_runs=5):
        """Test model predictions are stable across runs"""
        predictions = []

        for _ in range(n_runs):
            # Add small random noise to test stability
            X_noisy = self.X_test + np.random.normal(0, 0.001, self.X_test.shape)
            pred = self.model.predict(X_noisy)
            predictions.append(pred)

        # Calculate prediction variance
        prediction_variance = np.var(predictions, axis=0).mean()
        assert prediction_variance < 0.01, f"Model predictions too unstable: variance {prediction_variance}"

    def test_no_data_leakage(self):
        """Test for potential data leakage"""
        # Check for suspiciously high accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        assert accuracy < 0.99, f"Suspiciously high accuracy {accuracy:.3f} - possible data leakage"

        # Check for perfect correlation with target
        correlations = self.X_test.corrwith(pd.Series(self.y_test))
        max_correlation = correlations.abs().max()
        assert max_correlation < 0.95, f"Feature has {max_correlation:.3f} correlation with target - possible leakage"

# Pytest fixtures and tests
@pytest.fixture
def model_test_suite():
    return ModelTestSuite(
        model_path="models/churn_model.pkl",
        test_data_path="data/test/test.csv"
    )

def test_model_performance(model_test_suite):
    """Test suite for model performance"""
    model_test_suite.test_model_accuracy()
    model_test_suite.test_model_precision()
    model_test_suite.test_model_recall()

def test_model_outputs(model_test_suite):
    """Test suite for model outputs"""
    model_test_suite.test_prediction_range()
    model_test_suite.test_probability_range()

def test_model_robustness(model_test_suite):
    """Test suite for model robustness"""
    model_test_suite.test_model_stability()
    model_test_suite.test_no_data_leakage()
```

---

## Production Best Practices

### 1. Error Handling and Resilience

#### Robust Prediction Service
```python
import logging
import time
from functools import wraps
from typing import Optional, Dict, Any
import circuit_breaker

class ModelService:
    def __init__(self, model_path: str, fallback_model_path: Optional[str] = None):
        self.model = self.load_model(model_path)
        self.fallback_model = self.load_model(fallback_model_path) if fallback_model_path else None
        self.circuit_breaker = circuit_breaker.CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self.request_count = 0
        self.error_count = 0

    def load_model(self, model_path: str):
        """Load model with error handling"""
        try:
            import joblib
            return joblib.load(model_path)
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")
            raise

    @circuit_breaker
    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction with circuit breaker"""
        start_time = time.time()

        try:
            self.request_count += 1

            # Validate input
            validated_features = self.validate_input(features)

            # Make prediction
            prediction = self.model.predict([validated_features])[0]
            probability = self.model.predict_proba([validated_features])[0].max()

            # Validate output
            self.validate_output(prediction, probability)

            latency = time.time() - start_time

            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'latency': latency,
                'model_version': '1.0.0',
                'status': 'success'
            }

        except Exception as e:
            self.error_count += 1
            logging.error(f"Prediction failed: {e}")

            # Try fallback model
            if self.fallback_model:
                try:
                    return self.fallback_predict(features)
                except Exception as fallback_error:
                    logging.error(f"Fallback prediction failed: {fallback_error}")

            # Return error response
            return {
                'prediction': None,
                'probability': None,
                'latency': time.time() - start_time,
                'model_version': '1.0.0',
                'status': 'error',
                'error': str(e)
            }

    def fallback_predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction using simpler model"""
        validated_features = self.validate_input(features)
        prediction = self.fallback_model.predict([validated_features])[0]
        probability = self.fallback_model.predict_proba([validated_features])[0].max()

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'model_version': 'fallback',
            'status': 'fallback'
        }

    def validate_input(self, features: Dict[str, Any]) -> list:
        """Validate and transform input features"""
        required_features = ['age', 'income', 'tenure', 'usage']

        # Check required features
        missing_features = set(required_features) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Type validation and conversion
        validated = []
        for feature in required_features:
            value = features[feature]

            # Convert to float and validate range
            try:
                float_value = float(value)
                if feature == 'age' and not (18 <= float_value <= 100):
                    raise ValueError(f"Age {float_value} out of valid range [18, 100]")
                elif feature == 'income' and float_value < 0:
                    raise ValueError(f"Income {float_value} cannot be negative")
                elif feature == 'tenure' and float_value < 0:
                    raise ValueError(f"Tenure {float_value} cannot be negative")
                elif feature == 'usage' and float_value < 0:
                    raise ValueError(f"Usage {float_value} cannot be negative")

                validated.append(float_value)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid value for {feature}: {value}")

        return validated

    def validate_output(self, prediction: Any, probability: float):
        """Validate model output"""
        if prediction not in [0, 1]:
            raise ValueError(f"Invalid prediction value: {prediction}")

        if not (0 <= probability <= 1):
            raise ValueError(f"Invalid probability value: {probability}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status"""
        error_rate = self.error_count / max(self.request_count, 1)

        return {
            'status': 'healthy' if error_rate < 0.05 else 'degraded',
            'total_requests': self.request_count,
            'error_count': self.error_count,
            'error_rate': error_rate,
            'circuit_breaker_state': self.circuit_breaker.state,
            'model_loaded': self.model is not None,
            'fallback_available': self.fallback_model is not None
        }

# Circuit Breaker Implementation
class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = func(*args, **kwargs)
                self.on_success()
                return result
            except Exception as e:
                self.on_failure()
                raise

        return wrapper

    def on_success(self):
        self.failure_count = 0
        self.state = 'CLOSED'

    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
```

### 2. Model Versioning and Rollback

#### Safe Deployment Strategies
```python
class ModelDeploymentManager:
    def __init__(self, registry_client, kubernetes_client):
        self.registry = registry_client
        self.k8s = kubernetes_client
        self.deployment_history = []

    def deploy_model(self, model_name: str, version: str,
                    deployment_strategy: str = 'blue_green'):
        """Deploy model using specified strategy"""

        deployment_config = {
            'model_name': model_name,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'strategy': deployment_strategy
        }

        try:
            if deployment_strategy == 'blue_green':
                self.blue_green_deployment(model_name, version)
            elif deployment_strategy == 'canary':
                self.canary_deployment(model_name, version)
            elif deployment_strategy == 'rolling':
                self.rolling_deployment(model_name, version)
            else:
                raise ValueError(f"Unknown deployment strategy: {deployment_strategy}")

            deployment_config['status'] = 'success'
            self.deployment_history.append(deployment_config)

        except Exception as e:
            deployment_config['status'] = 'failed'
            deployment_config['error'] = str(e)
            self.deployment_history.append(deployment_config)

            # Automatic rollback on failure
            if self.deployment_history:
                last_successful = self.get_last_successful_deployment()
                if last_successful:
                    self.rollback_to_version(last_successful['version'])

            raise

    def blue_green_deployment(self, model_name: str, version: str):
        """Blue-green deployment strategy"""

        # Create new deployment (green)
        green_deployment = f"{model_name}-green"
        self.create_deployment(green_deployment, version)

        # Wait for green deployment to be ready
        self.wait_for_deployment_ready(green_deployment)

        # Run health checks
        self.run_health_checks(green_deployment)

        # Switch traffic to green
        self.switch_service_to_deployment(model_name, green_deployment)

        # Clean up old deployment (blue)
        blue_deployment = f"{model_name}-blue"
        self.cleanup_deployment(blue_deployment)

    def canary_deployment(self, model_name: str, version: str,
                         canary_percentage: int = 10):
        """Canary deployment strategy"""

        # Deploy canary version
        canary_deployment = f"{model_name}-canary"
        self.create_deployment(canary_deployment, version)

        # Configure traffic split
        self.configure_traffic_split(model_name, {
            f"{model_name}-stable": 100 - canary_percentage,
            canary_deployment: canary_percentage
        })

        # Monitor canary metrics
        canary_metrics = self.monitor_canary_metrics(canary_deployment)

        if canary_metrics['success_rate'] > 0.95:
            # Promote canary to stable
            self.promote_canary_to_stable(model_name, canary_deployment)
        else:
            # Rollback canary
            self.rollback_canary(model_name, canary_deployment)

    def rollback_to_version(self, version: str):
        """Rollback to previous model version"""
        logging.info(f"Rolling back to version {version}")

        # Implementation depends on deployment strategy
        # This is a simplified version
        self.deploy_model("model", version, "rolling")

    def get_last_successful_deployment(self):
        """Get last successful deployment from history"""
        for deployment in reversed(self.deployment_history):
            if deployment['status'] == 'success':
                return deployment
        return None
```

### 3. Security and Compliance

#### Model Security Framework
```python
import hashlib
import hmac
from cryptography.fernet import Fernet
import jwt

class ModelSecurityManager:
    def __init__(self, encryption_key: bytes, jwt_secret: str):
        self.cipher = Fernet(encryption_key)
        self.jwt_secret = jwt_secret

    def encrypt_model(self, model_data: bytes) -> bytes:
        """Encrypt model data for secure storage"""
        return self.cipher.encrypt(model_data)

    def decrypt_model(self, encrypted_data: bytes) -> bytes:
        """Decrypt model data"""
        return self.cipher.decrypt(encrypted_data)

    def generate_model_hash(self, model_data: bytes) -> str:
        """Generate hash for model integrity verification"""
        return hashlib.sha256(model_data).hexdigest()

    def verify_model_integrity(self, model_data: bytes, expected_hash: str) -> bool:
        """Verify model integrity using hash"""
        actual_hash = self.generate_model_hash(model_data)
        return hmac.compare_digest(actual_hash, expected_hash)

    def generate_api_token(self, user_id: str, permissions: list) -> str:
        """Generate JWT token for API access"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm='HS256')

    def verify_api_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")

class AuditLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.logger = logging.getLogger('audit')
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_prediction(self, user_id: str, model_version: str,
                      input_hash: str, prediction: Any):
        """Log prediction for audit trail"""
        self.logger.info(json.dumps({
            'event': 'prediction',
            'user_id': user_id,
            'model_version': model_version,
            'input_hash': input_hash,
            'prediction': prediction,
            'timestamp': datetime.utcnow().isoformat()
        }))

    def log_model_access(self, user_id: str, model_name: str, action: str):
        """Log model access for compliance"""
        self.logger.info(json.dumps({
            'event': 'model_access',
            'user_id': user_id,
            'model_name': model_name,
            'action': action,
            'timestamp': datetime.utcnow().isoformat()
        }))
```

---

## Conclusion

This comprehensive guide covers the essential aspects of MLOps and AI deployment strategies. Key takeaways include:

### Best Practices Summary

1. **Automation First**: Automate training, testing, and deployment pipelines
2. **Monitor Everything**: Track model performance, data quality, and system health
3. **Plan for Failure**: Implement circuit breakers, fallback models, and rollback strategies
4. **Security by Design**: Encrypt models, audit access, and validate inputs/outputs
5. **Gradual Rollouts**: Use blue-green, canary, or A/B testing for safe deployments
6. **Version Everything**: Models, data, code, and configurations
7. **Test Thoroughly**: Unit tests, integration tests, and performance validation

### Common Pitfalls to Avoid

- **Model Drift**: Implement continuous monitoring and retraining
- **Data Quality Issues**: Validate data at every stage
- **Security Vulnerabilities**: Encrypt sensitive data and implement proper authentication
- **Scalability Problems**: Design for horizontal scaling from the start
- **Vendor Lock-in**: Use open standards and containerization
- **Poor Observability**: Implement comprehensive logging and monitoring

### Next Steps

1. Start with a simple CI/CD pipeline
2. Implement basic monitoring and alerting
3. Add automated testing and validation
4. Scale up deployment strategies
5. Enhance security and compliance measures
6. Optimize for performance and cost

MLOps is an evolving field, and staying current with best practices and new tools is essential for success in production ML systems.
