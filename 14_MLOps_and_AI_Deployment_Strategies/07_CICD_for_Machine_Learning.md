# CI/CD for Machine Learning

**Navigation**: [← Module 6: Model Management and Versioning](06_Model_Management_and_Versioning.md) | [Main Index](README.md) | [Module 8: Edge AI and Federated Learning →](08_Edge_AI_and_Federated_Learning.md)

## Overview

CI/CD for Machine Learning extends traditional DevOps practices to handle the unique requirements of ML workflows, including data validation, model training, evaluation, and deployment.

## Complete MLOps Pipeline

### GitLab CI/CD Configuration

```yaml
# .gitlab-ci.yml - Complete MLOps Pipeline
stages:
  - validate
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
  PYTHON_VERSION: "3.9"

# Cache configuration
cache:
  paths:
    - .venv/
    - data/cache/
    - models/cache/

# Validation Stage
data-validation:
  stage: validate
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install -r requirements.txt
  script:
    - python scripts/validate_data_schema.py --data-path=data/raw
    - python scripts/check_data_quality.py --thresholds=config/data_quality.yaml
    - dvc repro data_validation.dvc
  artifacts:
    reports:
      junit: data_validation.xml
    paths:
      - data/validation_report.html
    expire_in: 1 week
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_MERGE_REQUEST_ID

unit-tests:
  stage: validate
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install -r requirements-test.txt
  script:
    - pytest tests/unit/ -v --cov=src --cov-report=xml --cov-report=html
    - python -m pylint src/ --output-format=pylint_junit.JUnitReporter
    - python -m mypy src/ --junit-xml=mypy.xml
  artifacts:
    reports:
      junit: pytest.xml
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - htmlcov/
      - pylint.xml
      - mypy.xml
    expire_in: 1 week

# Test Stage
integration-tests:
  stage: test
  image: python:${PYTHON_VERSION}
  services:
    - name: postgres:14
      alias: db
    - name: redis:7-alpine
      alias: redis
  variables:
    POSTGRES_DB: test_db
    POSTGRES_USER: test_user
    POSTGRES_PASSWORD: test_pass
    DATABASE_URL: "postgresql://test_user:test_pass@db:5432/test_db"
    REDIS_URL: "redis://redis:6379"
  before_script:
    - pip install -r requirements.txt
  script:
    - pytest tests/integration/ -v --tb=short
    - python scripts/test_api_endpoints.py --base-url=http://localhost:8000
  dependencies:
    - data-validation
  artifacts:
    reports:
      junit: integration_tests.xml
    expire_in: 1 week

model-validation:
  stage: test
  image: python:${PYTHON_VERSION}
  script:
    - python scripts/validate_model_inputs.py
    - python scripts/test_model_performance.py --min-accuracy=0.85
    - python scripts/check_model_fairness.py
  dependencies:
    - data-validation
  artifacts:
    paths:
      - model_validation_report.html
    expire_in: 1 week

# Train Stage
train-model:
  stage: train
  image: python:${PYTHON_VERSION}
  variables:
    GIT_STRATEGY: none
  before_script:
    - pip install -r requirements.txt
    - dvc pull
  script:
    - |
      python train.py \
        --data-path=data/processed \
        --config=config/hyperparameters.yaml \
        --experiment-name=${CI_COMMIT_SHORT_SHA} \
        --output-path=models/candidate
    - mlflow log_param "commit_sha" "${CI_COMMIT_SHA}"
    - mlflow log_param "job_id" "${CI_JOB_ID}"
  artifacts:
    paths:
      - models/candidate/
      - mlruns/
    expire_in: 1 week
  resource_group: training
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_MERGE_REQUEST_ID && $CI_MERGE_REQUEST_TARGET_BRANCH_NAME == "main"
  tags:
    - gpu
    - high-mem

hyperparameter-tuning:
  stage: train
  image: python:${PYTHON_VERSION}
  before_script:
    - pip install -r requirements.txt
    - dvc pull
  script:
    - python scripts/hyperparameter_optimization.py \
        --n-trials=50 \
        --study-name="${CI_COMMIT_SHORT_SHA}_hpo" \
        --output-path=models/hpo_results
  artifacts:
    paths:
      - models/hpo_results/
      - optuna_study.db
    expire_in: 1 week
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual
  tags:
    - gpu
    - high-mem

# Evaluate Stage
evaluate-model:
  stage: evaluate
  image: python:${PYTHON_VERSION}
  script:
    - |
      python evaluate.py \
        --model-path=models/candidate \
        --test-data=data/test \
        --thresholds=config/thresholds.yaml \
        --output-path=evaluation
    - python scripts/generate_evaluation_report.py
  dependencies:
    - train-model
  artifacts:
    paths:
      - evaluation/
    expire_in: 1 week

compare-models:
  stage: evaluate
  image: python:${PYTHON_VERSION}
  script:
    - |
      python scripts/compare_with_production.py \
        --candidate-model=models/candidate \
        --production-model="${PRODUCTION_MODEL_ID}" \
        --comparison-metrics=config/comparison_metrics.yaml \
        --output-path=model_comparison
    - python scripts/generate_comparison_report.py
  dependencies:
    - evaluate-model
  artifacts:
    paths:
      - model_comparison/
    expire_in: 1 week

# Register Stage
register-model:
  stage: register
  image: python:${PYTHON_VERSION}
  script:
    - |
      python scripts/register_model.py \
        --model-path=models/candidate \
        --model-name="${CI_PROJECT_NAME}" \
        --version="${CI_COMMIT_SHA}" \
        --metrics-path=evaluation/metrics.json \
        --tags="commit:${CI_COMMIT_SHA},job:${CI_JOB_ID}"
  dependencies:
    - evaluate-model
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual

# Deploy Stage
deploy-staging:
  stage: deploy
  image: google/cloud-sdk:latest
  environment:
    name: staging
    url: https://staging-api.your-domain.com
  before_script:
    - gcloud auth activate-service-account --key-file=$GCLOUD_SERVICE_KEY
    - gcloud config set project $GCLOUD_PROJECT_ID
    - gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE
  script:
    - |
      # Build and push Docker image
      docker build -t ${MODEL_REGISTRY}/${CI_PROJECT_NAME}:${CI_COMMIT_SHA} .
      docker push ${MODEL_REGISTRY}/${CI_PROJECT_NAME}:${CI_COMMIT_SHA}

      # Deploy to staging
      kubectl set image deployment/${CI_PROJECT_NAME} \
        ${CI_PROJECT_NAME}=${MODEL_REGISTRY}/${CI_PROJECT_NAME}:${CI_COMMIT_SHA} \
        -n staging

      # Wait for rollout
      kubectl rollout status deployment/${CI_PROJECT_NAME} -n staging

      # Run smoke tests
      python scripts/smoke_tests.py --env=staging
  dependencies:
    - register-model
  rules:
    - if: $CI_COMMIT_BRANCH == "main"

canary-deploy:
  stage: deploy
  image: google/cloud-sdk:latest
  environment:
    name: production
    url: https://api.your-domain.com
  before_script:
    - gcloud auth activate-service-account --key-file=$GCLOUD_SERVICE_KEY
    - gcloud config set project $GCLOUD_PROJECT_ID
    - gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE
  script:
    - |
      # Deploy canary version (10% traffic)
      python scripts/canary_deploy.py \
        --model-id=${CI_COMMIT_SHA} \
        --canary-percentage=10 \
        --monitoring-duration=1h

      # Monitor canary metrics
      python scripts/monitor_canary.py \
        --model-id=${CI_COMMIT_SHA} \
        --duration=1h \
        --thresholds=config/canary_thresholds.yaml
  dependencies:
    - deploy-staging
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual

deploy-production:
  stage: deploy
  image: google/cloud-sdk:latest
  environment:
    name: production
    url: https://api.your-domain.com
  before_script:
    - gcloud auth activate-service-account --key-file=$GCLOUD_SERVICE_KEY
    - gcloud config set project $GCLOUD_PROJECT_ID
    - gcloud container clusters get-credentials $GKE_CLUSTER --zone $GKE_ZONE
  script:
    - |
      # Full production deployment
      python scripts/blue_green_deploy.py \
        --model-id=${CI_COMMIT_SHA} \
        --strategy=blue-green

      # Run comprehensive tests
      python scripts/production_tests.py --env=production

      # Update monitoring dashboards
      python scripts/update_dashboards.py
  dependencies:
    - canary-deploy
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
      when: manual
      allow_failure: false

# Monitor Stage
monitor-deployment:
  stage: monitor
  image: python:${PYTHON_VERSION}
  script:
    - |
      # Monitor model performance
      python scripts/monitor_model_performance.py \
        --model-id=${CI_COMMIT_SHA} \
        --duration=24h \
        --metrics=config/monitoring_metrics.yaml

      # Check for drift
      python scripts/check_drift.py \
        --model-id=${CI_COMMIT_SHA} \
        --reference-data=data/reference

      # Generate monitoring report
      python scripts/generate_monitoring_report.py
  dependencies:
    - deploy-production
  rules:
    - if: $CI_COMMIT_BRANCH == "main"
  artifacts:
    paths:
      - monitoring_reports/
    expire_in: 1 week

# Security Scans
security-scan:
  stage: validate
  image: python:${PYTHON_VERSION}
  script:
    - pip install bandit safety
    - bandit -r src/ -f json -o bandit-report.json
    - safety check --json --output safety-report.json
  artifacts:
    reports:
      sast: bandit-report.json
    paths:
      - bandit-report.json
      - safety-report.json
    expire_in: 1 week
  allow_failure: true

container-security:
  stage: validate
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
        -v $PWD:/root aquasec/trivy:latest image --format json \
        --output trivy-report.json ${MODEL_REGISTRY}/${CI_PROJECT_NAME}:${CI_COMMIT_SHA}
  artifacts:
    reports:
      container_scanning: trivy-report.json
    paths:
      - trivy-report.json
    expire_in: 1 week
  allow_failure: true
```

### GitHub Actions Advanced Workflow

```yaml
# .github/workflows/advanced-ml-pipeline.yml
name: Advanced ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Daily retraining
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options: [staging, production]
      force_retrain:
        description: 'Force model retraining'
        required: false
        type: boolean
        default: false

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Data Validation Job
  validate-data:
    runs-on: ubuntu-latest
    outputs:
      data-quality-score: ${{ steps.data-validation.outputs.quality-score }}
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install great-expectations

    - name: Validate data quality
      id: data-validation
      run: |
        python scripts/validate_data.py \
          --data-path=data/raw \
          --config=config/validation_config.yaml
        echo "::set-output name=quality-score::$(python -c "import json; print(json.load(open('data_quality_report.json'))['overall_score'])")"

    - name: Upload validation report
      uses: actions/upload-artifact@v3
      with:
        name: data-validation-report
        path: data_quality_report.json

  # Model Training Job
  train-model:
    runs-on: ubuntu-latest
    needs: validate-data
    if: needs.validate-data.outputs.data-quality-score >= 0.9
    outputs:
      model-version: ${{ steps.train.outputs.model-version }}
      metrics: ${{ steps.train.outputs.metrics }}
    strategy:
      matrix:
        model-type: [random-forest, xgboost, neural-network]
    steps:
    - uses: actions/checkout@v4

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
        aws s3 sync s3://ml-data/${{ github.repository }}/training data/training
        aws s3 sync s3://ml-data/${{ github.repository }}/validation data/validation

    - name: Train model
      id: train
      run: |
        python train.py \
          --model-type=${{ matrix.model-type }} \
          --data-path=data/training \
          --validation-path=data/validation \
          --output-path=models/${{ matrix.model-type }} \
          --experiment-name="${{ github.sha }}_${{ matrix.model-type }}"

        echo "::set-output name=model-version::$(python -c "import json; print(json.load(open('models/${{ matrix.model-type }}/metadata.json'))['version'])")"
        echo "::set-output name=metrics::$(cat models/${{ matrix.model-type }}/metrics.json)"

    - name: Upload model artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-${{ matrix.model-type }}
        path: models/${{ matrix.model-type }}/

    - name: Upload to S3
      run: |
        aws s3 sync models/${{ matrix.model-type }}/ s3://ml-models/${{ github.repository }}/${{ github.sha }}_${{ matrix.model-type }}/

  # Model Evaluation Job
  evaluate-models:
    runs-on: ubuntu-latest
    needs: train-model
    outputs:
      best-model: ${{ steps.evaluate.outputs.best-model }}
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Download all models
      uses: actions/download-artifact@v3

    - name: Evaluate models
      id: evaluate
      run: |
        python scripts/evaluate_all_models.py \
          --models-dir=models/ \
          --test-data=data/test \
          --output-path=evaluation_results

        echo "::set-output name=best-model::$(python -c "import json; print(json.load(open('evaluation_results/best_model.json'))['model_type'])")"

    - name: Upload evaluation results
      uses: actions/upload-artifact@v3
      with:
        name: evaluation-results
        path: evaluation_results/

  # Build and Test Container
  build-container:
    runs-on: ubuntu-latest
    needs: evaluate-models
    outputs:
      image-digest: ${{ steps.build.outputs.digest }}
    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      id: build
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: |
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
        labels: |
          org.opencontainers.image.source=${{ github.server_url }}/${{ github.repository }}
          org.opencontainers.image.revision=${{ github.sha }}
          org.opencontainers.image.created=${{ steps.prep.outputs.created }}
          model.version=${{ needs.train-model.outputs.model-version }}
          model.type=${{ needs.evaluate-models.outputs.best-model }}

    - name: Run container tests
      run: |
        docker run --rm ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} python -m pytest tests/container/

    - name: Generate SBOM
      run: |
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          -v $PWD:/output anchore/syft:latest \
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
          -o cyclonedx-json=/output/sbom.json

    - name: Upload SBOM
      uses: actions/upload-artifact@v3
      with:
        name: sbom
        path: sbom.json

  # Security Scanning
  security-scan:
    runs-on: ubuntu-latest
    needs: build-container
    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # Deploy to Staging
  deploy-staging:
    runs-on: ubuntu-latest
    needs: [build-container, security-scan]
    if: github.ref == 'refs/heads/main'
    environment:
      name: staging
      url: https://staging-${{ github.repository }}.your-domain.com
    steps:
    - uses: actions/checkout@v4

    - name: Set up Kubernetes
      uses: azure/setup-kubectl@v3
      with:
        version: '1.25.0'

    - name: Configure kubeconfig
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > $HOME/.kube/config

    - name: Deploy to staging
      run: |
        envsubst < k8s/deployment.yaml | kubectl apply -f -
        kubectl rollout status deployment/${{ github.repository }} -n staging

    - name: Run integration tests
      run: |
        python scripts/integration_tests.py \
          --base-url=https://staging-${{ github.repository }}.your-domain.com \
          --timeout=300

  # Canary Deployment
  canary-deploy:
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://api.your-domain.com
    steps:
    - uses: actions/checkout@v4

    - name: Deploy canary (10% traffic)
      run: |
        python scripts/canary_deploy.py \
          --image="${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}" \
          --canary-percentage=10 \
          --duration=3600

    - name: Monitor canary performance
      run: |
        python scripts/monitor_canary.py \
          --duration=3600 \
          --thresholds=config/canary_thresholds.yaml

  # Production Deployment
  deploy-production:
    runs-on: ubuntu-latest
    needs: canary-deploy
    if: github.ref == 'refs/heads/main' && github.event_name == 'workflow_dispatch'
    environment:
      name: production
      url: https://api.your-domain.com
    steps:
    - uses: actions/checkout@v4

    - name: Deploy to production
      run: |
        python scripts/blue_green_deploy.py \
          --image="${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}" \
          --strategy=blue-green

    - name: Update monitoring dashboards
      run: |
        python scripts/update_grafana_dashboards.py \
          --model-version=${{ needs.train-model.outputs.model-version }}

  # Continuous Monitoring
  monitor-model:
    runs-on: ubuntu-latest
    needs: deploy-production
    if: always() && needs.deploy-production.result == 'success'
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Monitor model performance
      run: |
        python scripts/continuous_monitoring.py \
          --model-version=${{ needs.train-model.outputs.model-version }} \
          --duration=24h \
          --alert-thresholds=config/alert_thresholds.yaml

    - name: Check for model drift
      run: |
        python scripts/drift_detection.py \
          --model-version=${{ needs.train-model.outputs.model-version }}
```

## Model Testing Framework

### Comprehensive Test Suite

```python
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import tempfile
import os
import logging
from typing import Dict, Any, List, Optional

class ModelTestFramework:
    """
    Comprehensive testing framework for ML models.
    """

    def __init__(self, model_path: str, test_data_path: str, config_path: str = None):
        self.model = self.load_model(model_path)
        self.test_data = self.load_test_data(test_data_path)
        self.config = self.load_config(config_path) if config_path else {}
        self.test_results = {}
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Setup test logger"""
        logger = logging.getLogger('model_testing')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def load_model(self, model_path: str):
        """Load model with error handling"""
        try:
            return joblib.load(model_path)
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def load_test_data(self, test_data_path: str) -> pd.DataFrame:
        """Load test data"""
        try:
            return pd.read_csv(test_data_path)
        except Exception as e:
            self.logger.error(f"Failed to load test data: {e}")
            raise

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load test configuration"""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load config: {e}")
            return {}

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all model tests"""
        test_functions = [
            self.test_model_performance,
            self.test_prediction_consistency,
            self.test_model_robustness,
            self.test_feature_importance,
            self.test_data_distribution,
            self.test_model_fairness,
            self.test_inference_performance,
            self.test_model_interpretability,
            self.test_edge_cases,
            self.test_model_reproducibility
        ]

        results = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.get_model_info(),
            'test_summary': {'total': 0, 'passed': 0, 'failed': 0},
            'detailed_results': {}
        }

        for test_func in test_functions:
            test_name = test_func.__name__
            try:
                result = test_func()
                results['detailed_results'][test_name] = result

                if result.get('passed', False):
                    results['test_summary']['passed'] += 1
                else:
                    results['test_summary']['failed'] += 1

                results['test_summary']['total'] += 1

            except Exception as e:
                self.logger.error(f"Test {test_name} failed: {e}")
                results['detailed_results'][test_name] = {
                    'passed': False,
                    'error': str(e)
                }
                results['test_summary']['failed'] += 1
                results['test_summary']['total'] += 1

        # Calculate overall score
        results['overall_score'] = results['test_summary']['passed'] / results['test_summary']['total']

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            'model_type': type(self.model).__name__,
            'model_class': str(type(self.model))
        }

        # Add model-specific info
        if hasattr(self.model, 'n_features_in_'):
            info['n_features'] = self.model.n_features_in_
        if hasattr(self.model, 'classes_'):
            info['n_classes'] = len(self.model.classes_)
        if hasattr(self.model, 'feature_importances_'):
            info['has_feature_importance'] = True

        return info

    def test_model_performance(self) -> Dict[str, Any]:
        """Test model performance metrics"""
        X_test = self.test_data.drop('target', axis=1)
        y_test = self.test_data['target']

        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')

        # Get thresholds from config
        thresholds = self.config.get('performance_thresholds', {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.80,
            'f1': 0.80
        })

        results = {
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'thresholds': thresholds,
            'passed': True
        }

        # Check thresholds
        for metric, value in results['metrics'].items():
            if value < thresholds.get(metric, 0):
                results['passed'] = False
                results[f'{metric}_failure'] = f"{metric} {value:.3f} below threshold {thresholds[metric]}"

        return results

    def test_prediction_consistency(self) -> Dict[str, Any]:
        """Test prediction consistency across multiple runs"""
        X_test = self.test_data.drop('target', axis=1).head(100)  # Test on subset

        predictions = []
        for _ in range(5):
            pred = self.model.predict(X_test)
            predictions.append(pred)

        # Check if all predictions are identical
        all_identical = all(np.array_equal(predictions[0], pred) for pred in predictions[1:])

        # Calculate variance if applicable
        if hasattr(self.model, 'predict_proba'):
            probabilities = []
            for _ in range(5):
                prob = self.model.predict_proba(X_test)
                probabilities.append(prob)

            prob_variance = np.var(probabilities, axis=0).mean()
            variance_threshold = self.config.get('probability_variance_threshold', 0.001)

            return {
                'consistent_predictions': all_identical,
                'probability_variance': prob_variance,
                'variance_threshold': variance_threshold,
                'passed': all_identical and prob_variance < variance_threshold
            }

        return {
            'consistent_predictions': all_identical,
            'passed': all_identical
        }

    def test_model_robustness(self) -> Dict[str, Any]:
        """Test model robustness to input perturbations"""
        X_test = self.test_data.drop('target', axis=1).head(100)
        y_test = self.test_data['target'].head(100)

        # Original predictions
        original_pred = self.model.predict(X_test)
        original_accuracy = accuracy_score(y_test, original_pred)

        # Test with noise
        noise_levels = [0.001, 0.01, 0.1]
        robustness_results = {}

        for noise_level in noise_levels:
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise

            noisy_pred = self.model.predict(X_noisy)
            noisy_accuracy = accuracy_score(y_test, noisy_pred)

            accuracy_drop = original_accuracy - noisy_accuracy

            robustness_results[f'noise_{noise_level}'] = {
                'original_accuracy': original_accuracy,
                'noisy_accuracy': noisy_accuracy,
                'accuracy_drop': accuracy_drop,
                'acceptable': accuracy_drop < self.config.get('max_accuracy_drop', 0.05)
            }

        # Overall robustness score
        acceptable_drops = sum(1 for result in robustness_results.values() if result['acceptable'])
        robustness_score = acceptable_drops / len(robustness_results)

        return {
            'robustness_tests': robustness_results,
            'robustness_score': robustness_score,
            'passed': robustness_score >= self.config.get('robustness_threshold', 0.8)
        }

    def test_feature_importance(self) -> Dict[str, Any]:
        """Test feature importance stability and reasonableness"""
        if not hasattr(self.model, 'feature_importances_'):
            return {
                'has_feature_importance': False,
                'passed': True,
                'message': 'Model does not support feature importance'
            }

        X_test = self.test_data.drop('target', axis=1)
        feature_names = X_test.columns
        importances = self.model.feature_importances_

        # Test importance stability with bootstrapping
        bootstrap_importances = []
        n_bootstrap = self.config.get('n_bootstrap', 10)

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(X_test), len(X_test), replace=True)
            X_bootstrap = X_test.iloc[indices]

            if hasattr(self.model, 'feature_importances_'):
                # For models that have direct feature importance
                bootstrap_importances.append(self.model.feature_importances_)
            else:
                # Use permutation importance for other models
                from sklearn.inspection import permutation_importance
                result = permutation_importance(self.model, X_bootstrap, self.test_data['target'].iloc[indices])
                bootstrap_importances.append(result.importances_mean)

        bootstrap_importances = np.array(bootstrap_importances)
        importance_std = np.std(bootstrap_importances, axis=0)

        # Check for reasonable importance distribution
        max_importance = np.max(importances)
        min_importance = np.min(importances)
        importance_range = max_importance - min_importance

        # Top features
        top_features = feature_names[np.argsort(importances)[-5:]]

        return {
            'has_feature_importance': True,
            'feature_importances': dict(zip(feature_names, importances)),
            'importance_stability': {
                'mean_std': np.mean(importance_std),
                'max_std': np.max(importance_std)
            },
            'distribution': {
                'max_importance': max_importance,
                'min_importance': min_importance,
                'range': importance_range
            },
            'top_features': list(top_features),
            'passed': (
                np.mean(importance_std) < self.config.get('importance_stability_threshold', 0.05) and
                max_importance < self.config.get('max_single_importance', 0.8)
            )
        }

    def test_data_distribution(self) -> Dict[str, Any]:
        """Test if test data distribution matches training expectations"""
        X_test = self.test_data.drop('target', axis=1)

        # Check basic statistics
        results = {
            'data_shape': X_test.shape,
            'missing_values': X_test.isnull().sum().to_dict(),
            'statistical_summary': {}
        }

        # Compare with expected distribution if available
        expected_stats = self.config.get('expected_data_stats', {})

        if expected_stats:
            for column, expected in expected_stats.items():
                if column in X_test.columns:
                    actual = {
                        'mean': X_test[column].mean(),
                        'std': X_test[column].std(),
                        'min': X_test[column].min(),
                        'max': X_test[column].max()
                    }

                    # Check deviations
                    mean_deviation = abs(actual['mean'] - expected.get('mean', actual['mean']))
                    std_deviation = abs(actual['std'] - expected.get('std', actual['std']))

                    results['statistical_summary'][column] = {
                        'expected': expected,
                        'actual': actual,
                        'deviations': {
                            'mean': mean_deviation,
                            'std': std_deviation
                        }
                    }

        # Overall data quality check
        total_missing = sum(results['missing_values'].values())
        data_quality_score = 1 - (total_missing / X_test.size)

        results['data_quality_score'] = data_quality_score
        results['passed'] = data_quality_score > self.config.get('data_quality_threshold', 0.95)

        return results

    def test_model_fairness(self) -> Dict[str, Any]:
        """Test model fairness across different demographic groups"""
        if 'protected_attribute' not in self.config:
            return {
                'has_fairness_config': False,
                'passed': True,
                'message': 'No fairness configuration provided'
            }

        protected_attr = self.config['protected_attribute']

        if protected_attr not in self.test_data.columns:
            return {
                'has_protected_attribute': False,
                'passed': False,
                'message': f'Protected attribute {protected_attr} not found in data'
            }

        X_test = self.test_data.drop('target', axis=1)
        y_test = self.test_data['target']
        protected_values = self.test_data[protected_attr]

        predictions = self.model.predict(X_test)

        # Calculate metrics for each group
        unique_groups = protected_values.unique()
        group_metrics = {}

        for group in unique_groups:
            group_mask = protected_values == group
            group_y_true = y_test[group_mask]
            group_y_pred = predictions[group_mask]

            if len(group_y_true) > 0:
                group_metrics[group] = {
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'precision': precision_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(group_y_true, group_y_pred, average='weighted', zero_division=0),
                    'sample_size': len(group_y_true)
                }

        # Calculate fairness metrics
        if len(group_metrics) >= 2:
            accuracies = [metrics['accuracy'] for metrics in group_metrics.values()]
            accuracy_disparity = max(accuracies) - min(accuracies)

            # Statistical parity difference
            positive_rates = {}
            for group, metrics in group_metrics.items():
                group_mask = protected_values == group
                positive_rates[group] = predictions[group_mask].mean()

            parity_difference = max(positive_rates.values()) - min(positive_rates.values())

            fairness_threshold = self.config.get('fairness_threshold', 0.1)

            return {
                'has_fairness_config': True,
                'group_metrics': group_metrics,
                'fairness_metrics': {
                    'accuracy_disparity': accuracy_disparity,
                    'statistical_parity_difference': parity_difference
                },
                'passed': (
                    accuracy_disparity < fairness_threshold and
                    parity_difference < fairness_threshold
                )
            }

        return {
            'has_fairness_config': True,
            'insufficient_groups': len(group_metrics),
            'passed': False
        }

    def test_inference_performance(self) -> Dict[str, Any]:
        """Test inference performance (latency and throughput)"""
        X_test = self.test_data.drop('target', axis=1).head(100)  # Test on subset

        # Measure prediction latency
        latencies = []
        n_runs = self.config.get('performance_test_runs', 100)

        for _ in range(n_runs):
            start_time = datetime.now()
            self.model.predict(X_test.head(1))  # Single prediction
            end_time = datetime.now()
            latency = (end_time - start_time).total_seconds()
            latencies.append(latency)

        latencies = np.array(latencies)

        # Measure throughput
        start_time = datetime.now()
        batch_size = self.config.get('batch_size', 32)
        for _ in range(10):
            self.model.predict(X_test.head(batch_size))
        end_time = datetime.now()

        total_time = (end_time - start_time).total_seconds()
        throughput = (10 * batch_size) / total_time

        performance_thresholds = self.config.get('performance_thresholds', {
            'max_latency_p95': 0.1,  # 100ms
            'min_throughput': 100     # 100 predictions/second
        })

        return {
            'latency_metrics': {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'p95': np.percentile(latencies, 95),
                'p99': np.percentile(latencies, 99)
            },
            'throughput': throughput,
            'thresholds': performance_thresholds,
            'passed': (
                np.percentile(latencies, 95) < performance_thresholds['max_latency_p95'] and
                throughput > performance_thresholds['min_throughput']
            )
        }

    def test_model_interpretability(self) -> Dict[str, Any]:
        """Test model interpretability if applicable"""
        X_test = self.test_data.drop('target', axis=1).head(10)

        interpretability_results = {
            'methods_tested': [],
            'passed': True
        }

        # Test SHAP values if available
        try:
            import shap

            if hasattr(self.model, 'predict_proba'):
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test)

                interpretability_results['methods_tested'].append('shap')
                interpretability_results['shap_summary'] = {
                    'mean_shap_absolute': np.mean(np.abs(shap_values)) if isinstance(shap_values, np.ndarray) else None
                }
        except ImportError:
            interpretability_results['methods_tested'].append('shap_not_available')

        # Test LIME if available
        try:
            import lime.lime_tabular

            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_test.values,
                feature_names=X_test.columns,
                class_names=self.model.classes_ if hasattr(self.model, 'classes_') else ['negative', 'positive'],
                verbose=False
            )

            # Test explanation for one instance
            exp = explainer.explain_instance(
                X_test.iloc[0].values,
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict
            )

            interpretability_results['methods_tested'].append('lime')
            interpretability_results['lime_summary'] = {
                'explanation_generated': True,
                'top_features': len(exp.as_list())
            }
        except ImportError:
            interpretability_results['methods_tested'].append('lime_not_available')

        return interpretability_results

    def test_edge_cases(self) -> Dict[str, Any]:
        """Test model behavior on edge cases"""
        X_test = self.test_data.drop('target', axis=1)

        edge_case_results = {
            'tests_passed': 0,
            'total_tests': 0,
            'details': {}
        }

        # Test with empty features
        try:
            empty_features = pd.DataFrame(0, index=[0], columns=X_test.columns)
            pred = self.model.predict(empty_features)
            edge_case_results['details']['empty_features'] = {
                'handled': True,
                'prediction': pred[0] if len(pred) > 0 else None
            }
            edge_case_results['tests_passed'] += 1
        except Exception as e:
            edge_case_results['details']['empty_features'] = {
                'handled': False,
                'error': str(e)
            }
        edge_case_results['total_tests'] += 1

        # Test with extreme values
        try:
            extreme_features = X_test.copy()
            for col in extreme_features.columns:
                if extreme_features[col].dtype in ['int64', 'float64']:
                    extreme_features[col] = extreme_features[col] * 1000

            pred = self.model.predict(extreme_features.head(1))
            edge_case_results['details']['extreme_values'] = {
                'handled': True,
                'prediction': pred[0] if len(pred) > 0 else None
            }
            edge_case_results['tests_passed'] += 1
        except Exception as e:
            edge_case_results['details']['extreme_values'] = {
                'handled': False,
                'error': str(e)
            }
        edge_case_results['total_tests'] += 1

        # Test with missing values
        try:
            missing_features = X_test.copy()
            missing_features.iloc[0, 0] = np.nan

            pred = self.model.predict(missing_features.head(1))
            edge_case_results['details']['missing_values'] = {
                'handled': True,
                'prediction': pred[0] if len(pred) > 0 else None
            }
            edge_case_results['tests_passed'] += 1
        except Exception as e:
            edge_case_results['details']['missing_values'] = {
                'handled': False,
                'error': str(e)
            }
        edge_case_results['total_tests'] += 1

        edge_case_results['passed'] = (
            edge_case_results['tests_passed'] / edge_case_results['total_tests'] >=
            self.config.get('edge_case_pass_threshold', 0.5)
        )

        return edge_case_results

    def test_model_reproducibility(self) -> Dict[str, Any]:
        """Test model reproducibility with different random seeds"""
        import random

        X_test = self.test_data.drop('target', axis=1).head(100)

        # Test with different random seeds
        seeds = [42, 123, 456, 789, 999]
        seed_predictions = []

        for seed in seeds:
            # Set random seeds
            np.random.seed(seed)
            random.seed(seed)

            # Make predictions
            pred = self.model.predict(X_test)
            seed_predictions.append(pred)

        # Check consistency
        all_identical = all(np.array_equal(seed_predictions[0], pred) for pred in seed_predictions[1:])

        return {
            'reproducibility_test': {
                'seeds_tested': seeds,
                'all_predictions_identical': all_identical
            },
            'passed': all_identical
        }

# Pytest fixtures and tests
@pytest.fixture
def model_test_framework():
    """Pytest fixture for model testing"""
    return ModelTestFramework(
        model_path="models/churn_model.pkl",
        test_data_path="data/test/test.csv",
        config_path="config/test_config.yaml"
    )

def test_model_performance(model_test_framework):
    """Test model performance metrics"""
    result = model_test_framework.test_model_performance()
    assert result['passed'], f"Model performance test failed: {result}"

def test_model_robustness(model_test_framework):
    """Test model robustness"""
    result = model_test_framework.test_model_robustness()
    assert result['passed'], f"Model robustness test failed: {result}"

def test_model_fairness(model_test_framework):
    """Test model fairness"""
    result = model_test_framework.test_model_fairness()
    assert result['passed'], f"Model fairness test failed: {result}"

def test_edge_cases(model_test_framework):
    """Test edge cases"""
    result = model_test_framework.test_edge_cases()
    assert result['passed'], f"Edge case test failed: {result}"

# Integration test
class ModelDeploymentTest:
    """Integration tests for model deployment"""

    def __init__(self, api_url: str, model_version: str):
        self.api_url = api_url
        self.model_version = model_version
        self.test_results = {}

    def run_deployment_tests(self) -> Dict[str, Any]:
        """Run all deployment tests"""
        tests = [
            self.test_api_health,
            self.test_prediction_endpoint,
            self.test_model_version,
            self.test_rate_limiting,
            self.test_authentication,
            self.test_monitoring_integration
        ]

        results = {
            'timestamp': datetime.now().isoformat(),
            'api_url': self.api_url,
            'model_version': self.model_version,
            'tests': {}
        }

        for test in tests:
            test_name = test.__name__
            try:
                result = test()
                results['tests'][test_name] = result
            except Exception as e:
                results['tests'][test_name] = {
                    'passed': False,
                    'error': str(e)
                }

        return results

    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint"""
        import requests

        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return {
                'passed': response.status_code == 200,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds()
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }

    def test_prediction_endpoint(self) -> Dict[str, Any]:
        """Test prediction endpoint"""
        import requests
        import json

        # Test data
        test_input = {
            "features": [0.5, 0.3, 0.8, 0.1, 0.9]
        }

        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=test_input,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                return {
                    'passed': True,
                    'status_code': response.status_code,
                    'response_time': response.elapsed.total_seconds(),
                    'has_prediction': 'prediction' in result,
                    'has_probability': 'probability' in result
                }
            else:
                return {
                    'passed': False,
                    'status_code': response.status_code,
                    'error': response.text
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }

    def test_model_version(self) -> Dict[str, Any]:
        """Test model version endpoint"""
        import requests

        try:
            response = requests.get(f"{self.api_url}/model-info", timeout=5)

            if response.status_code == 200:
                info = response.json()
                return {
                    'passed': info.get('version') == self.model_version,
                    'model_version': info.get('version'),
                    'expected_version': self.model_version
                }
            else:
                return {
                    'passed': False,
                    'status_code': response.status_code
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }

    def test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting"""
        import requests
        import time

        requests_sent = 0
        rate_limited = 0

        start_time = time.time()
        for i in range(110):  # Send more than rate limit
            try:
                response = requests.post(
                    f"{self.api_url}/predict",
                    json={"features": [0.5, 0.3, 0.8, 0.1, 0.9]},
                    timeout=5
                )
                requests_sent += 1

                if response.status_code == 429:
                    rate_limited += 1

            except Exception:
                pass

        elapsed_time = time.time() - start_time

        return {
            'passed': rate_limited > 0,
            'requests_sent': requests_sent,
            'rate_limited_count': rate_limited,
            'elapsed_time': elapsed_time
        }

    def test_authentication(self) -> Dict[str, Any]:
        """Test authentication requirements"""
        import requests

        # Test without authentication
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={"features": [0.5, 0.3, 0.8, 0.1, 0.9]},
                timeout=5
            )

            # Should be unauthorized (401) or forbidden (403)
            auth_required = response.status_code in [401, 403]

            return {
                'passed': auth_required,
                'status_code': response.status_code,
                'authentication_required': auth_required
            }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }

    def test_monitoring_integration(self) -> Dict[str, Any]:
        """Test monitoring and metrics integration"""
        import requests

        # Test metrics endpoint
        try:
            response = requests.get(f"{self.api_url}/metrics", timeout=5)

            if response.status_code == 200:
                metrics = response.text
                return {
                    'passed': True,
                    'has_predictions_total': 'ml_predictions_total' in metrics,
                    'has_prediction_latency': 'ml_prediction_latency' in metrics
                }
            else:
                return {
                    'passed': False,
                    'status_code': response.status_code
                }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }

# Usage examples
if __name__ == "__main__":
    # Model testing
    test_framework = ModelTestFramework(
        model_path="models/churn_model.pkl",
        test_data_path="data/test/test.csv",
        config_path="config/test_config.yaml"
    )

    test_results = test_framework.run_all_tests()
    print("Test Results:", json.dumps(test_results, indent=2))

    # Deployment testing
    deployment_test = ModelDeploymentTest(
        api_url="https://staging-api.example.com",
        model_version="v1.0.0"
    )

    deployment_results = deployment_test.run_deployment_tests()
    print("Deployment Test Results:", json.dumps(deployment_results, indent=2))
```

## Quick Reference

### CI/CD Pipeline Components

```yaml
# Essential pipeline stages
stages:
  - validate    # Data validation, security scans
  - test        # Unit tests, integration tests
  - train       # Model training, hyperparameter tuning
  - evaluate    # Model evaluation, comparison
  - register    # Model registration, versioning
  - deploy      # Staging, canary, production
  - monitor     # Performance monitoring, drift detection
```

### Key Testing Categories

1. **Performance Testing**
   - Accuracy, precision, recall, F1-score
   - ROC-AUC, confusion matrix
   - Business metrics (ROI, cost savings)

2. **Robustness Testing**
   - Noise tolerance
   - Input perturbations
   - Edge case handling

3. **Fairness Testing**
   - Demographic parity
   - Equal opportunity
   - Disparate impact

4. **Performance Testing**
   - Latency (P50, P95, P99)
   - Throughput (requests/second)
   - Memory usage

5. **Security Testing**
   - Input validation
   - Access control
   - Data encryption

### Best Practices

1. **Pipeline Design**
   - Modular stages with clear dependencies
   - Artifact management for reproducibility
   - Parallel execution where possible
   - Automated rollback on failure

2. **Testing Strategy**
   - Comprehensive test coverage
   - Automated testing in CI/CD
   - Performance benchmarks
   - Security and compliance checks

3. **Deployment Strategy**
   - Gradual rollouts (canary, blue-green)
   - Automated health checks
   - Monitoring and alerting
   - Rollback procedures

4. **Monitoring Integration**
   - Model performance metrics
   - Data drift detection
   - System health monitoring
   - Business impact tracking

## Summary

This module provides comprehensive CI/CD pipelines for machine learning workflows, including:

- **Complete pipeline configurations** for GitLab CI/CD and GitHub Actions
- **Advanced testing frameworks** for model validation
- **Deployment strategies** with safety mechanisms
- **Security and compliance** integration
- **Performance monitoring** and alerting

The implementation ensures that ML models are thoroughly tested, safely deployed, and continuously monitored in production environments.

**Next**: [Module 8: Edge AI and Federated Learning](08_Edge_AI_and_Federated_Learning.md)