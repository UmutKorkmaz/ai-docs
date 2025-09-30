# Module 1: MLOps Fundamentals

## Navigation
- **← Previous**: [README.md](README.md)
- **→ Next**: [02_LLMops_and_Generative_AI_Operations.md](02_LLMops_and_Generative_AI_Operations.md)
- **↑ Up**: [README.md](README.md)

## What is MLOps?

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

## MLOps Architecture Components

### 1. Data Management Layer
```python
class DataManagementSystem:
    def __init__(self):
        self.data_registry = DataRegistry()
        self.version_control = DataVersionControl()
        self.quality_framework = DataQualityFramework()
        self.lineage_tracker = DataLineageTracker()

    def register_dataset(self, dataset_config: dict) -> str:
        """Register dataset with versioning and quality checks"""

        # Validate dataset quality
        quality_report = self.quality_framework.validate_dataset(
            dataset_config['path']
        )

        if not quality_report['passed']:
            raise DataQualityError(f"Dataset quality issues: {quality_report['issues']}")

        # Create version
        version = self.version_control.create_version(dataset_config)

        # Track lineage
        lineage = self.lineage_tracker.track_dataset(
            dataset_config['source'],
            version,
            dataset_config['transformations']
        )

        # Register in data registry
        dataset_id = self.data_registry.register({
            'config': dataset_config,
            'version': version,
            'quality_report': quality_report,
            'lineage': lineage
        })

        return dataset_id
```

### 2. Model Development Lifecycle
```python
class ModelDevelopmentLifecycle:
    def __init__(self):
        self.experiment_tracker = ExperimentTracker()
        self.model_registry = ModelRegistry()
        self.evaluation_framework = ModelEvaluationFramework()
        self.version_control = ModelVersionControl()

    def run_experiment(self, experiment_config: dict) -> dict:
        """Run complete ML experiment with tracking"""

        # Initialize experiment
        experiment_id = self.experiment_tracker.start_experiment(experiment_config)

        try:
            # Data preparation
            data_config = self._prepare_data(experiment_config)

            # Model training
            model, training_metrics = self._train_model(
                data_config,
                experiment_config['model_config']
            )

            # Model evaluation
            evaluation_results = self.evaluation_framework.evaluate_model(
                model,
                data_config['test_data']
            )

            # Register model version
            model_version = self.version_control.create_version(
                model,
                {
                    'experiment_id': experiment_id,
                    'training_metrics': training_metrics,
                    'evaluation_results': evaluation_results,
                    'config': experiment_config
                }
            )

            # Complete experiment
            experiment_result = self.experiment_tracker.complete_experiment(
                experiment_id,
                {
                    'model_version': model_version,
                    'metrics': {**training_metrics, **evaluation_results},
                    'status': 'completed'
                }
            )

            return experiment_result

        except Exception as e:
            # Log experiment failure
            self.experiment_tracker.fail_experiment(experiment_id, str(e))
            raise
```

### 3. Model Registry System
```python
class ModelRegistry:
    def __init__(self):
        self.storage = ModelStorage()
        self.metadata_manager = MetadataManager()
        self.artifact_manager = ArtifactManager()
        self.version_manager = VersionManager()

    def register_model(self, model_config: dict) -> str:
        """Register model with all artifacts and metadata"""

        # Store model artifacts
        artifact_paths = self.artifact_manager.store_artifacts(
            model_config['model'],
            model_config.get('artifacts', {})
        )

        # Extract and store metadata
        metadata = self.metadata_manager.extract_metadata(
            model_config['model'],
            model_config.get('additional_metadata', {})
        )

        # Create version
        version_info = self.version_manager.create_version(
            model_config['model_name'],
            {
                'artifacts': artifact_paths,
                'metadata': metadata,
                'config': model_config,
                'created_at': datetime.utcnow()
            }
        )

        # Store in registry
        model_id = self.storage.store_model({
            'name': model_config['model_name'],
            'version': version_info,
            'artifacts': artifact_paths,
            'metadata': metadata,
            'config': model_config
        })

        return model_id

    def get_model(self, model_id: str, version: str = None) -> dict:
        """Retrieve model by ID and version"""

        # Get model info
        model_info = self.storage.get_model(model_id)

        if version:
            # Get specific version
            version_info = self.version_manager.get_version(
                model_info['name'],
                version
            )
            model_info['version'] = version_info
            model_info['artifacts'] = self.artifact_manager.load_artifacts(
                version_info['artifacts']
            )
        else:
            # Get latest version
            latest_version = self.version_manager.get_latest_version(
                model_info['name']
            )
            model_info['version'] = latest_version
            model_info['artifacts'] = self.artifact_manager.load_artifacts(
                latest_version['artifacts']
            )

        return model_info
```

## MLOps Tools and Technologies

### Popular MLOps Platforms

#### 1. MLflow
```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

class MLflowIntegration:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(tracking_uri)

    def log_experiment(self, model, X_train, X_test, y_train, y_test):
        """Log complete experiment to MLflow"""

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'n_estimators': 100,
                'max_depth': 10,
                'learning_rate': 0.1
            })

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted')
            recall = recall_score(y_test, predictions, average='weighted')

            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            })

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log artifacts
            mlflow.log_artifact("confusion_matrix.png")
            mlflow.log_artifact("feature_importance.png")

            return mlflow.active_run().info.run_id
```

#### 2. Kubeflow
```python
from kfp import dsl
from kfp.components import create_component_from_func
from kfp.dsl import InputPath, OutputPath

@create_component_from_func
def preprocess_data(
    input_path: InputPath('CSV'),
    output_path: OutputPath('CSV')
):
    """Data preprocessing component"""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Load data
    df = pd.read_csv(input_path)

    # Preprocess
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Save processed data
    df.to_csv(output_path, index=False)

@create_component_from_func
def train_model(
    data_path: InputPath('CSV'),
    model_path: OutputPath('model')
):
    """Model training component"""
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # Load data
    df = pd.read_csv(data_path)

    # Split data
    X = df.drop('target', axis=1)
    y = df['target']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    joblib.dump(model, model_path)

@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML training pipeline'
)
def ml_pipeline():
    """Define ML pipeline"""

    # Data preprocessing
    preprocess_op = preprocess_data()

    # Model training
    train_op = train_model(
        data=preprocess_op.outputs['output']
    )
```

### 3. Airflow for ML Orchestration
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def train_model(**context):
    """Train ML model"""
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    # Load data
    df = pd.read_csv('/data/training_data.csv')

    # Prepare data
    X = df.drop('target', axis=1)
    y = df['target']

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save model
    model_path = f'/models/model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
    joblib.dump(model, model_path)

    # Push model path to XCom
    context['ti'].xcom_push(key='model_path', value=model_path)

def evaluate_model(**context):
    """Evaluate model performance"""
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    # Pull model path from XCom
    model_path = context['ti'].xcom_pull(key='model_path')

    # Load model and test data
    model = joblib.load(model_path)
    test_df = pd.read_csv('/data/test_data.csv')

    # Make predictions
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    predictions = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted')
    }

    # Log metrics
    print(f"Model evaluation metrics: {metrics}")

    return metrics

def deploy_model(**context):
    """Deploy model to production"""
    import mlflow
    import mlflow.sklearn
    import joblib

    # Pull model path from XCom
    model_path = context['ti'].xcom_pull(key='model_path')

    # Load model
    model = joblib.load(model_path)

    # Register model with MLflow
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(model_path, "original_model")

    # Update model registry
    print(f"Model deployed: {model_path}")

# Define DAG
default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    schedule_interval=timedelta(days=1),
    catchup=False,
    description='Daily ML model training and deployment'
)

# Define tasks
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag
)

# Set task dependencies
train_task >> evaluate_task >> deploy_task
```

## Key Takeaways

### Essential MLOps Components
1. **Data Management**: Versioning, quality control, lineage tracking
2. **Experiment Tracking**: Reproducible experiments with comprehensive logging
3. **Model Registry**: Centralized model storage with versioning
4. **CI/CD Pipeline**: Automated testing and deployment
5. **Monitoring**: Performance tracking and drift detection
6. **Infrastructure**: Scalable and reliable deployment environment

### Best Practices
- **Automate Everything**: From data preparation to model deployment
- **Version Control**: Track code, data, models, and configurations
- **Documentation**: Maintain comprehensive documentation
- **Testing**: Implement thorough testing at all stages
- **Monitoring**: Continuously monitor model performance and data drift

### Common Challenges
- **Data Quality**: Ensuring consistent and reliable data
- **Model Drift**: Detecting and handling performance degradation
- **Infrastructure**: Managing complex deployment environments
- **Collaboration**: Coordinating between different teams
- **Compliance**: Meeting regulatory requirements

---

## Next Steps

Continue to [Module 2: LLMOps and Generative AI Operations](02_LLMops_and_Generative_AI_Operations.md) to learn about specialized operations for large language models and generative AI systems.

## Quick Reference

### Key Concepts
- **MLOps**: Machine Learning Operations
- **Model Registry**: Centralized model storage
- **Experiment Tracking**: Reproducible experiment management
- **CI/CD**: Continuous Integration/Continuous Deployment
- **Data Versioning**: Tracking data changes over time

### Essential Tools
- **MLflow**: Experiment tracking and model management
- **Kubeflow**: Kubernetes-native ML platform
- **Airflow**: Workflow orchestration
- **DVC**: Data Version Control
- **Weights & Biases**: Experiment tracking and visualization

### Common Patterns
- **Training Pipeline**: Automated model training
- **Deployment Pipeline**: Automated model deployment
- **Monitoring Pipeline**: Continuous performance tracking
- **Retraining Pipeline**: Scheduled model updates
- **Validation Pipeline**: Data and model quality checks