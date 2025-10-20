---
title: "Mlops And Ai Deployment Strategies - Module 6: Model"
description: "## Navigation. Comprehensive guide covering algorithm, algorithms, machine learning, model training, data preprocessing. Part of AI documentation system with..."
keywords: "machine learning, algorithm, algorithm, algorithms, machine learning, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# Module 6: Model Management and Versioning

## Navigation
- **← Previous**: [05_Monitoring_and_Observability.md](05_Monitoring_and_Observability.md)
- **→ Next**: [07_CICD_for_Machine_Learning.md](07_CICD_for_Machine_Learning.md)
- **↑ Up**: [README.md](README.md)

## Overview

Model management and versioning are fundamental components of MLOps, enabling teams to track, deploy, and manage machine learning models throughout their lifecycle. This module covers comprehensive model registry systems, version control strategies, and lifecycle management.

## Complete Model Registry System

### Advanced Model Registry Implementation

```python
import hashlib
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pickle
import joblib
import dill
import cloudpickle
from dataclasses import dataclass, asdict, field
import boto3
import botocore
from sqlalchemy import create_engine, Column, String, Float, DateTime, JSON, Integer, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from enum import Enum
import yaml

Base = declarative_base()

class ModelStatus(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class StorageBackend(Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"

@dataclass
class ModelMetadata:
    """Comprehensive model metadata structure"""
    model_id: str
    model_name: str
    version: str
    algorithm: str
    framework: str
    framework_version: str
    created_at: datetime
    created_by: str
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None
    status: ModelStatus = ModelStatus.DEVELOPMENT
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    description: str = ""
    training_data_version: str = ""
    validation_data_version: str = ""
    feature_schema: Dict[str, str] = field(default_factory=dict)
    target_schema: Dict[str, str] = field(default_factory=dict)
    model_size_bytes: int = 0
    model_hash: str = ""
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)
    environment: str = ""
    git_commit: str = ""
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    parent_model_id: Optional[str] = None
    performance_score: float = 0.0
    business_value: float = 0.0
    compliance_status: str = ""
    risk_level: str = ""
    owner: str = ""
    team: str = ""
    cost_per_inference: float = 0.0
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

class ModelRecord(Base):
    """Database model for model registry"""
    __tablename__ = 'models'

    model_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)
    algorithm = Column(String(100))
    framework = Column(String(50))
    framework_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_by = Column(String(255))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    updated_by = Column(String(255))
    status = Column(String(20), default=ModelStatus.DEVELOPMENT.value, index=True)
    metrics = Column(JSON)
    parameters = Column(JSON)
    hyperparameters = Column(JSON)
    tags = Column(JSON)
    description = Column(Text)
    training_data_version = Column(String(100))
    validation_data_version = Column(String(100))
    feature_schema = Column(JSON)
    target_schema = Column(JSON)
    model_size_bytes = Column(Integer)
    model_hash = Column(String(64), unique=True)
    artifact_paths = Column(JSON)
    dependencies = Column(JSON)
    environment = Column(String(50))
    git_commit = Column(String(40))
    experiment_id = Column(String(36))
    run_id = Column(String(36))
    parent_model_id = Column(String(36))
    performance_score = Column(Float)
    business_value = Column(Float)
    compliance_status = Column(String(20))
    risk_level = Column(String(20))
    owner = Column(String(255))
    team = Column(String(100))
    cost_per_inference = Column(Float)
    deployment_config = Column(JSON)
    custom_metadata = Column(JSON)
    deployed_at = Column(DateTime)
    retired_at = Column(DateTime)
    is_active = Column(Boolean, default=True)

    # Relationships
    deployments = relationship("ModelDeployment", back_populates="model")
    evaluations = relationship("ModelEvaluation", back_populates="model")
    lineage = relationship("ModelLineage", foreign_keys="[ModelLineage.child_model_id]", back_populates="child_model")

class ModelDeployment(Base):
    """Track model deployments"""
    __tablename__ = 'model_deployments'

    deployment_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(36), ForeignKey('models.model_id'))
    environment = Column(String(50), nullable=False)
    deployment_config = Column(JSON)
    deployment_status = Column(String(20))
    deployed_at = Column(DateTime, default=datetime.utcnow)
    deployed_by = Column(String(255))
    endpoint_url = Column(String(500))
    health_status = Column(String(20))
    last_health_check = Column(DateTime)
    rollback_deployment_id = Column(String(36))

    model = relationship("ModelRecord", back_populates="deployments")

class ModelEvaluation(Base):
    """Track model evaluations"""
    __tablename__ = 'model_evaluations'

    evaluation_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String(36), ForeignKey('models.model_id'))
    dataset_name = Column(String(255), nullable=False)
    dataset_version = Column(String(100))
    evaluation_type = Column(String(50), nullable=False)  # 'validation', 'test', 'production'
    metrics = Column(JSON, nullable=False)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    evaluated_by = Column(String(255))
    comments = Column(Text)

    model = relationship("ModelRecord", back_populations="evaluations")

class ModelLineage(Base):
    """Track model lineage and relationships"""
    __tablename__ = 'model_lineage'

    lineage_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_model_id = Column(String(36), ForeignKey('models.model_id'))
    child_model_id = Column(String(36), ForeignKey('models.model_id'))
    relationship_type = Column(String(50), nullable=False)  # 'parent', 'fork', 'retrain'
    created_at = Column(DateTime, default=datetime.utcnow)

    parent_model = relationship("ModelRecord", foreign_keys=[parent_model_id], back_populates="lineage")
    child_model = relationship("ModelRecord", foreign_keys=[child_model_id])

class AdvancedModelRegistry:
    """
    Advanced model registry with comprehensive features for ML model management.
    """

    def __init__(self,
                 storage_backend: StorageBackend = StorageBackend.LOCAL,
                 storage_config: Dict[str, Any] = None,
                 db_config: Dict[str, Any] = None,
                 enable_async: bool = True,
                 enable_cache: bool = True):

        self.storage_backend = storage_backend
        self.storage_config = storage_config or {}
        self.db_config = db_config or {'url': 'sqlite:///model_registry.db'}
        self.enable_async = enable_async
        self.enable_cache = enable_cache

        # Initialize storage backend
        self.initialize_storage()

        # Initialize database
        self.initialize_database()

        # Initialize cache
        self.cache = {}
        if enable_cache:
            self.initialize_cache()

        # Initialize async components
        if enable_async:
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.event_queue = asyncio.Queue()
            asyncio.create_task(self.process_events())

        # Initialize validation
        self.validators = {
            'model': ModelValidator(),
            'metadata': MetadataValidator(),
            'deployment': DeploymentValidator()
        }

        # Initialize search index
        self.search_index = SearchIndex()

        # Performance tracking
        self.performance_metrics = PerformanceTracker()

        # Event handlers
        self.event_handlers = []

        logging.info("Advanced Model Registry initialized")

    def initialize_storage(self):
        """Initialize storage backend"""
        if self.storage_backend == StorageBackend.LOCAL:
            self.storage_path = Path(self.storage_config.get('path', './model_registry'))
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.s3_client = None

        elif self.storage_backend == StorageBackend.S3:
            self.s3_client = boto3.client('s3')
            self.s3_bucket = self.storage_config['bucket']
            self.storage_path = None

        elif self.storage_backend == StorageBackend.GCS:
            # Initialize Google Cloud Storage client
            from google.cloud import storage
            self.gcs_client = storage.Client()
            self.gcs_bucket = self.gcs_client.bucket(self.storage_config['bucket'])
            self.storage_path = None

        elif self.storage_backend == StorageBackend.AZURE_BLOB:
            # Initialize Azure Blob Storage client
            from azure.storage.blob import BlobServiceClient
            self.azure_client = BlobServiceClient(
                account_url=self.storage_config['account_url'],
                credential=self.storage_config['credential']
            )
            self.azure_container = self.azure_client.get_container_client(self.storage_config['container'])
            self.storage_path = None

    def initialize_database(self):
        """Initialize database connection"""
        self.engine = create_engine(self.db_config['url'])
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def initialize_cache(self):
        """Initialize caching layer"""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.storage_config.get('redis_host', 'localhost'),
                port=self.storage_config.get('redis_port', 6379),
                db=0,
                decode_responses=True
            )
        except ImportError:
            logging.warning("Redis not available, using in-memory cache")
            self.redis_client = None

    async def register_model(self,
                           model: Any,
                           model_name: str,
                           version: str,
                           metadata: ModelMetadata,
                           artifacts: Dict[str, Any] = None,
                           validate: bool = True) -> str:
        """Register a new model with comprehensive validation"""

        try:
            # Validate inputs
            if validate:
                self.validators['model'].validate(model)
                self.validators['metadata'].validate(metadata)

            # Generate model ID if not provided
            if not metadata.model_id:
                metadata.model_id = self.generate_model_id(model_name, version)

            # Calculate model hash and size
            model_bytes = self.serialize_model(model)
            metadata.model_hash = hashlib.sha256(model_bytes).hexdigest()
            metadata.model_size_bytes = len(model_bytes)

            # Save model artifact
            model_path = await self.save_model_artifact(model, metadata.model_id)

            # Save additional artifacts
            artifact_paths = {}
            if artifacts:
                for artifact_name, artifact_data in artifacts.items():
                    artifact_path = await self.save_artifact(artifact_data, metadata.model_id, artifact_name)
                    artifact_paths[artifact_name] = artifact_path

            metadata.artifact_paths = artifact_paths

            # Save metadata to database
            model_record = await self.save_metadata_to_db(metadata)

            # Update search index
            await self.search_index.add_model(metadata)

            # Update cache
            if self.enable_cache:
                await self.update_cache(metadata.model_id, model, metadata)

            # Emit event
            await self.emit_event('model_registered', {
                'model_id': metadata.model_id,
                'model_name': model_name,
                'version': version,
                'status': metadata.status.value
            })

            logging.info(f"Model registered successfully: {metadata.model_id}")
            return metadata.model_id

        except Exception as e:
            logging.error(f"Failed to register model: {e}")
            raise

    async def get_model(self,
                      model_id: str = None,
                      model_name: str = None,
                      version: str = None,
                      use_cache: bool = True) -> tuple[Any, ModelMetadata]:
        """Retrieve model with metadata"""

        if model_id is None:
            model_id = self.generate_model_id(model_name, version)

        # Check cache first
        if use_cache and self.enable_cache:
            cached_result = await self.get_from_cache(model_id)
            if cached_result:
                return cached_result

        # Get metadata from database
        session = self.Session()
        try:
            model_record = session.query(ModelRecord).filter_by(model_id=model_id, is_active=True).first()

            if not model_record:
                raise ValueError(f"Model not found: {model_id}")

            # Convert to metadata
            metadata = self.record_to_metadata(model_record)

            # Load model artifact
            model = await self.load_model_artifact(model_id)

            # Update cache
            if self.enable_cache and use_cache:
                await self.update_cache(model_id, model, metadata)

            # Update performance metrics
            self.performance_metrics.track_access(model_id)

            return model, metadata

        finally:
            session.close()

    async def promote_model(self,
                          model_id: str,
                          target_stage: ModelStatus,
                          reason: str = "",
                          validation_required: bool = True) -> bool:
        """Promote model to different stage with validation"""

        valid_transitions = {
            ModelStatus.DEVELOPMENT: [ModelStatus.STAGING, ModelStatus.ARCHIVED],
            ModelStatus.STAGING: [ModelStatus.PRODUCTION, ModelStatus.DEVELOPMENT, ModelStatus.ARCHIVED],
            ModelStatus.PRODUCTION: [ModelStatus.STAGING, ModelStatus.DEPRECATED, ModelStatus.ARCHIVED],
            ModelStatus.ARCHIVED: [ModelStatus.DEVELOPMENT],
            ModelStatus.DEPRECATED: [ModelStatus.ARCHIVED]
        }

        session = self.Session()
        try:
            model_record = session.query(ModelRecord).filter_by(model_id=model_id, is_active=True).first()

            if not model_record:
                raise ValueError(f"Model not found: {model_id}")

            current_status = ModelStatus(model_record.status)

            # Validate transition
            if target_stage not in valid_transitions.get(current_status, []):
                raise ValueError(f"Invalid transition from {current_status} to {target_stage}")

            # Validation requirements
            if validation_required and target_stage == ModelStatus.PRODUCTION:
                await self.validate_for_production(model_id)

            # Update status
            old_status = current_status
            model_record.status = target_stage.value
            model_record.updated_at = datetime.utcnow()

            if target_stage == ModelStatus.PRODUCTION:
                model_record.deployed_at = datetime.utcnow()
            elif target_stage in [ModelStatus.ARCHIVED, ModelStatus.DEPRECATED]:
                model_record.retired_at = datetime.utcnow()

            session.commit()

            # Emit event
            await self.emit_event('model_promoted', {
                'model_id': model_id,
                'from_status': old_status.value,
                'to_status': target_stage.value,
                'reason': reason
            })

            # Trigger deployment automation
            if target_stage == ModelStatus.PRODUCTION:
                await self.trigger_deployment_automation(model_id)

            logging.info(f"Model {model_id} promoted from {old_status} to {target_stage}")
            return True

        finally:
            session.close()

    async def list_models(self,
                         filters: Dict[str, Any] = None,
                         sort_by: str = 'created_at',
                         sort_order: str = 'desc',
                         limit: int = 100,
                         offset: int = 0) -> List[ModelMetadata]:
        """List models with filtering and pagination"""

        session = self.Session()
        try:
            query = session.query(ModelRecord).filter_by(is_active=True)

            # Apply filters
            if filters:
                if 'model_name' in filters:
                    query = query.filter(ModelRecord.model_name.ilike(f"%{filters['model_name']}%"))
                if 'status' in filters:
                    query = query.filter_by(status=filters['status'])
                if 'algorithm' in filters:
                    query = query.filter_by(algorithm=filters['algorithm'])
                if 'framework' in filters:
                    query = query.filter_by(framework=filters['framework'])
                if 'tags' in filters:
                    # Filter by tags (assuming JSON array)
                    for tag in filters['tags']:
                        query = query.filter(ModelRecord.tags.contains([tag]))
                if 'owner' in filters:
                    query = query.filter_by(owner=filters['owner'])
                if 'team' in filters:
                    query = query.filter_by(team=filters['team'])
                if 'created_after' in filters:
                    query = query.filter(ModelRecord.created_at >= filters['created_after'])
                if 'created_before' in filters:
                    query = query.filter(ModelRecord.created_at <= filters['created_before'])

            # Apply sorting
            if hasattr(ModelRecord, sort_by):
                sort_column = getattr(ModelRecord, sort_by)
                if sort_order.lower() == 'desc':
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())

            # Apply pagination
            models = query.offset(offset).limit(limit).all()

            # Convert to metadata
            return [self.record_to_metadata(model) for model in models]

        finally:
            session.close()

    async def search_models(self,
                          query: str,
                          filters: Dict[str, Any] = None,
                          limit: int = 50) -> List[ModelMetadata]:
        """Search models using full-text search"""

        return await self.search_index.search(query, filters, limit)

    async def create_model_lineage(self,
                                parent_model_id: str,
                                child_model_id: str,
                                relationship_type: str) -> bool:
        """Create lineage relationship between models"""

        session = self.Session()
        try:
            # Verify both models exist
            parent = session.query(ModelRecord).filter_by(model_id=parent_model_id, is_active=True).first()
            child = session.query(ModelRecord).filter_by(model_id=child_model_id, is_active=True).first()

            if not parent or not child:
                raise ValueError("Parent or child model not found")

            # Create lineage record
            lineage = ModelLineage(
                parent_model_id=parent_model_id,
                child_model_id=child_model_id,
                relationship_type=relationship_type
            )

            session.add(lineage)
            session.commit()

            # Emit event
            await self.emit_event('lineage_created', {
                'parent_model_id': parent_model_id,
                'child_model_id': child_model_id,
                'relationship_type': relationship_type
            })

            logging.info(f"Lineage created: {parent_model_id} -> {child_model_id}")
            return True

        finally:
            session.close()

    async def get_model_lineage(self, model_id: str) -> Dict[str, List[ModelMetadata]]:
        """Get complete lineage for a model"""

        session = self.Session()
        try:
            # Get ancestors
            ancestors_query = session.query(ModelLineage).filter_by(child_model_id=model_id)
            ancestors = []
            for lineage in ancestors_query:
                parent_model = session.query(ModelRecord).filter_by(model_id=lineage.parent_model_id, is_active=True).first()
                if parent_model:
                    ancestors.append(self.record_to_metadata(parent_model))

            # Get descendants
            descendants_query = session.query(ModelLineage).filter_by(parent_model_id=model_id)
            descendants = []
            for lineage in descendants_query:
                child_model = session.query(ModelRecord).filter_by(model_id=lineage.child_model_id, is_active=True).first()
                if child_model:
                    descendants.append(self.record_to_metadata(child_model))

            return {
                'ancestors': ancestors,
                'descendants': descendants
            }

        finally:
            session.close()

    async def compare_models(self,
                          model_ids: List[str],
                          metrics: List[str] = None) -> Dict[str, Any]:
        """Compare multiple models"""

        if len(model_ids) < 2:
            raise ValueError("At least 2 models required for comparison")

        models_data = []
        for model_id in model_ids:
            try:
                _, metadata = await self.get_model(model_id)
                models_data.append(metadata)
            except ValueError:
                logging.warning(f"Model not found: {model_id}")
                continue

        if len(models_data) < 2:
            raise ValueError("Not enough valid models for comparison")

        # Calculate comparison metrics
        comparison = {
            'models': models_data,
            'metrics_comparison': {},
            'performance_diff': {},
            'recommendation': None
        }

        if metrics:
            for metric in metrics:
                values = []
                for model in models_data:
                    if metric in model.metrics:
                        values.append((model.model_id, model.metrics[metric]))

                if values:
                    comparison['metrics_comparison'][metric] = values
                    if len(values) >= 2:
                        max_val = max(v[1] for v in values)
                        min_val = min(v[1] for v in values)
                        comparison['performance_diff'][metric] = {
                            'range': max_val - min_val,
                            'best_model': next(v[0] for v in values if v[1] == max_val)
                        }

        # Generate recommendation
        comparison['recommendation'] = self.generate_comparison_recommendation(comparison)

        return comparison

    async def validate_for_production(self, model_id: str) -> bool:
        """Validate model for production deployment"""

        model, metadata = await self.get_model(model_id)

        validation_results = {
            'performance': await self.validate_performance(metadata),
            'compliance': await self.validate_compliance(metadata),
            'security': await self.validate_security(metadata),
            'operational': await self.validate_operational_readiness(metadata)
        }

        overall_valid = all(result['valid'] for result in validation_results.values())

        if not overall_valid:
            failed_validations = [k for k, v in validation_results.items() if not v['valid']]
            raise ValueError(f"Model validation failed: {failed_validations}")

        return True

    async def generate_model_report(self, model_id: str) -> Dict[str, Any]:
        """Generate comprehensive model report"""

        model, metadata = await self.get_model(model_id)

        # Get lineage
        lineage = await self.get_model_lineage(model_id)

        # Get deployment history
        deployments = await self.get_deployment_history(model_id)

        # Get evaluation history
        evaluations = await self.get_evaluation_history(model_id)

        # Calculate statistics
        stats = await self.calculate_model_statistics(model_id)

        report = {
            'model_info': asdict(metadata),
            'lineage': lineage,
            'deployment_history': deployments,
            'evaluation_history': evaluations,
            'statistics': stats,
            'recommendations': await self.generate_model_recommendations(model_id),
            'generated_at': datetime.utcnow().isoformat()
        }

        return report

    # Helper methods
    def generate_model_id(self, model_name: str, version: str) -> str:
        """Generate unique model ID"""
        unique_string = f"{model_name}_{version}_{datetime.utcnow().timestamp()}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def serialize_model(self, model: Any) -> bytes:
        """Serialize model using appropriate method"""
        try:
            # Try joblib first (best for scikit-learn)
            return joblib.dumps(model)
        except Exception:
            try:
                # Try pickle
                return pickle.dumps(model)
            except Exception:
                try:
                    # Try cloudpickle (for complex objects)
                    return cloudpickle.dumps(model)
                except Exception:
                    try:
                        # Try dill (most comprehensive)
                        return dill.dumps(model)
                    except Exception as e:
                        raise ValueError(f"Could not serialize model: {e}")

    async def save_model_artifact(self, model: Any, model_id: str) -> str:
        """Save model artifact to storage"""

        model_bytes = self.serialize_model(model)

        if self.storage_backend == StorageBackend.LOCAL:
            model_path = self.storage_path / "models" / f"{model_id}.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, 'wb') as f:
                f.write(model_bytes)
            return str(model_path)

        elif self.storage_backend == StorageBackend.S3:
            key = f"models/{model_id}.pkl"
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=model_bytes
            )
            return f"s3://{self.s3_bucket}/{key}"

        # Add other storage backends...

    async def load_model_artifact(self, model_id: str) -> Any:
        """Load model artifact from storage"""

        if self.storage_backend == StorageBackend.LOCAL:
            model_path = self.storage_path / "models" / f"{model_id}.pkl"
            with open(model_path, 'rb') as f:
                return joblib.load(f)

        elif self.storage_backend == StorageBackend.S3:
            key = f"models/{model_id}.pkl"
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=key)
            return joblib.loads(response['Body'].read())

        # Add other storage backends...

    def record_to_metadata(self, record: ModelRecord) -> ModelMetadata:
        """Convert database record to metadata object"""
        return ModelMetadata(
            model_id=record.model_id,
            model_name=record.model_name,
            version=record.version,
            algorithm=record.algorithm,
            framework=record.framework,
            framework_version=record.framework_version,
            created_at=record.created_at,
            created_by=record.created_by,
            updated_at=record.updated_at,
            updated_by=record.updated_by,
            status=ModelStatus(record.status),
            metrics=record.metrics or {},
            parameters=record.parameters or {},
            hyperparameters=record.hyperparameters or {},
            tags=record.tags or [],
            description=record.description or "",
            training_data_version=record.training_data_version or "",
            validation_data_version=record.validation_data_version or "",
            feature_schema=record.feature_schema or {},
            target_schema=record.target_schema or {},
            model_size_bytes=record.model_size_bytes or 0,
            model_hash=record.model_hash or "",
            artifact_paths=record.artifact_paths or {},
            dependencies=record.dependencies or {},
            environment=record.environment or "",
            git_commit=record.git_commit or "",
            experiment_id=record.experiment_id,
            run_id=record.run_id,
            parent_model_id=record.parent_model_id,
            performance_score=record.performance_score or 0.0,
            business_value=record.business_value or 0.0,
            compliance_status=record.compliance_status or "",
            risk_level=record.risk_level or "",
            owner=record.owner or "",
            team=record.team or "",
            cost_per_inference=record.cost_per_inference or 0.0,
            deployment_config=record.deployment_config or {},
            custom_metadata=record.custom_metadata or {}
        )

    async def save_metadata_to_db(self, metadata: ModelMetadata) -> ModelRecord:
        """Save metadata to database"""
        session = self.Session()
        try:
            model_record = ModelRecord(
                model_id=metadata.model_id,
                model_name=metadata.model_name,
                version=metadata.version,
                algorithm=metadata.algorithm,
                framework=metadata.framework,
                framework_version=metadata.framework_version,
                created_at=metadata.created_at,
                created_by=metadata.created_by,
                updated_at=metadata.updated_at,
                updated_by=metadata.updated_by,
                status=metadata.status.value,
                metrics=metadata.metrics,
                parameters=metadata.parameters,
                hyperparameters=metadata.hyperparameters,
                tags=metadata.tags,
                description=metadata.description,
                training_data_version=metadata.training_data_version,
                validation_data_version=metadata.validation_data_version,
                feature_schema=metadata.feature_schema,
                target_schema=metadata.target_schema,
                model_size_bytes=metadata.model_size_bytes,
                model_hash=metadata.model_hash,
                artifact_paths=metadata.artifact_paths,
                dependencies=metadata.dependencies,
                environment=metadata.environment,
                git_commit=metadata.git_commit,
                experiment_id=metadata.experiment_id,
                run_id=metadata.run_id,
                parent_model_id=metadata.parent_model_id,
                performance_score=metadata.performance_score,
                business_value=metadata.business_value,
                compliance_status=metadata.compliance_status,
                risk_level=metadata.risk_level,
                owner=metadata.owner,
                team=metadata.team,
                cost_per_inference=metadata.cost_per_inference,
                deployment_config=metadata.deployment_config,
                custom_metadata=metadata.custom_metadata
            )

            session.add(model_record)
            session.commit()
            return model_record

        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    # Additional helper methods would be implemented here...
    # (cache management, event processing, search indexing, etc.)

class ModelValidator:
    """Validate model artifacts and structure"""

    def validate(self, model: Any) -> bool:
        """Validate model structure"""
        # Basic validation - can be extended
        return model is not None

class MetadataValidator:
    """Validate metadata structure"""

    def validate(self, metadata: ModelMetadata) -> bool:
        """Validate metadata completeness"""
        required_fields = ['model_name', 'version', 'algorithm', 'framework', 'created_by']
        return all(getattr(metadata, field, None) for field in required_fields)

class DeploymentValidator:
    """Validate deployment configurations"""

    def validate(self, deployment_config: Dict[str, Any]) -> bool:
        """Validate deployment configuration"""
        return isinstance(deployment_config, dict) and 'environment' in deployment_config

class SearchIndex:
    """Simple search index for models"""

    async def add_model(self, metadata: ModelMetadata):
        """Add model to search index"""
        # Implementation would use proper search engine like Elasticsearch
        pass

    async def search(self, query: str, filters: Dict[str, Any], limit: int) -> List[ModelMetadata]:
        """Search models"""
        # Simple implementation - in production, use proper search engine
        return []

class PerformanceTracker:
    """Track registry performance metrics"""

    def track_access(self, model_id: str):
        """Track model access"""
        # Implementation would track access patterns
        pass

# Usage example
if __name__ == "__main__":
    # Initialize registry
    registry = AdvancedModelRegistry(
        storage_backend=StorageBackend.LOCAL,
        storage_config={'path': './model_registry'},
        db_config={'url': 'sqlite:///model_registry.db'}
    )

    # Example usage would go here...
```

## Key Takeaways

### Model Registry Components
1. **Comprehensive Metadata**: Complete model lifecycle tracking
2. **Multiple Storage Backends**: Support for local, S3, GCS, Azure
3. **Advanced Search**: Full-text search and filtering
4. **Lineage Tracking**: Model relationships and provenance
5. **Performance Monitoring**: Access patterns and usage tracking

### Model Lifecycle Management
1. **Version Control**: Comprehensive versioning system
2. **Stage Management**: Development → Staging → Production
3. **Validation**: Automated validation for deployments
4. **Lineage Tracking**: Complete model provenance
5. **Reporting**: Comprehensive model reports and statistics

### Best Practices
- **Comprehensive Metadata**: Track all relevant model information
- **Validation**: Ensure models meet production requirements
- **Lineage Tracking**: Maintain complete model provenance
- **Performance Monitoring**: Track model usage and performance
- **Security**: Implement proper access controls and validation

### Common Challenges
- **Scalability**: Managing large numbers of models
- **Consistency**: Ensuring consistency across storage backends
- **Performance**: Optimizing for high-volume operations
- **Integration**: Integrating with existing ML workflows
- **Compliance**: Meeting regulatory requirements

---

## Next Steps

Continue to [Module 7: CI/CD for Machine Learning](07_CICD_for_Machine_Learning.md) to learn about automated CI/CD pipelines for ML systems.

## Quick Reference

### Key Concepts
- **Model Registry**: Centralized model storage and management
- **Version Control**: Tracking model versions and changes
- **Lineage Tracking**: Model provenance and relationships
- **Metadata Management**: Comprehensive model information
- **Lifecycle Management**: Complete model lifecycle control

### Essential Tools
- **MLflow**: Experiment tracking and model management
- **DVC**: Data version control
- **Weights & Biases**: Experiment tracking
- **Artifactory**: Artifact management
- **Elasticsearch**: Search and indexing

### Common Patterns
- **Model Versioning**: Semantic versioning for models
- **Stage Promotion**: Progressive deployment stages
- **Lineage Tracking**: Parent-child relationships
- **Metadata Enrichment**: Comprehensive model documentation
- **Automated Validation**: Pre-deployment validation checks