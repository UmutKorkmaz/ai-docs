# Electronic Health Records (EHR) Management AI

## Module Overview
This module provides comprehensive implementation examples of AI systems for Electronic Health Records management, including clinical NLP, predictive analytics, and clinical decision support.

## Table of Contents
1. [Comprehensive EHR AI System](#comprehensive-ehr-ai-system)
2. [Clinical NLP Processing](#clinical-nlp-processing)
3. [EHR Data Integration](#ehr-data-integration)
4. [Clinical Rules Engine](#clinical-rules-engine)
5. [Predictive Analytics](#predictive-analytics)
6. [Real-world Implementation](#real-world-implementation)
7. [Clinical Workflow Integration](#clinical-workflow-integration)

## Comprehensive EHR AI System

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import redis
import pymongo
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.condition import Condition
import requests
from bs4 import BeautifulSoup
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class EHRAIManager:
    """
    Comprehensive AI system for Electronic Health Records management
    including clinical NLP, predictive analytics, and decision support
    """

    def __init__(self, config: Dict):
        self.config = config
        self.nlp = spacy.load("en_core_medical_lg")
        self.clinical_nlp = ClinicalNLPProcessor()
        self.predictive_models = {}
        self.data_integrator = EHRDataIntegrator(config['data_sources'])
        self.knowledge_base = ClinicalKnowledgeBase()
        self.clinical_rules_engine = ClinicalRulesEngine()
        self.quality_metrics = QualityMetricsManager()

        # Initialize database connections
        self.db_connections = self.initialize_db_connections()

        # Initialize Redis for caching
        self.redis_client = redis.Redis(
            host=config['redis']['host'],
            port=config['redis']['port'],
            db=0
        )

        # Initialize predictive models
        self.initialize_predictive_models()

        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def initialize_db_connections(self) -> Dict:
        """Initialize connections to various healthcare databases"""

        connections = {}

        # Epic EHR connection
        if 'epic' in self.config['data_sources']:
            connections['epic'] = {
                'engine': create_engine(self.config['data_sources']['epic']['connection_string']),
                'type': 'sql'
            }

        # Cerner EHR connection
        if 'cerner' in self.config['data_sources']:
            connections['cerner'] = {
                'engine': create_engine(self.config['data_sources']['cerner']['connection_string']),
                'type': 'sql'
            }

        # MongoDB for clinical documents
        if 'mongodb' in self.config['data_sources']:
            connections['mongodb'] = pymongo.MongoClient(
                self.config['data_sources']['mongodb']['connection_string']
            )

        return connections

    def initialize_predictive_models(self):
        """Initialize predictive models for various clinical use cases"""

        # Readmission risk model
        self.predictive_models['readmission'] = self.build_readmission_model()

        # Sepsis prediction model
        self.predictive_models['sepsis'] = self.build_sepsis_model()

        # Diabetes risk model
        self.predictive_models['diabetes'] = self.build_diabetes_model()

        # Heart failure prediction model
        self.predictive_models['heart_failure'] = self.build_heart_failure_model()

        # Clinical deterioration model
        self.predictive_models['deterioration'] = self.build_deterioration_model()

    def build_readmission_model(self) -> Dict:
        """Build patient readmission risk prediction model"""

        # Feature engineering for readmission
        feature_engineer = ReadmissionFeatureEngineer()

        # Model pipeline
        model_pipeline = {
            'feature_engineer': feature_engineer,
            'scaler': StandardScaler(),
            'classifier': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'calibrator': ProbabilityCalibrator()
        }

        return model_pipeline

    def process_patient_record(self, patient_id: str) -> Dict:
        """
        Process complete patient record and generate comprehensive insights
        """

        try:
            # Check cache first
            cached_result = self.redis_client.get(f"patient_record_{patient_id}")
            if cached_result:
                return json.loads(cached_result)

            # Fetch patient data from EHR
            patient_data = self.data_integrator.get_patient_data(patient_id)

            # Process clinical notes
            processed_notes = self.clinical_nlp.process_clinical_notes(
                patient_data['clinical_notes']
            )

            # Extract structured data
            structured_data = self.extract_structured_data(patient_data)

            # Generate predictions
            predictions = self.generate_clinical_predictions(
                patient_id, structured_data
            )

            # Apply clinical rules
            rule_outcomes = self.clinical_rules_engine.apply_rules(
                structured_data, predictions
            )

            # Generate recommendations
            recommendations = self.generate_recommendations(
                predictions, rule_outcomes
            )

            # Compile comprehensive result
            result = {
                'patient_id': patient_id,
                'processed_data': structured_data,
                'predictions': predictions,
                'rule_outcomes': rule_outcomes,
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat(),
                'data_quality_score': self.assess_data_quality(patient_data)
            }

            # Cache result
            self.redis_client.setex(
                f"patient_record_{patient_id}",
                3600,  # 1 hour cache
                json.dumps(result, default=str)
            )

            return result

        except Exception as e:
            self.logger.error(f"Error processing patient record: {str(e)}")
            return {
                'patient_id': patient_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def extract_structured_data(self, patient_data: Dict) -> Dict:
        """Extract and structure patient data"""

        structured_data = {
            'demographics': self.extract_demographics(patient_data),
            'vitals': self.extract_vitals(patient_data),
            'lab_results': self.extract_lab_results(patient_data),
            'medications': self.extract_medications(patient_data),
            'diagnoses': self.extract_diagnoses(patient_data),
            'procedures': self.extract_procedures(patient_data),
            'social_history': self.extract_social_history(patient_data),
            'family_history': self.extract_family_history(patient_data)
        }

        return structured_data

    def generate_clinical_predictions(self, patient_id: str,
                                    structured_data: Dict) -> Dict:
        """Generate clinical predictions using ML models"""

        predictions = {}

        # Readmission risk
        if 'readmission' in self.predictive_models:
            readmission_features = self.feature_engineer.extract_readmission_features(
                structured_data
            )
            readmission_risk = self.predictive_models['readmission']['classifier'].predict_proba(
                [readmission_features]
            )[0][1]
            predictions['readmission_risk'] = {
                'probability': float(readmission_risk),
                'level': self.risk_level(readmission_risk),
                'confidence': self.calculate_prediction_confidence(readmission_features)
            }

        # Sepsis risk
        if 'sepsis' in self.predictive_models:
            sepsis_features = self.feature_engineer.extract_sepsis_features(
                structured_data
            )
            sepsis_risk = self.predictive_models['sepsis']['classifier'].predict_proba(
                [sepsis_features]
            )[0][1]
            predictions['sepsis_risk'] = {
                'probability': float(sepsis_risk),
                'level': self.risk_level(sepsis_risk),
                'timeframe': '24 hours',
                'confidence': self.calculate_prediction_confidence(sepsis_features)
            }

        # Disease risk predictions
        disease_risks = self.predict_disease_risks(structured_data)
        predictions['disease_risks'] = disease_risks

        return predictions

    def generate_recommendations(self, predictions: Dict,
                               rule_outcomes: Dict) -> List[Dict]:
        """Generate clinical recommendations based on predictions and rules"""

        recommendations = []

        # High readmission risk recommendations
        if predictions.get('readmission_risk', {}).get('level') == 'high':
            recommendations.extend([
                {
                    'type': 'care_coordination',
                    'priority': 'high',
                    'action': 'Schedule follow-up within 7 days',
                    'rationale': 'High readmission risk detected',
                    'evidence': predictions['readmission_risk']
                },
                {
                    'type': 'medication_reconciliation',
                    'priority': 'high',
                    'action': 'Perform comprehensive medication review',
                    'rationale': 'Medication-related issues contribute to readmissions'
                }
            ])

        # Sepsis risk recommendations
        if predictions.get('sepsis_risk', {}).get('level') == 'high':
            recommendations.extend([
                {
                    'type': 'monitoring',
                    'priority': 'urgent',
                    'action': 'Increase vital sign monitoring frequency',
                    'rationale': 'High sepsis risk detected',
                    'evidence': predictions['sepsis_risk']
                },
                {
                    'type': 'laboratory',
                    'priority': 'urgent',
                    'action': 'Order lactate and blood cultures',
                    'rationale': 'Early detection of sepsis'
                }
            ])

        # Rule-based recommendations
        for rule_outcome in rule_outcomes.get('triggered_rules', []):
            if rule_outcome['severity'] in ['high', 'critical']:
                recommendations.append({
                    'type': 'clinical_rule',
                    'priority': rule_outcome['severity'],
                    'action': rule_outcome['recommendation'],
                    'rationale': rule_outcome['rule_name'],
                    'evidence': rule_outcome
                })

        return recommendations
```

## Clinical NLP Processing

```python
class ClinicalNLPProcessor:
    """Process clinical notes using NLP techniques"""

    def __init__(self):
        self.nlp = spacy.load("en_core_medical_lg")
        self.umls_connect = UMLSConnector()
        self.negex = Negex()

    def process_clinical_notes(self, notes: List[Dict]) -> Dict:
        """Process clinical notes and extract clinical entities"""

        processed_notes = []

        for note in notes:
            # Process note text
            doc = self.nlp(note['text'])

            # Extract entities
            entities = self.extract_clinical_entities(doc)

            # Detect negations
            entities = self.detect_negations(entities, note['text'])

            # Extract medications
            medications = self.extract_medications(doc)

            # Extract lab results
            lab_results = self.extract_lab_results(doc)

            # Extract symptoms
            symptoms = self.extract_symptoms(doc)

            # Extract procedures
            procedures = self.extract_procedures(doc)

            processed_note = {
                'note_id': note['id'],
                'note_type': note['type'],
                'date': note['date'],
                'entities': entities,
                'medications': medications,
                'lab_results': lab_results,
                'symptoms': symptoms,
                'procedures': procedures,
                'sentiment': self.analyze_sentiment(note['text'])
            }

            processed_notes.append(processed_note)

        return {
            'processed_notes': processed_notes,
            'summary': self.generate_clinical_summary(processed_notes)
        }

    def extract_clinical_entities(self, doc) -> List[Dict]:
        """Extract clinical entities from processed document"""

        entities = []

        for ent in doc.ents:
            if ent.label_ in ['DISEASE', 'SYMPTOM', 'MEDICATION', 'PROCEDURE', 'LAB']:
                entity = {
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': ent._.confidence if hasattr(ent._, 'confidence') else 0.0,
                    'umls_concept': self.umls_connect.get_concept(ent.text)
                }
                entities.append(entity)

        return entities
```

## EHR Data Integration

```python
class EHRDataIntegrator:
    """Integrate data from multiple EHR systems"""

    def __init__(self, data_sources: Dict):
        self.data_sources = data_sources
        self.fhir_client = FHIRClient()
        self.hl7_client = HL7Client()
        self.api_clients = self.initialize_api_clients()

    def get_patient_data(self, patient_id: str) -> Dict:
        """Get comprehensive patient data from all sources"""

        patient_data = {
            'patient_id': patient_id,
            'demographics': {},
            'clinical_notes': [],
            'lab_results': [],
            'medications': [],
            'diagnoses': [],
            'procedures': [],
            'vitals': [],
            'immunizations': [],
            'allergies': []
        }

        # Fetch from each data source
        for source_name, source_config in self.data_sources.items():
            try:
                source_data = self.fetch_from_source(source_name, patient_id, source_config)
                patient_data = self.merge_patient_data(patient_data, source_data)
            except Exception as e:
                logging.error(f"Error fetching from {source_name}: {str(e)}")

        return patient_data

    def fetch_from_epic(self, patient_id: str, config: Dict) -> Dict:
        """Fetch patient data from Epic EHR"""

        # Epic API integration
        epic_client = EpicClient(config['api_url'], config['api_key'])

        demographics = epic_client.get_patient_demographics(patient_id)
        clinical_notes = epic_client.get_clinical_notes(patient_id)
        lab_results = epic_client.get_lab_results(patient_id)
        medications = epic_client.get_medications(patient_id)

        return {
            'demographics': demographics,
            'clinical_notes': clinical_notes,
            'lab_results': lab_results,
            'medications': medications
        }

    def fetch_from_cerner(self, patient_id: str, config: Dict) -> Dict:
        """Fetch patient data from Cerner EHR"""

        # Cerner API integration
        cerner_client = CernerClient(config['api_url'], config['api_key'])

        diagnoses = cerner_client.get_diagnoses(patient_id)
        procedures = cerner_client.get_procedures(patient_id)
        vitals = cerner_client.get_vitals(patient_id)
        immunizations = cerner_client.get_immunizations(patient_id)

        return {
            'diagnoses': diagnoses,
            'procedures': procedures,
            'vitals': vitals,
            'immunizations': immunizations
        }
```

## Clinical Rules Engine

```python
class ClinicalRulesEngine:
    """Engine for applying clinical rules and guidelines"""

    def __init__(self):
        self.rules = self.load_clinical_rules()
        self.guidelines = self.load_clinical_guidelines()

    def apply_rules(self, structured_data: Dict, predictions: Dict) -> Dict:
        """Apply clinical rules to patient data"""

        triggered_rules = []
        alerts = []

        # Apply each rule
        for rule in self.rules:
            rule_result = self.evaluate_rule(rule, structured_data, predictions)
            if rule_result['triggered']:
                triggered_rules.append(rule_result)

                # Generate alert if high severity
                if rule_result['severity'] in ['high', 'critical']:
                    alerts.append({
                        'type': 'clinical_alert',
                        'rule_name': rule['name'],
                        'severity': rule_result['severity'],
                        'message': rule_result['message'],
                        'recommendation': rule_result['recommendation']
                    })

        return {
            'triggered_rules': triggered_rules,
            'alerts': alerts,
            'rule_count': len(triggered_rules)
        }

    def evaluate_rule(self, rule: Dict, structured_data: Dict,
                     predictions: Dict) -> Dict:
        """Evaluate a single clinical rule"""

        conditions_met = True

        # Check conditions
        for condition in rule['conditions']:
            if not self.evaluate_condition(condition, structured_data, predictions):
                conditions_met = False
                break

        return {
            'rule_name': rule['name'],
            'triggered': conditions_met,
            'severity': rule['severity'],
            'message': rule['message'] if conditions_met else None,
            'recommendation': rule['recommendation'] if conditions_met else None
        }
```

## Predictive Analytics

The EHR AI system includes several predictive models:

- **Readmission Risk Prediction**: Identifies patients at high risk of hospital readmission
- **Sepsis Early Detection**: Predicts sepsis onset 24-48 hours in advance
- **Disease Risk Assessment**: Evaluates risk for various chronic conditions
- **Clinical Deterioration**: Monitors patients for signs of clinical decline
- **Medication Adherence**: Predicts likelihood of medication non-adherence

## Real-world Implementation

```python
def implement_ehr_ai_system():
    """Example implementation for a healthcare system"""

    # Configuration
    config = {
        'data_sources': {
            'epic': {
                'connection_string': 'postgresql://user:pass@epic-db:5432/epic',
                'api_url': 'https://api.epic.com',
                'api_key': 'your_api_key'
            },
            'cerner': {
                'connection_string': 'oracle://user:pass@cerner-db:1521/cerner',
                'api_url': 'https://api.cerner.com',
                'api_key': 'your_api_key'
            },
            'mongodb': {
                'connection_string': 'mongodb://localhost:27017/ehr_data'
            }
        },
        'redis': {
            'host': 'localhost',
            'port': 6379
        },
        'models_path': '/models/ehr_ai'
    }

    # Initialize EHR AI system
    ehr_ai = EHRAIManager(config)

    # Example patient processing
    patient_id = "P123456789"

    try:
        # Process patient record
        patient_analysis = ehr_ai.process_patient_record(patient_id)

        # Generate clinical summary
        clinical_summary = ehr_ai.generate_clinical_summary(patient_analysis)

        # Create care plan recommendations
        care_plan = ehr_ai.generate_care_plan(patient_analysis)

        # Quality metrics
        quality_metrics = ehr_ai.quality_metrics.calculate_quality_metrics(
            patient_analysis
        )

        # Save results
        ehr_ai.save_patient_analysis(patient_id, {
            'analysis': patient_analysis,
            'summary': clinical_summary,
            'care_plan': care_plan,
            'quality_metrics': quality_metrics
        })

        print(f"Successfully processed patient {patient_id}")
        print(f"Generated {len(care_plan['recommendations'])} recommendations")
        print(f"Quality score: {quality_metrics['overall_score']}")

        return ehr_ai

    except Exception as e:
        print(f"Error processing patient: {str(e)}")
        return None
```

## Clinical Workflow Integration

```python
class ClinicalWorkflowIntegration:
    """Integrate AI insights into clinical workflows"""

    def __init__(self, ehr_ai: EHRAIManager):
        self.ehr_ai = ehr_ai
        self.notification_system = ClinicalNotificationSystem()
        self.cds_hooks = CDSHooksService()

    def integrate_with_epic(self):
        """Integrate with Epic EHR workflow"""

        # Set up CDS Hooks
        self.cds_hooks.register_service({
            'id': 'ehr-ai-predictions',
            'description': 'AI-powered clinical predictions',
            'hook': 'patient-view',
            'url': 'https://your-service.com/cds/predictions'
        })

        # Set up real-time alerts
        self.notification_system.set_up_alerts({
            'sepsis': 'urgent',
            'deterioration': 'high',
            'readmission': 'medium'
        })

    def integrate_with_cerner(self):
        """Integrate with Cerner EHR workflow"""

        # Set up PowerChart integration
        powerchart_integration = PowerChartIntegration()

        # Add AI insights to clinical viewer
        powerchart_integration.add_panel({
            'title': 'AI Clinical Insights',
            'content': 'ai_insights',
            'position': 'right'
        })
```

## Navigation

- **Next Module**: [03_Drug_Discovery_and_Development.md](03_Drug_Discovery_and_Development.md) - AI applications in pharmaceutical research
- **Previous Module**: [01_Medical_Imaging_and_Diagnostics.md](01_Medical_Imaging_and_Diagnostics.md) - Medical imaging AI systems
- **Related**: See implementation strategies in [05_Implementation_and_Integration.md](05_Implementation_and_Integration.md)

## Key Features Covered
- Multi-EHR system integration (Epic, Cerner)
- Clinical NLP and entity extraction
- Predictive analytics for various clinical scenarios
- Clinical rules engine and decision support
- Real-time alerting and notifications
- CDS Hooks integration
- Workflow automation and care coordination

---

*Module 2 of 5 in Healthcare AI Examples series*