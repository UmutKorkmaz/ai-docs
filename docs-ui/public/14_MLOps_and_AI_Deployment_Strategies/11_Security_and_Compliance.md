---
title: "Mlops And Ai Deployment Strategies - MLOps and AI"
description: "> Navigation: \u2190 Previous: Production Best Practices | Main Index | Next: Future Trends \u2192. Comprehensive guide covering classification, algorithms, machine le..."
keywords: "machine learning, classification, classification, algorithms, machine learning, artificial intelligence, machine learning, AI documentation"
author: "AI Documentation Team"
---

# MLOps and AI Deployment Strategies - Module 11: Security and Compliance

> **Navigation**: [← Previous: Production Best Practices](10_Production_Best_Practices.md) | [Main Index](README.md) | [Next: Future Trends →](12_Future_Trends.md)

## Security and Compliance in MLOps

Security and compliance are critical components of MLOps that ensure the protection of sensitive data, model integrity, and regulatory adherence. This module covers comprehensive security practices and compliance frameworks for machine learning systems.

### Module Overview
- **Core Security Principles**: Confidentiality, Integrity, Availability
- **Data Security**: Classification, encryption, access controls
- **Model Security**: Protection against adversarial attacks
- **Compliance Frameworks**: GDPR, HIPAA, SOC 2, ISO 27001
- **Audit Trails**: Comprehensive logging and monitoring
- **Risk Management**: Threat modeling and mitigation

## 11.1 Core Security Principles

### 11.1.1 CIA Triad for ML Systems

```python
# Security principles implementation
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib
import json
import logging
from datetime import datetime

class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

@dataclass
class SecurityContext:
    level: SecurityLevel
    encryption_required: bool
    audit_required: bool
    retention_policy: int
    compliance_frameworks: List[str]

class MLSecurityPrinciples:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.security_levels = {
            SecurityLevel.PUBLIC: SecurityContext(
                level=SecurityLevel.PUBLIC,
                encryption_required=False,
                audit_required=True,
                retention_policy=365,
                compliance_frameworks=[]
            ),
            SecurityLevel.INTERNAL: SecurityContext(
                level=SecurityLevel.INTERNAL,
                encryption_required=True,
                audit_required=True,
                retention_policy=730,
                compliance_frameworks=["ISO27001"]
            ),
            SecurityLevel.CONFIDENTIAL: SecurityContext(
                level=SecurityLevel.CONFIDENTIAL,
                encryption_required=True,
                audit_required=True,
                retention_policy=2555,
                compliance_frameworks=["GDPR", "ISO27001"]
            ),
            SecurityLevel.RESTRICTED: SecurityContext(
                level=SecurityLevel.RESTRICTED,
                encryption_required=True,
                audit_required=True,
                retention_policy=3650,
                compliance_frameworks=["HIPAA", "GDPR", "SOC2", "ISO27001"]
            )
        }

    def get_security_context(self, level: SecurityLevel) -> SecurityContext:
        """Get security context for data classification level"""
        return self.security_levels.get(level)

    def validate_security_controls(self, context: SecurityContext, data: Dict) -> bool:
        """Validate security controls are properly implemented"""
        validations = []

        # Check encryption requirements
        if context.encryption_required:
            validations.append(self._validate_encryption(data))

        # Check audit requirements
        if context.audit_required:
            validations.append(self._validate_audit_trail(data))

        return all(validations)

    def _validate_encryption(self, data: Dict) -> bool:
        """Validate data encryption"""
        # Implement encryption validation logic
        return True

    def _validate_audit_trail(self, data: Dict) -> bool:
        """Validate audit trail is maintained"""
        # Implement audit validation logic
        return True
```

### 11.1.2 Threat Modeling for ML Systems

```python
# Threat modeling implementation
from typing import Set, Tuple
import networkx as nx

class MLThreatModel:
    def __init__(self):
        self.threat_categories = {
            "data_poisoning": {
                "description": "Malicious data injection during training",
                "mitigation": ["data_validation", "adversarial_training"],
                "risk_level": "high"
            },
            "model_inversion": {
                "description": "Reconstructing training data from model outputs",
                "mitigation": ["differential_privacy", "output_perturbation"],
                "risk_level": "medium"
            },
            "membership_inference": {
                "description": "Determining if specific data was in training set",
                "mitigation": ["confidence_thresholds", "privacy_preserving"],
                "risk_level": "medium"
            },
            "model_stealing": {
                "description": "Extracting model parameters through queries",
                "mitigation": ["query_limits", "api_protection"],
                "risk_level": "high"
            },
            "evasion_attacks": {
                "description": "Crafting inputs to fool model predictions",
                "mitigation": ["adversarial_defenses", "robustness_training"],
                "risk_level": "high"
            }
        }

    def analyze_threats(self, model_info: Dict) -> Dict:
        """Analyze potential threats to ML system"""
        threats = {}

        for threat_type, threat_info in self.threat_categories.items():
            risk_score = self._calculate_risk_score(model_info, threat_type)
            threats[threat_type] = {
                **threat_info,
                "risk_score": risk_score,
                "priority": self._get_priority(risk_score)
            }

        return threats

    def _calculate_risk_score(self, model_info: Dict, threat_type: str) -> float:
        """Calculate risk score for specific threat"""
        # Implement risk calculation logic
        base_risk = {
            "data_poisoning": 0.8,
            "model_inversion": 0.6,
            "membership_inference": 0.5,
            "model_stealing": 0.7,
            "evasion_attacks": 0.9
        }

        return base_risk.get(threat_type, 0.5)

    def _get_priority(self, risk_score: float) -> str:
        """Get priority level based on risk score"""
        if risk_score >= 0.8:
            return "critical"
        elif risk_score >= 0.6:
            return "high"
        elif risk_score >= 0.4:
            return "medium"
        else:
            return "low"

    def generate_mitigation_plan(self, threats: Dict) -> Dict:
        """Generate comprehensive mitigation plan"""
        mitigation_plan = {}

        for threat_type, threat_info in threats.items():
            if threat_info["priority"] in ["critical", "high"]:
                mitigation_plan[threat_type] = {
                    "mitigations": threat_info["mitigation"],
                    "timeline": self._get_timeline(threat_info["priority"]),
                    "resources": self._estimate_resources(threat_info["risk_score"])
                }

        return mitigation_plan

    def _get_timeline(self, priority: str) -> str:
        """Get implementation timeline based on priority"""
        timelines = {
            "critical": "immediate",
            "high": "2-4 weeks",
            "medium": "1-3 months",
            "low": "3-6 months"
        }
        return timelines.get(priority, "3-6 months")

    def _estimate_resources(self, risk_score: float) -> Dict:
        """Estimate resources required for mitigation"""
        if risk_score >= 0.8:
            return {"engineers": 3, "weeks": 4, "budget": "high"}
        elif risk_score >= 0.6:
            return {"engineers": 2, "weeks": 3, "budget": "medium"}
        else:
            return {"engineers": 1, "weeks": 2, "budget": "low"}
```

## 11.2 Data Security

### 11.2.1 Data Classification and Handling

```python
# Data classification system
from typing import Any
import pandas as pd
import numpy as np

class DataClassifier:
    def __init__(self):
        self.classification_rules = {
            "personal_identifiable": {
                "patterns": ["ssn", "email", "phone", "address"],
                "level": SecurityLevel.CONFIDENTIAL
            },
            "health_data": {
                "patterns": ["diagnosis", "treatment", "medical_record"],
                "level": SecurityLevel.RESTRICTED
            },
            "financial_data": {
                "patterns": ["credit_card", "bank_account", "salary"],
                "level": SecurityLevel.RESTRICTED
            },
            "business_confidential": {
                "patterns": ["revenue", "strategy", "proprietary"],
                "level": SecurityLevel.CONFIDENTIAL
            }
        }

    def classify_data(self, data: pd.DataFrame) -> Dict[str, SecurityLevel]:
        """Classify data columns based on content"""
        classifications = {}

        for column in data.columns:
            level = self._classify_column(data[column], column)
            classifications[column] = level

        return classifications

    def _classify_column(self, column: pd.Series, column_name: str) -> SecurityLevel:
        """Classify individual column"""
        # Check column name patterns
        for category, rule in self.classification_rules.items():
            if any(pattern in column_name.lower() for pattern in rule["patterns"]):
                return rule["level"]

        # Check content patterns
        sample_data = column.dropna().head(100)
        if self._contains_pii(sample_data):
            return SecurityLevel.CONFIDENTIAL

        return SecurityLevel.INTERNAL

    def _contains_pii(self, data: pd.Series) -> bool:
        """Check if data contains PII"""
        # Implement PII detection logic
        return False

    def generate_handling_procedures(self, classifications: Dict[str, SecurityLevel]) -> Dict:
        """Generate data handling procedures"""
        procedures = {}

        for column, level in classifications.items():
            context = MLSecurityPrinciples().get_security_context(level)
            procedures[column] = {
                "security_level": level.value,
                "encryption": context.encryption_required,
                "audit": context.audit_required,
                "retention": context.retention_policy,
                "compliance": context.compliance_frameworks
            }

        return procedures
```

### 11.2.2 Data Encryption and Protection

```python
# Data encryption implementation
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os

class DataEncryption:
    def __init__(self):
        self.key = self._generate_key()
        self.cipher_suite = Fernet(self.key)

    def _generate_key(self) -> bytes:
        """Generate encryption key"""
        return Fernet.generate_key()

    def encrypt_data(self, data: Any) -> Dict:
        """Encrypt sensitive data"""
        try:
            if isinstance(data, (pd.DataFrame, pd.Series)):
                return self._encrypt_dataframe(data)
            elif isinstance(data, dict):
                return self._encrypt_dict(data)
            elif isinstance(data, (list, tuple)):
                return self._encrypt_sequence(data)
            else:
                return self._encrypt_string(str(data))
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: Dict) -> Any:
        """Decrypt encrypted data"""
        try:
            if encrypted_data["type"] == "dataframe":
                return self._decrypt_dataframe(encrypted_data)
            elif encrypted_data["type"] == "dict":
                return self._decrypt_dict(encrypted_data)
            elif encrypted_data["type"] == "sequence":
                return self._decrypt_sequence(encrypted_data)
            else:
                return self._decrypt_string(encrypted_data)
        except Exception as e:
            self.logger.error(f"Decryption failed: {e}")
            raise

    def _encrypt_dataframe(self, df: pd.DataFrame) -> Dict:
        """Encrypt DataFrame"""
        data_json = df.to_json()
        encrypted_data = self.cipher_suite.encrypt(data_json.encode())

        return {
            "type": "dataframe",
            "data": base64.b64encode(encrypted_data).decode(),
            "shape": df.shape,
            "columns": df.columns.tolist()
        }

    def _encrypt_dict(self, data: Dict) -> Dict:
        """Encrypt dictionary"""
        data_json = json.dumps(data)
        encrypted_data = self.cipher_suite.encrypt(data_json.encode())

        return {
            "type": "dict",
            "data": base64.b64encode(encrypted_data).decode(),
            "keys": list(data.keys())
        }

    def _encrypt_sequence(self, data: Any) -> Dict:
        """Encrypt sequence (list/tuple)"""
        data_json = json.dumps(data)
        encrypted_data = self.cipher_suite.encrypt(data_json.encode())

        return {
            "type": "sequence",
            "data": base64.b64encode(encrypted_data).decode(),
            "length": len(data)
        }

    def _encrypt_string(self, data: str) -> Dict:
        """Encrypt string"""
        encrypted_data = self.cipher_suite.encrypt(data.encode())

        return {
            "type": "string",
            "data": base64.b64encode(encrypted_data).decode()
        }

    def _decrypt_dataframe(self, encrypted_data: Dict) -> pd.DataFrame:
        """Decrypt DataFrame"""
        encrypted_bytes = base64.b64decode(encrypted_data["data"].encode())
        decrypted_json = self.cipher_suite.decrypt(encrypted_bytes).decode()
        return pd.read_json(decrypted_json)

    def _decrypt_dict(self, encrypted_data: Dict) -> Dict:
        """Decrypt dictionary"""
        encrypted_bytes = base64.b64decode(encrypted_data["data"].encode())
        decrypted_json = self.cipher_suite.decrypt(encrypted_bytes).decode()
        return json.loads(decrypted_json)

    def _decrypt_sequence(self, encrypted_data: Dict) -> Any:
        """Decrypt sequence"""
        encrypted_bytes = base64.b64decode(encrypted_data["data"].encode())
        decrypted_json = self.cipher_suite.decrypt(encrypted_bytes).decode()
        return json.loads(decrypted_json)

    def _decrypt_string(self, encrypted_data: Dict) -> str:
        """Decrypt string"""
        encrypted_bytes = base64.b64decode(encrypted_data["data"].encode())
        return self.cipher_suite.decrypt(encrypted_bytes).decode()
```

## 11.3 Model Security

### 11.3.1 Adversarial Attack Defense

```python
# Adversarial defense mechanisms
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple

class AdversarialDefense:
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def adversarial_training(self,
                           train_loader,
                           epsilon: float = 0.1,
                           alpha: float = 0.01,
                           num_iter: int = 10):
        """Perform adversarial training on model"""
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_iter):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                # Generate adversarial examples
                adv_data = self._generate_adversarial_examples(
                    data, target, epsilon, alpha
                )

                # Train on both clean and adversarial examples
                optimizer.zero_grad()

                # Clean loss
                output_clean = self.model(data)
                loss_clean = criterion(output_clean, target)

                # Adversarial loss
                output_adv = self.model(adv_data)
                loss_adv = criterion(output_adv, target)

                # Combined loss
                total_loss = 0.5 * (loss_clean + loss_adv)
                total_loss.backward()
                optimizer.step()

    def _generate_adversarial_examples(self,
                                     data: torch.Tensor,
                                     target: torch.Tensor,
                                     epsilon: float,
                                     alpha: float) -> torch.Tensor:
        """Generate adversarial examples using PGD attack"""
        adv_data = data.clone().detach().requires_grad_(True)

        for _ in range(10):  # Number of attack iterations
            output = self.model(adv_data)
            loss = nn.CrossEntropyLoss()(output, target)

            grad = torch.autograd.grad(loss, adv_data)[0]
            adv_data = adv_data + alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(adv_data - data, -epsilon, epsilon)
            adv_data = data + delta

            # Clamp to valid range
            adv_data = torch.clamp(adv_data, 0, 1)

        return adv_data.detach()

    def detect_adversarial_examples(self,
                                  data: torch.Tensor,
                                  threshold: float = 0.1) -> List[bool]:
        """Detect adversarial examples"""
        detections = []

        for sample in data:
            # Use multiple detection methods
            detection_scores = []

            # Method 1: Local Intrinsic Dimensionality
            lid_score = self._compute_lid(sample)
            detection_scores.append(lid_score)

            # Method 2: Mahalanobis Distance
            mahalanobis_score = self._compute_mahalanobis_distance(sample)
            detection_scores.append(mahalanobis_score)

            # Method 3: Feature Squeezing
            squeezing_score = self._compute_squeezing_score(sample)
            detection_scores.append(squeezing_score)

            # Combine scores
            combined_score = np.mean(detection_scores)
            detections.append(combined_score > threshold)

        return detections

    def _compute_lid(self, sample: torch.Tensor) -> float:
        """Compute Local Intrinsic Dimensionality"""
        # Simplified LID computation
        with torch.no_grad():
            output = self.model(sample.unsqueeze(0))
            # Implement LID computation
            return 0.1  # Placeholder

    def _compute_mahalanobis_distance(self, sample: torch.Tensor) -> float:
        """Compute Mahalanobis distance"""
        # Simplified Mahalanobis distance
        with torch.no_grad():
            output = self.model(sample.unsqueeze(0))
            # Implement Mahalanobis distance
            return 0.2  # Placeholder

    def _compute_squeezing_score(self, sample: torch.Tensor) -> float:
        """Compute feature squeezing score"""
        with torch.no_grad():
            output_original = self.model(sample.unsqueeze(0))

            # Apply bit reduction
            squeezed_sample = torch.floor(sample * 16) / 16
            output_squeezed = self.model(squeezed_sample.unsqueeze(0))

            # Compute difference
            diff = torch.norm(output_original - output_squeezed)
            return diff.item()

    def apply_defenses(self, data: torch.Tensor) -> torch.Tensor:
        """Apply defensive preprocessing"""
        # Apply multiple defensive transformations
        defended_data = data.clone()

        # 1. Gaussian noise
        defended_data += torch.randn_like(defended_data) * 0.01

        # 2. Random rotation
        if len(defended_data.shape) == 3:  # Image data
            angle = torch.rand(1) * 30 - 15  # Random rotation -15 to 15 degrees
            defended_data = self._rotate_image(defended_data, angle)

        # 3. Color jitter
        if len(defended_data.shape) == 3 and defended_data.shape[0] == 3:  # RGB image
            defended_data = self._color_jitter(defended_data)

        return torch.clamp(defended_data, 0, 1)

    def _rotate_image(self, image: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        """Rotate image by given angle"""
        # Simplified rotation (would need proper implementation)
        return image

    def _color_jitter(self, image: torch.Tensor) -> torch.Tensor:
        """Apply color jittering"""
        # Simplified color jittering
        brightness = 0.1 * torch.rand(1)
        contrast = 0.1 * torch.rand(1)
        saturation = 0.1 * torch.rand(1)

        image = image * (1 + brightness)
        image = torch.clamp(image, 0, 1)

        return image
```

### 11.3.2 Model Robustness Testing

```python
# Model robustness testing framework
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

class ModelRobustnessTester:
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        self.results = {}

    def comprehensive_robustness_test(self) -> Dict:
        """Run comprehensive robustness testing"""
        results = {
            "baseline_performance": self._baseline_test(),
            "adversarial_robustness": self._adversarial_test(),
            "noise_robustness": self._noise_test(),
            "distribution_shift": self._distribution_shift_test(),
            "input_variations": self._input_variation_test(),
            "overall_robustness_score": None
        }

        # Calculate overall robustness score
        results["overall_robustness_score"] = self._calculate_overall_score(results)

        return results

    def _baseline_test(self) -> Dict:
        """Test baseline performance on clean data"""
        X_test, y_test = self.test_data

        with torch.no_grad():
            predictions = self.model(X_test)
            predicted_labels = torch.argmax(predictions, dim=1)

        accuracy = accuracy_score(y_test, predicted_labels)
        precision = precision_score(y_test, predicted_labels, average='weighted')
        recall = recall_score(y_test, predicted_labels, average='weighted')

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * (precision * recall) / (precision + recall)
        }

    def _adversarial_test(self) -> Dict:
        """Test against adversarial attacks"""
        defense = AdversarialDefense(self.model)
        X_test, y_test = self.test_data

        # Test against different attack types
        attack_results = {}

        # FGSM attack
        fgsm_accuracy = self._test_fgsm_attack(X_test, y_test)
        attack_results["fgsm"] = fgsm_accuracy

        # PGD attack
        pgd_accuracy = self._test_pgd_attack(X_test, y_test)
        attack_results["pgd"] = pgd_accuracy

        # Detection rate
        detections = defense.detect_adversarial_examples(X_test)
        detection_rate = np.mean(detections)

        return {
            "attack_accuracies": attack_results,
            "detection_rate": detection_rate,
            "robustness_score": np.mean(list(attack_results.values()))
        }

    def _test_fgsm_attack(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Test against FGSM attack"""
        epsilon = 0.1
        X_adv = self._fgsm_attack(X_test, y_test, epsilon)

        with torch.no_grad():
            predictions = self.model(X_adv)
            predicted_labels = torch.argmax(predictions, dim=1)

        return accuracy_score(y_test, predicted_labels)

    def _fgsm_attack(self, X: torch.Tensor, y: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Fast Gradient Sign Method attack"""
        X.requires_grad = True

        output = self.model(X)
        loss = nn.CrossEntropyLoss()(output, y)

        grad = torch.autograd.grad(loss, X)[0]
        X_adv = X + epsilon * grad.sign()

        return torch.clamp(X_adv, 0, 1)

    def _test_pgd_attack(self, X_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Test against PGD attack"""
        epsilon = 0.1
        alpha = 0.01
        num_iter = 10

        X_adv = self._pgd_attack(X_test, y_test, epsilon, alpha, num_iter)

        with torch.no_grad():
            predictions = self.model(X_adv)
            predicted_labels = torch.argmax(predictions, dim=1)

        return accuracy_score(y_test, predicted_labels)

    def _pgd_attack(self, X: torch.Tensor, y: torch.Tensor,
                   epsilon: float, alpha: float, num_iter: int) -> torch.Tensor:
        """Projected Gradient Descent attack"""
        X_adv = X.clone().detach()

        for _ in range(num_iter):
            X_adv.requires_grad = True

            output = self.model(X_adv)
            loss = nn.CrossEntropyLoss()(output, y)

            grad = torch.autograd.grad(loss, X_adv)[0]
            X_adv = X_adv + alpha * grad.sign()

            # Project back to epsilon ball
            delta = torch.clamp(X_adv - X, -epsilon, epsilon)
            X_adv = X + delta

            X_adv = torch.clamp(X_adv, 0, 1).detach()

        return X_adv

    def _noise_test(self) -> Dict:
        """Test robustness against noise"""
        X_test, y_test = self.test_data

        noise_levels = [0.01, 0.05, 0.1, 0.2]
        noise_results = {}

        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = torch.randn_like(X_test) * noise_level
            X_noisy = torch.clamp(X_test + noise, 0, 1)

            with torch.no_grad():
                predictions = self.model(X_noisy)
                predicted_labels = torch.argmax(predictions, dim=1)

            accuracy = accuracy_score(y_test, predicted_labels)
            noise_results[noise_level] = accuracy

        return noise_results

    def _distribution_shift_test(self) -> Dict:
        """Test robustness to distribution shift"""
        X_test, y_test = self.test_data

        # Simulate distribution shift
        shift_results = {}

        # Brightness shift
        X_bright = torch.clamp(X_test * 1.2, 0, 1)
        bright_accuracy = self._evaluate_shifted_data(X_bright, y_test)
        shift_results["brightness"] = bright_accuracy

        # Contrast shift
        X_contrast = torch.clamp((X_test - 0.5) * 1.5 + 0.5, 0, 1)
        contrast_accuracy = self._evaluate_shifted_data(X_contrast, y_test)
        shift_results["contrast"] = contrast_accuracy

        # Blur shift
        X_blur = self._apply_blur(X_test)
        blur_accuracy = self._evaluate_shifted_data(X_blur, y_test)
        shift_results["blur"] = blur_accuracy

        return shift_results

    def _input_variation_test(self) -> Dict:
        """Test robustness to input variations"""
        X_test, y_test = self.test_data

        variation_results = {}

        # Rotation test
        rotation_accuracies = []
        for angle in [15, 30, 45]:
            X_rotated = self._rotate_batch(X_test, angle)
            accuracy = self._evaluate_shifted_data(X_rotated, y_test)
            rotation_accuracies.append(accuracy)
        variation_results["rotation"] = np.mean(rotation_accuracies)

        # Scaling test
        scaling_accuracies = []
        for scale in [0.8, 1.2, 1.5]:
            X_scaled = self._scale_batch(X_test, scale)
            accuracy = self._evaluate_shifted_data(X_scaled, y_test)
            scaling_accuracies.append(accuracy)
        variation_results["scaling"] = np.mean(scaling_accuracies)

        return variation_results

    def _evaluate_shifted_data(self, X_shifted: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluate model on shifted data"""
        with torch.no_grad():
            predictions = self.model(X_shifted)
            predicted_labels = torch.argmax(predictions, dim=1)

        return accuracy_score(y_test, predicted_labels)

    def _apply_blur(self, X: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian blur to images"""
        # Simplified blur implementation
        return X  # Placeholder

    def _rotate_batch(self, X: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate batch of images"""
        # Simplified rotation implementation
        return X  # Placeholder

    def _scale_batch(self, X: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale batch of images"""
        # Simplified scaling implementation
        return X  # Placeholder

    def _calculate_overall_score(self, results: Dict) -> float:
        """Calculate overall robustness score"""
        # Weight different components
        weights = {
            "baseline_performance": 0.2,
            "adversarial_robustness": 0.3,
            "noise_robustness": 0.2,
            "distribution_shift": 0.15,
            "input_variations": 0.15
        }

        # Normalize scores to 0-1 range
        baseline_score = results["baseline_performance"]["accuracy"]
        adversarial_score = results["adversarial_robustness"]["robustness_score"]
        noise_score = np.mean(list(results["noise_robustness"].values()))
        shift_score = np.mean(list(results["distribution_shift"].values()))
        variation_score = np.mean(list(results["input_variation"].values()))

        overall_score = (
            weights["baseline_performance"] * baseline_score +
            weights["adversarial_robustness"] * adversarial_score +
            weights["noise_robustness"] * noise_score +
            weights["distribution_shift"] * shift_score +
            weights["input_variations"] * variation_score
        )

        return overall_score

    def generate_robustness_report(self, results: Dict) -> str:
        """Generate comprehensive robustness report"""
        report = f"""
# Model Robustness Testing Report

## Overall Robustness Score: {results['overall_robustness_score']:.2f}

### 1. Baseline Performance
- Accuracy: {results['baseline_performance']['accuracy']:.2f}
- Precision: {results['baseline_performance']['precision']:.2f}
- Recall: {results['baseline_performance']['recall']:.2f}
- F1 Score: {results['baseline_performance']['f1_score']:.2f}

### 2. Adversarial Robustness
- FGSM Attack Accuracy: {results['adversarial_robustness']['attack_accuracies']['fgsm']:.2f}
- PGD Attack Accuracy: {results['adversarial_robustness']['attack_accuracies']['pgd']:.2f}
- Detection Rate: {results['adversarial_robustness']['detection_rate']:.2f}
- Robustness Score: {results['adversarial_robustness']['robustness_score']:.2f}

### 3. Noise Robustness
{self._format_noise_results(results['noise_robustness'])}

### 4. Distribution Shift Robustness
{self._format_shift_results(results['distribution_shift'])}

### 5. Input Variation Robustness
{self._format_variation_results(results['input_variation'])}

## Recommendations
{self._generate_recommendations(results)}
"""
        return report

    def _format_noise_results(self, noise_results: Dict) -> str:
        """Format noise test results"""
        formatted = []
        for noise_level, accuracy in noise_results.items():
            formatted.append(f"- Noise Level {noise_level}: {accuracy:.2f}")
        return "\n".join(formatted)

    def _format_shift_results(self, shift_results: Dict) -> str:
        """Format distribution shift results"""
        formatted = []
        for shift_type, accuracy in shift_results.items():
            formatted.append(f"- {shift_type.capitalize()} Shift: {accuracy:.2f}")
        return "\n".join(formatted)

    def _format_variation_results(self, variation_results: Dict) -> str:
        """Format input variation results"""
        formatted = []
        for variation_type, accuracy in variation_results.items():
            formatted.append(f"- {variation_type.capitalize()}: {accuracy:.2f}")
        return "\n".join(formatted)

    def _generate_recommendations(self, results: Dict) -> str:
        """Generate improvement recommendations"""
        recommendations = []

        if results['adversarial_robustness']['robustness_score'] < 0.7:
            recommendations.append("- Implement adversarial training")

        if np.mean(list(results['noise_robustness'].values())) < 0.8:
            recommendations.append("- Add noise augmentation to training data")

        if np.mean(list(results['distribution_shift'].values())) < 0.75:
            recommendations.append("- Apply data augmentation techniques")

        if results['overall_robustness_score'] < 0.7:
            recommendations.append("- Consider defensive preprocessing techniques")

        return "\n".join(recommendations)
```

## 11.4 Compliance Frameworks

### 11.4.1 GDPR Compliance

```python
# GDPR compliance implementation
from datetime import datetime, timedelta
import hashlib

class GDPRCompliance:
    def __init__(self):
        self.data_subjects = {}
        self.consent_records = {}
        self.data_processing_activities = []

    def register_data_subject(self, subject_id: str, personal_data: Dict) -> Dict:
        """Register data subject with their personal data"""
        consent_record = {
            "subject_id": subject_id,
            "data_collected": list(personal_data.keys()),
            "consent_given": datetime.now(),
            "consent_purpose": "model_training",
            "retention_period": timedelta(days=365),
            "data_hashes": self._hash_personal_data(personal_data)
        }

        self.consent_records[subject_id] = consent_record
        self.data_subjects[subject_id] = personal_data

        return consent_record

    def _hash_personal_data(self, data: Dict) -> Dict:
        """Create hashes of personal data for verification"""
        hashes = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float)):
                hashes[key] = hashlib.sha256(str(value).encode()).hexdigest()
        return hashes

    def request_data_access(self, subject_id: str) -> Dict:
        """Handle data subject access request"""
        if subject_id not in self.data_subjects:
            raise ValueError("Data subject not found")

        return {
            "subject_id": subject_id,
            "personal_data": self.data_subjects[subject_id],
            "processing_activities": self._get_subject_activities(subject_id),
            "data_retention": self.consent_records[subject_id]["retention_period"],
            "consent_details": self.consent_records[subject_id]
        }

    def request_data_deletion(self, subject_id: str) -> bool:
        """Handle right to be forgotten request"""
        if subject_id not in self.data_subjects:
            return False

        # Remove personal data
        del self.data_subjects[subject_id]

        # Remove consent records
        if subject_id in self.consent_records:
            del self.consent_records[subject_id]

        # Log deletion request
        self._log_deletion_request(subject_id)

        return True

    def _log_deletion_request(self, subject_id: str):
        """Log data deletion request"""
        log_entry = {
            "timestamp": datetime.now(),
            "action": "data_deletion",
            "subject_id": subject_id,
            "compliance": "GDPR"
        }
        self.data_processing_activities.append(log_entry)

    def _get_subject_activities(self, subject_id: str) -> List[Dict]:
        """Get processing activities for specific subject"""
        return [
            activity for activity in self.data_processing_activities
            if activity.get("subject_id") == subject_id
        ]

    def anonymize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Anonymize personal data according to GDPR standards"""
        anonymized_data = data.copy()

        # Apply anonymization techniques
        for column in anonymized_data.columns:
            if self._is_personal_data(column):
                anonymized_data[column] = self._anonymize_column(
                    anonymized_data[column]
                )

        return anonymized_data

    def _is_personal_data(self, column_name: str) -> bool:
        """Check if column contains personal data"""
        personal_indicators = ["name", "email", "phone", "address", "id"]
        return any(indicator in column_name.lower() for indicator in personal_indicators)

    def _anonymize_column(self, column: pd.Series) -> pd.Series:
        """Anonymize individual column"""
        # Apply k-anonymity
        return column.apply(lambda x: f"ANON_{hash(str(x)) % 10000}")

    def generate_compliance_report(self) -> Dict:
        """Generate GDPR compliance report"""
        return {
            "total_data_subjects": len(self.data_subjects),
            "active_consents": len(self.consent_records),
            "processing_activities": len(self.data_processing_activities),
            "data_retention_compliance": self._check_retention_compliance(),
            "data_protection_measures": self._list_protection_measures(),
            "data_breach_incidents": self._get_breach_incidents()
        }

    def _check_retention_compliance(self) -> Dict:
        """Check data retention compliance"""
        compliance_status = {}
        current_date = datetime.now()

        for subject_id, record in self.consent_records.items():
            retention_end = record["consent_given"] + record["retention_period"]
            is_compliant = current_date <= retention_end
            compliance_status[subject_id] = {
                "compliant": is_compliant,
                "retention_end": retention_end,
                "days_remaining": (retention_end - current_date).days
            }

        return compliance_status

    def _list_protection_measures(self) -> List[str]:
        """List data protection measures implemented"""
        return [
            "Data encryption at rest and in transit",
            "Access control and authentication",
            "Regular security audits",
            "Data anonymization techniques",
            "Consent management system",
            "Data retention policies"
        ]

    def _get_breach_incidents(self) -> List[Dict]:
        """Get data breach incidents (mock implementation)"""
        return []
```

### 11.4.2 HIPAA Compliance

```python
# HIPAA compliance implementation
class HIPAACompliance:
    def __init__(self):
        self.phi_records = {}
        self.access_logs = []
        self.breach_notifications = []

    def classify_phi(self, data: Dict) -> Dict:
        """Classify Protected Health Information"""
        phi_categories = {
            "demographic": ["name", "address", "date_of_birth"],
            "medical": ["diagnosis", "treatment", "medications"],
            "financial": ["insurance", "billing"],
            "identifiers": ["ssn", "medical_record_number"]
        }

        classified_phi = {}
        for category, fields in phi_categories.items():
            phi_fields = {field: data.get(field) for field in fields if field in data}
            if phi_fields:
                classified_phi[category] = phi_fields

        return classified_phi

    def implement_safeguards(self) -> Dict:
        """Implement HIPAA security safeguards"""
        safeguards = {
            "administrative": [
                "Risk analysis and management",
                "Security management process",
                "Workforce security training",
                "Information access management",
                "Contingency planning"
            ],
            "physical": [
                "Facility access controls",
                "Workstation security",
                "Device and media controls",
                "Physical security monitoring"
            ],
            "technical": [
                "Access control",
                "Audit controls",
                "Integrity controls",
                "Transmission security"
            ]
        }

        return safeguards

    def conduct_risk_analysis(self) -> Dict:
        """Conduct HIPAA risk analysis"""
        risk_areas = [
            "Unauthorized access to PHI",
            "Data breaches",
            "Inadequate employee training",
            "Insufficient backup systems",
            "Weak access controls"
        ]

        risk_assessment = {}
        for area in risk_areas:
            risk_score = self._assess_risk(area)
            mitigation = self._suggest_mitigation(area, risk_score)

            risk_assessment[area] = {
                "risk_score": risk_score,
                "mitigation": mitigation,
                "priority": self._get_priority(risk_score)
            }

        return risk_assessment

    def _assess_risk(self, area: str) -> float:
        """Assess risk level for specific area"""
        # Simplified risk assessment
        risk_mapping = {
            "Unauthorized access to PHI": 0.8,
            "Data breaches": 0.9,
            "Inadequate employee training": 0.6,
            "Insufficient backup systems": 0.7,
            "Weak access controls": 0.8
        }
        return risk_mapping.get(area, 0.5)

    def _suggest_mitigation(self, area: str, risk_score: float) -> List[str]:
        """Suggest mitigation strategies"""
        mitigation_strategies = {
            "Unauthorized access to PHI": [
                "Implement multi-factor authentication",
                "Enhance encryption protocols",
                "Regular access reviews"
            ],
            "Data breaches": [
                "Implement intrusion detection systems",
                "Regular security audits",
                "Employee security training"
            ],
            "Inadequate employee training": [
                "Mandatory HIPAA training",
                "Regular security awareness programs",
                "Simulated phishing tests"
            ]
        }
        return mitigation_strategies.get(area, ["Review security measures"])

    def _get_priority(self, risk_score: float) -> str:
        """Get priority level based on risk score"""
        if risk_score >= 0.8:
            return "high"
        elif risk_score >= 0.6:
            return "medium"
        else:
            return "low"

    def log_phi_access(self, user_id: str, phi_id: str, purpose: str):
        """Log access to Protected Health Information"""
        access_log = {
            "timestamp": datetime.now(),
            "user_id": user_id,
            "phi_id": phi_id,
            "purpose": purpose,
            "action": "access",
            "compliance": "HIPAA"
        }
        self.access_logs.append(access_log)

    def report_breach(self, breach_info: Dict) -> Dict:
        """Report data breach according to HIPAA requirements"""
        breach_report = {
            "breach_id": f"BREACH_{len(self.breach_notifications) + 1}",
            "discovery_date": breach_info.get("discovery_date", datetime.now()),
            "breach_type": breach_info.get("breach_type", "unauthorized_access"),
            "affected_individuals": breach_info.get("affected_count", 0),
            "phi_types": breach_info.get("phi_types", []),
            "notification_required": self._check_notification_requirement(breach_info),
            "remediation_steps": breach_info.get("remediation_steps", [])
        }

        self.breach_notifications.append(breach_report)
        return breach_report

    def _check_notification_requirement(self, breach_info: Dict) -> bool:
        """Check if breach notification is required"""
        affected_count = breach_info.get("affected_count", 0)
        return affected_count >= 500  # HIPAA threshold
```

## 11.5 Audit Trails and Logging

### 11.5.1 Comprehensive Audit System

```python
# Audit trail implementation
import json
import threading
from queue import Queue
from typing import Callable, Any

class AuditTrail:
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = log_file
        self.audit_queue = Queue()
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()

        # Audit event types
        self.event_types = {
            "model_access": "high",
            "data_access": "high",
            "model_deployment": "high",
            "model_update": "medium",
            "user_login": "medium",
            "configuration_change": "high",
            "error_event": "medium",
            "security_alert": "critical"
        }

    def start_audit_thread(self):
        """Start background audit thread"""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._audit_worker)
            self.thread.daemon = True
            self.thread.start()

    def stop_audit_thread(self):
        """Stop background audit thread"""
        self.is_running = False
        if self.thread:
            self.thread.join()

    def log_event(self, event_type: str, user_id: str, details: Dict, severity: str = None):
        """Log audit event"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "severity": severity or self.event_types.get(event_type, "medium"),
            "details": details,
            "ip_address": self._get_client_ip(),
            "user_agent": self._get_user_agent()
        }

        # Add to queue for async processing
        self.audit_queue.put(event)

        # Also log to console for immediate visibility
        if severity in ["critical", "high"]:
            self._log_to_console(event)

    def _audit_worker(self):
        """Background worker for processing audit events"""
        while self.is_running:
            try:
                event = self.audit_queue.get(timeout=1)
                self._write_audit_log(event)
                self.audit_queue.task_done()
            except:
                continue

    def _write_audit_log(self, event: Dict):
        """Write audit event to log file"""
        with self.lock:
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(event) + '\n')
            except Exception as e:
                print(f"Failed to write audit log: {e}")

    def _log_to_console(self, event: Dict):
        """Log critical events to console"""
        print(f"[AUDIT] {event['timestamp']} - {event['event_type']} - User: {event['user_id']}")

    def _get_client_ip(self) -> str:
        """Get client IP address"""
        # Mock implementation
        return "127.0.0.1"

    def _get_user_agent(self) -> str:
        """Get user agent string"""
        # Mock implementation
        return "AuditSystem/1.0"

    def query_audit_logs(self,
                        start_time: datetime = None,
                        end_time: datetime = None,
                        event_type: str = None,
                        user_id: str = None,
                        limit: int = 100) -> List[Dict]:
        """Query audit logs"""
        logs = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())

                        # Apply filters
                        if start_time and datetime.fromisoformat(event['timestamp']) < start_time:
                            continue
                        if end_time and datetime.fromisoformat(event['timestamp']) > end_time:
                            continue
                        if event_type and event['event_type'] != event_type:
                            continue
                        if user_id and event['user_id'] != user_id:
                            continue

                        logs.append(event)

                        if len(logs) >= limit:
                            break
                    except:
                        continue
        except FileNotFoundError:
            pass

        return logs

    def generate_audit_report(self,
                            start_date: datetime = None,
                            end_date: datetime = None) -> Dict:
        """Generate comprehensive audit report"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        if not end_date:
            end_date = datetime.now()

        logs = self.query_audit_logs(start_date, end_date)

        # Analyze logs
        event_counts = {}
        user_activity = {}
        severity_counts = {}

        for log in logs:
            # Count by event type
            event_type = log['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # Count by user
            user_id = log['user_id']
            user_activity[user_id] = user_activity.get(user_id, 0) + 1

            # Count by severity
            severity = log['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_events": len(logs),
            "event_distribution": event_counts,
            "user_activity": user_activity,
            "severity_distribution": severity_counts,
            "security_alerts": len([l for l in logs if l['severity'] == 'critical']),
            "recommendations": self._generate_audit_recommendations(logs)
        }

    def _generate_audit_recommendations(self, logs: List[Dict]) -> List[str]:
        """Generate recommendations based on audit logs"""
        recommendations = []

        # Check for unusual patterns
        event_counts = {}
        for log in logs:
            event_type = log['event_type']
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # High number of errors
        error_count = event_counts.get('error_event', 0)
        if error_count > len(logs) * 0.1:
            recommendations.append("High error rate detected - investigate system stability")

        # Security alerts
        security_alerts = len([l for l in logs if l['severity'] == 'critical'])
        if security_alerts > 0:
            recommendations.append(f"Security alerts detected: {security_alerts} - immediate attention required")

        # Unusual user activity
        user_activity = {}
        for log in logs:
            user_id = log['user_id']
            user_activity[user_id] = user_activity.get(user_id, 0) + 1

        avg_activity = np.mean(list(user_activity.values()))
        for user_id, activity in user_activity.items():
            if activity > avg_activity * 3:
                recommendations.append(f"Unusual activity detected for user {user_id}")

        return recommendations
```

### 11.5.2 Compliance Monitoring

```python
# Compliance monitoring system
from typing import Dict, List, Set
import asyncio

class ComplianceMonitor:
    def __init__(self):
        self.compliance_rules = {}
        self.violations = []
        self.monitoring_active = False
        self.audit_trail = AuditTrail()

        self._initialize_compliance_rules()

    def _initialize_compliance_rules(self):
        """Initialize compliance monitoring rules"""
        self.compliance_rules = {
            "data_retention": {
                "description": "Data retention policy compliance",
                "check_function": self._check_data_retention,
                "severity": "high",
                "frequency": "daily"
            },
            "access_control": {
                "description": "Access control compliance",
                "check_function": self._check_access_control,
                "severity": "critical",
                "frequency": "hourly"
            },
            "encryption": {
                "description": "Data encryption compliance",
                "check_function": self._check_encryption,
                "severity": "high",
                "frequency": "daily"
            },
            "audit_logging": {
                "description": "Audit logging compliance",
                "check_function": self._check_audit_logging,
                "severity": "medium",
                "frequency": "hourly"
            },
            "consent_management": {
                "description": "Consent management compliance",
                "check_function": self._check_consent_management,
                "severity": "high",
                "frequency": "daily"
            }
        }

    async def start_monitoring(self):
        """Start compliance monitoring"""
        self.monitoring_active = True
        self.audit_trail.start_audit_thread()

        # Start monitoring tasks
        tasks = []
        for rule_name, rule in self.compliance_rules.items():
            task = asyncio.create_task(
                self._monitor_rule(rule_name, rule)
            )
            tasks.append(task)

        # Keep monitoring running
        while self.monitoring_active:
            await asyncio.sleep(60)  # Check every minute

    async def stop_monitoring(self):
        """Stop compliance monitoring"""
        self.monitoring_active = False
        self.audit_trail.stop_audit_thread()

    async def _monitor_rule(self, rule_name: str, rule: Dict):
        """Monitor specific compliance rule"""
        while self.monitoring_active:
            try:
                # Run compliance check
                is_compliant, details = await rule["check_function"]()

                if not is_compliant:
                    # Log violation
                    violation = {
                        "rule": rule_name,
                        "timestamp": datetime.now().isoformat(),
                        "severity": rule["severity"],
                        "details": details
                    }
                    self.violations.append(violation)

                    # Log to audit trail
                    self.audit_trail.log_event(
                        "compliance_violation",
                        "system",
                        violation,
                        rule["severity"]
                    )

                # Wait for next check based on frequency
                wait_time = self._get_wait_time(rule["frequency"])
                await asyncio.sleep(wait_time)

            except Exception as e:
                print(f"Error monitoring rule {rule_name}: {e}")
                await asyncio.sleep(60)

    def _get_wait_time(self, frequency: str) -> int:
        """Get wait time in seconds based on frequency"""
        frequency_map = {
            "hourly": 3600,
            "daily": 86400,
            "weekly": 604800
        }
        return frequency_map.get(frequency, 3600)

    async def _check_data_retention(self) -> tuple:
        """Check data retention compliance"""
        # Mock implementation
        is_compliant = True
        details = {"retention_period": "365 days", "data_status": "compliant"}
        return is_compliant, details

    async def _check_access_control(self) -> tuple:
        """Check access control compliance"""
        # Mock implementation
        is_compliant = True
        details = {"access_controls": "enabled", "authentication": "multi-factor"}
        return is_compliant, details

    async def _check_encryption(self) -> tuple:
        """Check encryption compliance"""
        # Mock implementation
        is_compliant = True
        details = {"encryption_at_rest": "AES-256", "encryption_in_transit": "TLS 1.3"}
        return is_compliant, details

    async def _check_audit_logging(self) -> tuple:
        """Check audit logging compliance"""
        # Mock implementation
        is_compliant = True
        details = {"audit_logs": "enabled", "retention": "1 year"}
        return is_compliant, details

    async def _check_consent_management(self) -> tuple:
        """Check consent management compliance"""
        # Mock implementation
        is_compliant = True
        details = {"consent_records": "maintained", "withdrawal_process": "available"}
        return is_compliant, details

    def get_compliance_status(self) -> Dict:
        """Get current compliance status"""
        active_violations = [v for v in self.violations
                           if (datetime.now() - datetime.fromisoformat(v["timestamp"])).days < 30]

        return {
            "monitoring_active": self.monitoring_active,
            "active_violations": len(active_violations),
            "total_violations": len(self.violations),
            "rules_monitored": len(self.compliance_rules),
            "compliance_score": self._calculate_compliance_score(active_violations),
            "recent_violations": active_violations[-10:]
        }

    def _calculate_compliance_score(self, violations: List[Dict]) -> float:
        """Calculate overall compliance score"""
        if not violations:
            return 1.0

        # Weight violations by severity
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        }

        total_weight = 0
        max_weight = len(self.compliance_rules)  # Maximum possible violations

        for violation in violations:
            weight = severity_weights.get(violation["severity"], 0.1)
            total_weight += weight

        # Score = 1 - (total_weight / max_weight)
        score = 1 - (total_weight / max_weight)
        return max(0, min(1, score))

    def generate_compliance_report(self) -> Dict:
        """Generate comprehensive compliance report"""
        status = self.get_compliance_status()

        # Analyze violations by type
        violations_by_rule = {}
        for violation in self.violations:
            rule = violation["rule"]
            if rule not in violations_by_rule:
                violations_by_rule[rule] = []
            violations_by_rule[rule].append(violation)

        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(violations_by_rule)

        return {
            "report_generated": datetime.now().isoformat(),
            "compliance_status": status,
            "violations_by_rule": violations_by_rule,
            "recommendations": recommendations,
            "compliance_frameworks": ["GDPR", "HIPAA", "SOC2", "ISO27001"]
        }

    def _generate_compliance_recommendations(self, violations_by_rule: Dict) -> List[str]:
        """Generate compliance improvement recommendations"""
        recommendations = []

        for rule, violations in violations_by_rule.items():
            if len(violations) > 5:  # Frequent violations
                recommendations.append(f"Address frequent violations in {rule} - review implementation")

            # Check for critical violations
            critical_violations = [v for v in violations if v["severity"] == "critical"]
            if critical_violations:
                recommendations.append(f"Critical violations detected in {rule} - immediate action required")

        # General recommendations
        if not violations_by_rule:
            recommendations.append("Excellent compliance record - maintain current practices")
        else:
            recommendations.append("Schedule regular compliance reviews and training")

        return recommendations
```

## 11.6 Quick Reference

### 11.6.1 Security Checklist

```python
# Security implementation checklist
SECURITY_CHECKLIST = {
    "data_classification": [
        "✓ Implement data classification system",
        "✓ Define security levels and handling procedures",
        "✓ Train staff on data classification",
        "✓ Regular classification audits"
    ],
    "encryption": [
        "✓ Encrypt data at rest (AES-256)",
        "✓ Encrypt data in transit (TLS 1.3)",
        "✓ Manage encryption keys securely",
        "✓ Regular encryption validation"
    ],
    "access_control": [
        "✓ Implement least privilege access",
        "✓ Multi-factor authentication",
        "✓ Regular access reviews",
        "✓ Account lifecycle management"
    ],
    "model_security": [
        "✓ Implement adversarial defenses",
        "✓ Regular model robustness testing",
        "✓ Model versioning and integrity checks",
        "✓ Secure model deployment practices"
    ],
    "audit_logging": [
        "✓ Comprehensive audit trails",
        "✓ Log all sensitive operations",
        "✓ Regular log reviews",
        "✓ Secure log storage"
    ],
    "compliance": [
        "✓ Identify applicable regulations",
        "✓ Implement compliance monitoring",
        "✓ Regular compliance assessments",
        "✓ Documentation and reporting"
    ]
}
```

### 11.6.2 Common Security Issues and Solutions

```python
# Security issues and solutions
SECURITY_ISSUES = {
    "data_breaches": {
        "description": "Unauthorized access to sensitive data",
        "causes": ["Weak access controls", "Unencrypted data", "Insider threats"],
        "solutions": [
            "Implement strong authentication",
            "Encrypt all sensitive data",
            "Regular security audits",
            "Employee security training"
        ],
        "prevention": "Multi-layered security approach"
    },
    "adversarial_attacks": {
        "description": "Malicious inputs designed to fool ML models",
        "causes": ["Lack of robustness testing", "No adversarial training"],
        "solutions": [
            "Adversarial training",
            "Input validation and sanitization",
            "Anomaly detection systems",
            "Regular robustness testing"
        ],
        "prevention": "Defense-in-depth for ML systems"
    },
    "model_theft": {
        "description": "Unauthorized extraction of model parameters",
        "causes": ["Unprotected API endpoints", "Excessive model access"],
        "solutions": [
            "API rate limiting",
            "Model output perturbation",
            "Query monitoring",
            "Access control restrictions"
        ],
        "prevention": "Protect model intellectual property"
    },
    "privacy_violations": {
        "description": "Unauthorized use or disclosure of personal data",
        "causes": ["Inadequate data governance", "Lack of consent management"],
        "solutions": [
            "Implement data minimization",
            "Consent management system",
            "Data anonymization",
            "Regular privacy audits"
        ],
        "prevention": "Privacy by design principles"
    }
}
```

### 11.6.3 Compliance Framework Summary

```python
# Compliance frameworks summary
COMPLIANCE_FRAMEWORKS = {
    "GDPR": {
        "focus": "Data protection and privacy",
        "key_requirements": [
            "Lawful basis for processing",
            "Data subject rights",
            "Consent management",
            "Data breach notification",
            "Data protection officer"
        ],
        "penalties": "Up to 4% of global revenue or €20M",
        "applicability": "EU citizens' data"
    },
    "HIPAA": {
        "focus": "Protected Health Information (PHI)",
        "key_requirements": [
            "PHI safeguards",
            "Breach notification",
            "Access controls",
            "Audit controls",
            "Risk analysis"
        ],
        "penalties": "Up to $50,000 per violation",
        "applicability": "Healthcare organizations"
    },
    "SOC2": {
        "focus": "Service organization controls",
        "key_requirements": [
            "Security",
            "Availability",
            "Processing integrity",
            "Confidentiality",
            "Privacy"
        ],
        "penalties": "Loss of certification",
        "applicability": "Service organizations"
    },
    "ISO27001": {
        "focus": "Information security management",
        "key_requirements": [
            "ISMS implementation",
            "Risk assessment",
            "Security controls",
            "Continuous improvement",
            "Management review"
        ],
        "penalties": "Loss of certification",
        "applicability": "All organizations"
    }
}
```

## Key Takeaways

**Security Fundamentals:**
- Implement CIA triad (Confidentiality, Integrity, Availability)
- Classify data and apply appropriate security controls
- Use threat modeling to identify and mitigate risks

**Data Security:**
- Encrypt sensitive data at rest and in transit
- Implement data classification and handling procedures
- Maintain data retention policies

**Model Security:**
- Protect against adversarial attacks
- Implement robustness testing
- Secure model deployment and access

**Compliance:**
- Understand applicable regulations (GDPR, HIPAA, SOC2, ISO27001)
- Implement compliance monitoring
- Maintain comprehensive audit trails

**Best Practices:**
- Regular security assessments and audits
- Employee security training
- Incident response planning
- Continuous improvement

---

**Navigation**: [← Previous: Production Best Practices](10_Production_Best_Practices.md) | [Main Index](README.md) | [Next: Future Trends →](12_Future_Trends.md)