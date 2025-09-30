# Edge AI and Federated Learning

**Navigation**: [← Module 7: CI/CD for Machine Learning](07_CICD_for_Machine_Learning.md) | [Main Index](README.md) | [Module 9: AIOps and Automation →](09_AIOps_and_Automation.md)

## Overview

Edge AI and Federated Learning enable distributed machine learning where models can be trained and deployed directly on edge devices while preserving data privacy and reducing latency.

## Edge AI Deployment

### Model Optimization for Edge Devices

```python
import tensorflow as tf
import torch
import torch.nn as nn
import torch.quantization
import numpy as np
from typing import Dict, Any, Tuple, Optional
import tempfile
import os
import json
import logging
from abc import ABC, abstractmethod

class ModelOptimizer:
    """
    Comprehensive model optimization for edge deployment.
    """

    def __init__(self, model_path: str, output_dir: str = "optimized_models"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.logger = self.setup_logger()
        os.makedirs(output_dir, exist_ok=True)

    def setup_logger(self):
        """Setup logging for optimization process"""
        logger = logging.getLogger('model_optimizer')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def optimize_tensorflow_model(self, target_platform: str = 'mobile') -> Dict[str, Any]:
        """Optimize TensorFlow models for edge deployment"""
        try:
            # Load model
            model = tf.keras.models.load_model(self.model_path)

            optimization_results = {
                'original_model': {
                    'size_mb': self.get_model_size(self.model_path),
                    'parameters': model.count_params()
                },
                'optimizations': {}
            }

            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            # Basic optimization
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            basic_path = os.path.join(self.output_dir, 'model_basic.tflite')
            with open(basic_path, 'wb') as f:
                f.write(tflite_model)

            optimization_results['optimizations']['basic'] = {
                'path': basic_path,
                'size_mb': os.path.getsize(basic_path) / (1024 * 1024),
                'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(basic_path) / (1024 * 1024))
            }

            # Quantization optimization
            if target_platform == 'mobile':
                # Full integer quantization
                def representative_dataset():
                    # This should be provided by the user
                    for _ in range(100):
                        data = np.random.rand(1, *model.input_shape[1:]).astype(np.float32)
                        yield [data]

                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

                quantized_model = converter.convert()
                quantized_path = os.path.join(self.output_dir, 'model_quantized.tflite')
                with open(quantized_path, 'wb') as f:
                    f.write(quantized_model)

                optimization_results['optimizations']['quantized'] = {
                    'path': quantized_path,
                    'size_mb': os.path.getsize(quantized_path) / (1024 * 1024),
                    'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(quantized_path) / (1024 * 1024))
                }

            # Pruning optimization
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                    0.5, begin_step=0, frequency=100
                )
            }

            try:
                import tensorflow_model_optimization as tfmot
                model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

                # Fine-tune pruned model
                model_for_pruning.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                # Add fine-tuning logic here

                pruned_model_path = os.path.join(self.output_dir, 'model_pruned.h5')
                model_for_pruning.save(pruned_model_path)

                optimization_results['optimizations']['pruned'] = {
                    'path': pruned_model_path,
                    'size_mb': os.path.getsize(pruned_model_path) / (1024 * 1024),
                    'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(pruned_model_path) / (1024 * 1024))
                }
            except ImportError:
                self.logger.warning("TensorFlow Model Optimization Toolkit not available")

            # Generate metadata
            metadata_path = os.path.join(self.output_dir, 'optimization_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(optimization_results, f, indent=2)

            self.logger.info(f"TensorFlow model optimization completed. Results saved to {self.output_dir}")
            return optimization_results

        except Exception as e:
            self.logger.error(f"TensorFlow optimization failed: {e}")
            raise

    def optimize_pytorch_model(self, target_platform: str = 'mobile') -> Dict[str, Any]:
        """Optimize PyTorch models for edge deployment"""
        try:
            # Load model
            device = torch.device('cpu')
            model = torch.load(self.model_path, map_location=device)
            model.eval()

            optimization_results = {
                'original_model': {
                    'size_mb': self.get_model_size(self.model_path),
                    'parameters': sum(p.numel() for p in model.parameters())
                },
                'optimizations': {}
            }

            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )

            quantized_path = os.path.join(self.output_dir, 'model_quantized_dynamic.pt')
            torch.save(quantized_model.state_dict(), quantized_path)

            optimization_results['optimizations']['dynamic_quantized'] = {
                'path': quantized_path,
                'size_mb': os.path.getsize(quantized_path) / (1024 * 1024),
                'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(quantized_path) / (1024 * 1024))
            }

            # Static quantization (requires calibration)
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)

            # Calibrate with representative data
            # This should be provided by the user
            for _ in range(100):
                data = torch.randn(1, *list(model.input_shape[1:]))
                model(data)

            quantized_static = torch.quantization.convert(model, inplace=False)
            static_path = os.path.join(self.output_dir, 'model_quantized_static.pt')
            torch.save(quantized_static.state_dict(), static_path)

            optimization_results['optimizations']['static_quantized'] = {
                'path': static_path,
                'size_mb': os.path.getsize(static_path) / (1024 * 1024),
                'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(static_path) / (1024 * 1024))
            }

            # Convert to TorchScript for deployment
            scripted_model = torch.jit.script(quantized_model)
            scripted_path = os.path.join(self.output_dir, 'model_scripted.pt')
            torch.jit.save(scripted_model, scripted_path)

            optimization_results['optimizations']['torchscript'] = {
                'path': scripted_path,
                'size_mb': os.path.getsize(scripted_path) / (1024 * 1024),
                'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(scripted_path) / (1024 * 1024))
            }

            # Convert to ONNX for cross-platform compatibility
            if target_platform == 'mobile':
                dummy_input = torch.randn(1, *list(model.input_shape[1:]))
                onnx_path = os.path.join(self.output_dir, 'model.onnx')
                torch.onnx.export(
                    quantized_model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )

                optimization_results['optimizations']['onnx'] = {
                    'path': onnx_path,
                    'size_mb': os.path.getsize(onnx_path) / (1024 * 1024),
                    'compression_ratio': optimization_results['original_model']['size_mb'] / (os.path.getsize(onnx_path) / (1024 * 1024))
                }

            # Generate metadata
            metadata_path = os.path.join(self.output_dir, 'optimization_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(optimization_results, f, indent=2)

            self.logger.info(f"PyTorch model optimization completed. Results saved to {self.output_dir}")
            return optimization_results

        except Exception as e:
            self.logger.error(f"PyTorch optimization failed: {e}")
            raise

    def get_model_size(self, model_path: str) -> float:
        """Get model size in MB"""
        return os.path.getsize(model_path) / (1024 * 1024)

    def benchmark_model_performance(self, model_path: str, input_shape: Tuple[int, ...],
                                  n_runs: int = 100) -> Dict[str, Any]:
        """Benchmark model performance on target hardware"""
        try:
            if model_path.endswith('.tflite'):
                return self.benchmark_tflite_model(model_path, input_shape, n_runs)
            elif model_path.endswith('.pt'):
                return self.benchmark_pytorch_model(model_path, input_shape, n_runs)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")

        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            raise

    def benchmark_tflite_model(self, model_path: str, input_shape: Tuple[int, ...],
                              n_runs: int = 100) -> Dict[str, Any]:
        """Benchmark TensorFlow Lite model"""
        try:
            import tflite_runtime.interpreter as tflite

            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Warm up
            input_data = np.random.randn(*input_shape).astype(np.float32)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Benchmark
            latencies = []
            memory_usage = []

            for _ in range(n_runs):
                import time
                start_time = time.time()

                input_data = np.random.randn(*input_shape).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])

                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

                # Memory usage (approximate)
                import psutil
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

            return {
                'model_path': model_path,
                'format': 'tflite',
                'latency_stats': {
                    'mean_ms': np.mean(latencies),
                    'median_ms': np.median(latencies),
                    'p95_ms': np.percentile(latencies, 95),
                    'p99_ms': np.percentile(latencies, 99),
                    'std_ms': np.std(latencies)
                },
                'memory_stats': {
                    'mean_mb': np.mean(memory_usage),
                    'max_mb': np.max(memory_usage)
                },
                'throughput': n_runs / sum(latencies) * 1000,  # predictions/second
                'runs': n_runs
            }

        except Exception as e:
            self.logger.error(f"TFLite benchmarking failed: {e}")
            raise

    def benchmark_pytorch_model(self, model_path: str, input_shape: Tuple[int, ...],
                              n_runs: int = 100) -> Dict[str, Any]:
        """Benchmark PyTorch model"""
        try:
            device = torch.device('cpu')
            model = torch.load(model_path, map_location=device)
            model.eval()

            # Warm up
            input_data = torch.randn(*input_shape)
            with torch.no_grad():
                _ = model(input_data)

            # Benchmark
            latencies = []
            memory_usage = []

            for _ in range(n_runs):
                import time
                start_time = time.time()

                input_data = torch.randn(*input_shape)
                with torch.no_grad():
                    output = model(input_data)

                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms

                # Memory usage
                import psutil
                process = psutil.Process()
                memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB

            return {
                'model_path': model_path,
                'format': 'pytorch',
                'latency_stats': {
                    'mean_ms': np.mean(latencies),
                    'median_ms': np.median(latencies),
                    'p95_ms': np.percentile(latencies, 95),
                    'p99_ms': np.percentile(latencies, 99),
                    'std_ms': np.std(latencies)
                },
                'memory_stats': {
                    'mean_mb': np.mean(memory_usage),
                    'max_mb': np.max(memory_usage)
                },
                'throughput': n_runs / sum(latencies) * 1000,  # predictions/second
                'runs': n_runs
            }

        except Exception as e:
            self.logger.error(f"PyTorch benchmarking failed: {e}")
            raise

class EdgeDeploymentManager:
    """
    Manage deployment of optimized models to edge devices.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self.setup_logger()
        self.device_registry = DeviceRegistry(config.get('devices', []))

    def setup_logger(self):
        """Setup logging for deployment manager"""
        logger = logging.getLogger('edge_deployment')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def deploy_to_device(self, model_path: str, device_id: str,
                        deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to specific edge device"""
        try:
            device = self.device_registry.get_device(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")

            deployment_results = {
                'device_id': device_id,
                'model_path': model_path,
                'deployment_time': datetime.now().isoformat(),
                'status': 'success'
            }

            # Validate device compatibility
            compatibility_result = self.validate_device_compatibility(model_path, device)
            if not compatibility_result['compatible']:
                return {
                    **deployment_results,
                    'status': 'failed',
                    'error': f"Device compatibility check failed: {compatibility_result['issues']}"
                }

            # Transfer model to device
            transfer_result = self.transfer_model_to_device(model_path, device)
            if not transfer_result['success']:
                return {
                    **deployment_results,
                    'status': 'failed',
                    'error': transfer_result['error']
                }

            # Install model on device
            install_result = self.install_model_on_device(
                transfer_result['remote_path'],
                device,
                deployment_config
            )
            if not install_result['success']:
                return {
                    **deployment_results,
                    'status': 'failed',
                    'error': install_result['error']
                }

            # Validate deployment
            validation_result = self.validate_deployment(device, deployment_config)
            if not validation_result['success']:
                return {
                    **deployment_results,
                    'status': 'failed',
                    'error': validation_result['error']
                }

            deployment_results.update({
                'model_size_mb': os.path.getsize(model_path) / (1024 * 1024),
                'transfer_time_ms': transfer_result['transfer_time_ms'],
                'install_time_ms': install_result['install_time_ms'],
                'validation_metrics': validation_result['metrics']
            })

            self.logger.info(f"Successfully deployed model to device {device_id}")
            return deployment_results

        except Exception as e:
            self.logger.error(f"Deployment to device {device_id} failed: {e}")
            return {
                'device_id': device_id,
                'model_path': model_path,
                'deployment_time': datetime.now().isoformat(),
                'status': 'failed',
                'error': str(e)
            }

    def validate_device_compatibility(self, model_path: str, device: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if device can run the model"""
        try:
            issues = []

            # Check model format support
            model_format = model_path.split('.')[-1].lower()
            supported_formats = device.get('supported_formats', [])

            if model_format not in supported_formats:
                issues.append(f"Device does not support {model_format} format")

            # Check memory requirements
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            available_memory = device.get('available_memory_mb', 0)

            if model_size > available_memory * 0.8:  # Use 80% of available memory
                issues.append(f"Model size ({model_size:.2f}MB) exceeds available memory ({available_memory}MB)")

            # Check hardware acceleration
            if model_format in ['tflite', 'pt'] and not device.get('has_gpu', False):
                issues.append("Device lacks GPU acceleration for optimal performance")

            return {
                'compatible': len(issues) == 0,
                'issues': issues
            }

        except Exception as e:
            return {
                'compatible': False,
                'issues': [f"Compatibility check failed: {str(e)}"]
            }

    def transfer_model_to_device(self, model_path: str, device: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer model file to edge device"""
        try:
            import time
            start_time = time.time()

            # Simulate file transfer (replace with actual implementation)
            import hashlib
            file_hash = hashlib.md5(open(model_path, 'rb').read()).hexdigest()

            remote_path = f"/models/{os.path.basename(model_path)}"

            # Simulate transfer time based on file size and network speed
            file_size = os.path.getsize(model_path)
            network_speed = device.get('network_speed_mbps', 10)  # Mbps
            transfer_time = (file_size * 8) / (network_speed * 1024 * 1024)  # seconds

            time.sleep(min(transfer_time, 5))  # Cap at 5 seconds for simulation

            transfer_time_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'remote_path': remote_path,
                'file_hash': file_hash,
                'transfer_time_ms': transfer_time_ms,
                'file_size_bytes': file_size
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def install_model_on_device(self, remote_path: str, device: Dict[str, Any],
                              config: Dict[str, Any]) -> Dict[str, Any]:
        """Install and configure model on device"""
        try:
            import time
            start_time = time.time()

            # Simulate installation process
            install_steps = [
                "Extracting model files",
                "Setting up dependencies",
                "Configuring model parameters",
                "Setting up monitoring",
                "Verifying installation"
            ]

            for step in install_steps:
                self.logger.info(f"Installation step: {step}")
                time.sleep(0.1)  # Simulate processing time

            install_time_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'install_time_ms': install_time_ms,
                'installation_steps': install_steps,
                'config_applied': config
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def validate_deployment(self, device: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that model is working correctly on device"""
        try:
            # Simulate validation tests
            validation_tests = [
                {
                    'name': 'model_load_test',
                    'description': 'Test if model loads correctly',
                    'passed': True
                },
                {
                    'name': 'inference_test',
                    'description': 'Test basic inference functionality',
                    'passed': True
                },
                {
                    'name': 'performance_test',
                    'description': 'Test inference performance meets requirements',
                    'passed': True,
                    'latency_ms': 15.3,
                    'throughput': 65.2
                },
                {
                    'name': 'memory_test',
                    'description': 'Test memory usage within limits',
                    'passed': True,
                    'memory_usage_mb': 45.2
                }
            ]

            all_passed = all(test['passed'] for test in validation_tests)

            return {
                'success': all_passed,
                'tests': validation_tests,
                'metrics': {
                    'total_tests': len(validation_tests),
                    'passed_tests': sum(1 for test in validation_tests if test['passed']),
                    'success_rate': sum(1 for test in validation_tests if test['passed']) / len(validation_tests)
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def deploy_to_multiple_devices(self, model_path: str, device_ids: List[str],
                                 deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy model to multiple devices in parallel"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        deployment_results = {
            'total_devices': len(device_ids),
            'successful_deployments': 0,
            'failed_deployments': 0,
            'device_results': {}
        }

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all deployment tasks
            future_to_device = {
                executor.submit(self.deploy_to_device, model_path, device_id, deployment_config): device_id
                for device_id in device_ids
            }

            # Collect results
            for future in as_completed(future_to_device):
                device_id = future_to_device[future]
                try:
                    result = future.result()
                    deployment_results['device_results'][device_id] = result

                    if result['status'] == 'success':
                        deployment_results['successful_deployments'] += 1
                    else:
                        deployment_results['failed_deployments'] += 1

                except Exception as e:
                    deployment_results['device_results'][device_id] = {
                        'device_id': device_id,
                        'status': 'failed',
                        'error': str(e)
                    }
                    deployment_results['failed_deployments'] += 1

        deployment_results['success_rate'] = (
            deployment_results['successful_deployments'] / deployment_results['total_devices']
        )

        return deployment_results

class DeviceRegistry:
    """Registry for managing edge devices"""

    def __init__(self, devices: List[Dict[str, Any]]):
        self.devices = {device['device_id']: device for device in devices}

    def get_device(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get device information by ID"""
        return self.devices.get(device_id)

    def register_device(self, device: Dict[str, Any]) -> bool:
        """Register new edge device"""
        device_id = device.get('device_id')
        if not device_id:
            return False

        self.devices[device_id] = device
        return True

    def get_compatible_devices(self, model_requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get devices that meet model requirements"""
        compatible_devices = []

        for device_id, device in self.devices.items():
            is_compatible = True

            # Check format support
            if 'required_format' in model_requirements:
                if model_requirements['required_format'] not in device.get('supported_formats', []):
                    is_compatible = False

            # Check memory requirements
            if 'required_memory_mb' in model_requirements:
                if device.get('available_memory_mb', 0) < model_requirements['required_memory_mb']:
                    is_compatible = False

            # Check GPU requirements
            if model_requirements.get('requires_gpu', False):
                if not device.get('has_gpu', False):
                    is_compatible = False

            if is_compatible:
                compatible_devices.append(device)

        return compatible_devices

# Mobile deployment example
class MobileDeploymentManager:
    """Specialized manager for mobile device deployment"""

    def __init__(self):
        self.platform_configs = {
            'android': {
                'supported_formats': ['tflite', 'onnx'],
                'package_manager': 'gradle',
                'deployment_method': 'app_bundle'
            },
            'ios': {
                'supported_formats': ['coreml', 'tflite'],
                'package_manager': 'cocoapods',
                'deployment_method': 'app_store'
            }
        }

    def create_android_deployment_package(self, model_path: str, app_config: Dict[str, Any]) -> str:
        """Create Android deployment package"""
        try:
            import zipfile
            import tempfile

            # Create temporary directory for package
            with tempfile.TemporaryDirectory() as temp_dir:
                package_path = os.path.join(temp_dir, 'ml_model.aar')

                with zipfile.ZipFile(package_path, 'w') as zip_file:
                    # Add model file
                    zip_file.write(model_path, 'assets/model.tflite')

                    # Add metadata
                    metadata = {
                        'model_version': app_config.get('version', '1.0.0'),
                        'model_hash': self.calculate_file_hash(model_path),
                        'input_shape': app_config.get('input_shape'),
                        'output_shape': app_config.get('output_shape'),
                        'description': app_config.get('description', 'Mobile ML Model')
                    }

                    zip_file.writestr('assets/metadata.json', json.dumps(metadata, indent=2))

                    # Add AndroidManifest.xml
                    manifest_content = self.generate_android_manifest(app_config)
                    zip_file.writestr('AndroidManifest.xml', manifest_content)

                # Copy to final location
                final_path = os.path.join('deployments', 'android', f'model_{app_config.get("version", "1.0.0")}.aar')
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                import shutil
                shutil.copy2(package_path, final_path)

                return final_path

        except Exception as e:
            raise Exception(f"Android package creation failed: {e}")

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def generate_android_manifest(self, app_config: Dict[str, Any]) -> str:
        """Generate AndroidManifest.xml content"""
        return f"""<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="{app_config.get('package_name', 'com.example.mlapp')}">

    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />

    <application
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true"
        android:theme="@style/AppTheme">

        <meta-data
            android:name="com.google.ml.vision.DEPENDENCIES"
            android:value="ocr" />

    </application>
</manifest>
"""

# Usage examples
if __name__ == "__main__":
    # TensorFlow model optimization
    tf_optimizer = ModelOptimizer(
        model_path="models/tensorflow_model.h5",
        output_dir="optimized_models/tensorflow"
    )

    tf_results = tf_optimizer.optimize_tensorflow_model(target_platform='mobile')
    print("TensorFlow Optimization Results:", json.dumps(tf_results, indent=2))

    # PyTorch model optimization
    pt_optimizer = ModelOptimizer(
        model_path="models/pytorch_model.pt",
        output_dir="optimized_models/pytorch"
    )

    pt_results = pt_optimizer.optimize_pytorch_model(target_platform='mobile')
    print("PyTorch Optimization Results:", json.dumps(pt_results, indent=2))

    # Benchmark performance
    benchmark_results = tf_optimizer.benchmark_model_performance(
        "optimized_models/tensorflow/model_quantized.tflite",
        input_shape=(1, 224, 224, 3),
        n_runs=100
    )
    print("Benchmark Results:", json.dumps(benchmark_results, indent=2))

    # Edge deployment
    deployment_config = {
        'devices': [
            {
                'device_id': 'edge_device_001',
                'device_type': 'raspberry_pi',
                'supported_formats': ['tflite', 'pt'],
                'available_memory_mb': 1024,
                'has_gpu': False,
                'network_speed_mbps': 100
            },
            {
                'device_id': 'mobile_device_001',
                'device_type': 'android',
                'supported_formats': ['tflite'],
                'available_memory_mb': 512,
                'has_gpu': True,
                'network_speed_mbps': 50
            }
        ]
    }

    deployment_manager = EdgeDeploymentManager(deployment_config)

    # Deploy to single device
    single_deployment = deployment_manager.deploy_to_device(
        model_path="optimized_models/tensorflow/model_quantized.tflite",
        device_id="edge_device_001",
        deployment_config={
            'model_name': 'image_classifier',
            'version': '1.0.0',
            'monitoring_enabled': True
        }
    )
    print("Single Device Deployment:", json.dumps(single_deployment, indent=2))

    # Deploy to multiple devices
    multi_deployment = deployment_manager.deploy_to_multiple_devices(
        model_path="optimized_models/tensorflow/model_quantized.tflite",
        device_ids=["edge_device_001", "mobile_device_001"],
        deployment_config={
            'model_name': 'image_classifier',
            'version': '1.0.0',
            'monitoring_enabled': True
        }
    )
    print("Multi-Device Deployment:", json.dumps(multi_deployment, indent=2))
```

## Federated Learning Framework

### Complete Federated Learning System

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Tuple, Callable
import json
import logging
import hashlib
import threading
import time
from datetime import datetime
from abc import ABC, abstractmethod
from collections import defaultdict
import asyncio
import aiohttp
from dataclasses import dataclass
import pickle

@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    num_rounds: int = 100
    num_clients: int = 10
    fraction_fit: float = 0.1
    fraction_evaluate: float = 0.1
    min_fit_clients: int = 2
    min_evaluate_clients: int = 2
    min_available_clients: int = 2
    round_timeout: float = 600.0  # seconds
    accept_failures: bool = True
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    aggregation_strategy: str = 'fedavg'  # fedavg, fedprox, fedbn
    privacy_mechanism: Optional[str] = None  # None, 'dp', 'pa', 'sm'
    privacy_budget: Optional[float] = None
    secure_aggregation: bool = False

class FederatedClient:
    """
    Client-side federated learning participant.
    """

    def __init__(self, client_id: str, model: nn.Module, train_data: Dataset,
                 test_data: Dataset, config: FederatedConfig):
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Setup client logger"""
        logger = logging.getLogger(f'federated_client_{self.client_id}')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def train_round(self, current_round: int) -> Dict[str, Any]:
        """Train model locally for one round"""
        try:
            self.model.train()
            train_loader = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )

            criterion = nn.CrossEntropyLoss()

            epoch_losses = []
            for epoch in range(self.config.local_epochs):
                batch_losses = []
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    batch_losses.append(loss.item())

                epoch_loss = np.mean(batch_losses)
                epoch_losses.append(epoch_loss)

                self.logger.info(f"Round {current_round}, Epoch {epoch}, Loss: {epoch_loss:.4f}")

            # Calculate model update
            model_update = self.calculate_model_update()

            # Apply privacy mechanism if configured
            if self.config.privacy_mechanism:
                model_update = self.apply_privacy_mechanism(model_update)

            # Evaluate model
            metrics = self.evaluate_model()

            return {
                'client_id': self.client_id,
                'round': current_round,
                'model_update': model_update,
                'num_examples': len(self.train_data),
                'metrics': metrics,
                'training_loss': np.mean(epoch_losses),
                'status': 'success'
            }

        except Exception as e:
            self.logger.error(f"Training round {current_round} failed for client {self.client_id}: {e}")
            return {
                'client_id': self.client_id,
                'round': current_round,
                'status': 'failed',
                'error': str(e)
            }

    def calculate_model_update(self) -> Dict[str, torch.Tensor]:
        """Calculate model update (parameter differences)"""
        model_update = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                model_update[name] = param.data.clone()

        return model_update

    def apply_privacy_mechanism(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply privacy mechanism to model updates"""
        if self.config.privacy_mechanism == 'dp':
            return self.apply_differential_privacy(model_update)
        elif self.config.privacy_mechanism == 'pa':
            return self.apply_perturbation_aggregation(model_update)
        elif self.config.privacy_mechanism == 'sm':
            return self.apply_secure_multiplication(model_update)
        else:
            return model_update

    def apply_differential_privacy(self, model_update: Dict[str, torch.Tensor],
                                  epsilon: float = 1.0, delta: float = 1e-5) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to model updates"""
        sensitivity = 2.0  # L2 sensitivity
        scale = sensitivity / epsilon

        private_update = {}
        for name, tensor in model_update.items():
            # Add Gaussian noise
            noise = torch.normal(0, scale, tensor.shape)
            private_update[name] = tensor + noise.to(self.device)

        return private_update

    def apply_perturbation_aggregation(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply perturbation for secure aggregation"""
        # Generate random perturbation masks
        perturbation = {}
        for name, tensor in model_update.items():
            mask = torch.randn_like(tensor)
            perturbation[name] = tensor + 0.1 * mask

        return perturbation

    def apply_secure_multiplication(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply secure multiplication using secret sharing"""
        # Simplified implementation
        secure_update = {}
        for name, tensor in model_update.items():
            # Encode values for secure computation
            encoded_tensor = tensor * 2  # Simple encoding
            secure_update[name] = encoded_tensor

        return secure_update

    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate model on test data"""
        self.model.eval()
        test_loader = DataLoader(self.test_data, batch_size=self.config.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total
        }

    def receive_global_model(self, global_parameters: Dict[str, torch.Tensor]):
        """Receive and load global model parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in global_parameters:
                    param.data.copy_(global_parameters[name])

class FederatedServer:
    """
    Server-side federated learning coordinator.
    """

    def __init__(self, model: nn.Module, config: FederatedConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.clients = {}
        self.round_history = []
        self.global_parameters = self.get_model_parameters()
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Setup server logger"""
        logger = logging.getLogger('federated_server')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def register_client(self, client: FederatedClient):
        """Register a federated learning client"""
        self.clients[client.client_id] = client
        self.logger.info(f"Registered client: {client.client_id}")

    def get_model_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current model parameters"""
        parameters = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters[name] = param.data.clone()
        return parameters

    def set_model_parameters(self, parameters: Dict[str, torch.Tensor]):
        """Set model parameters"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data.copy_(parameters[name])

    def select_clients(self, fraction: float) -> List[str]:
        """Select subset of clients for participation"""
        num_selected = max(1, int(len(self.clients) * fraction))
        available_clients = list(self.clients.keys())
        selected_clients = np.random.choice(
            available_clients,
            size=min(num_selected, len(available_clients)),
            replace=False
        )
        return list(selected_clients)

    def aggregate_updates(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client model updates"""
        if not client_updates:
            return self.global_parameters

        if self.config.aggregation_strategy == 'fedavg':
            return self.fedavg_aggregation(client_updates)
        elif self.config.aggregation_strategy == 'fedprox':
            return self.fedprox_aggregation(client_updates)
        elif self.config.aggregation_strategy == 'fedbn':
            return self.fedbn_aggregation(client_updates)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.config.aggregation_strategy}")

    def fedavg_aggregation(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Federated Averaging (FedAvg) aggregation"""
        # Weight by number of examples
        total_examples = sum(update['num_examples'] for update in client_updates)

        aggregated_parameters = {}
        for name in self.global_parameters.keys():
            weighted_sum = torch.zeros_like(self.global_parameters[name])

            for update in client_updates:
                if name in update['model_update']:
                    weight = update['num_examples'] / total_examples
                    weighted_sum += weight * update['model_update'][name]

            aggregated_parameters[name] = weighted_sum

        return aggregated_parameters

    def fedprox_aggregation(self, client_updates: List[Dict[str, Any]],
                           mu: float = 0.01) -> Dict[str, torch.Tensor]:
        """Federated Proximal (FedProx) aggregation"""
        # First do FedAvg
        fedavg_params = self.fedavg_aggregation(client_updates)

        # Add proximal term
        proximal_params = {}
        for name in self.global_parameters.keys():
            if name in fedavg_params:
                proximal_params[name] = (
                    fedavg_params[name] + mu * (fedavg_params[name] - self.global_parameters[name])
                )
            else:
                proximal_params[name] = self.global_parameters[name]

        return proximal_params

    def fedbn_aggregation(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Federated Batch Normalization (FedBN) aggregation"""
        # Aggregate all parameters except batch normalization
        aggregated_parameters = {}

        for name in self.global_parameters.keys():
            if 'bn' not in name.lower():  # Not batch normalization
                if any(name in update['model_update'] for update in client_updates):
                    # Weighted average
                    total_examples = sum(update['num_examples'] for update in client_updates)
                    weighted_sum = torch.zeros_like(self.global_parameters[name])

                    for update in client_updates:
                        if name in update['model_update']:
                            weight = update['num_examples'] / total_examples
                            weighted_sum += weight * update['model_update'][name]

                    aggregated_parameters[name] = weighted_sum
                else:
                    aggregated_parameters[name] = self.global_parameters[name]
            else:
                # Keep local batch normalization parameters
                aggregated_parameters[name] = self.global_parameters[name]

        return aggregated_parameters

    def run_federated_round(self, round_num: int) -> Dict[str, Any]:
        """Run one round of federated learning"""
        start_time = time.time()

        self.logger.info(f"Starting federated round {round_num}")

        # Select participating clients
        selected_clients = self.select_clients(self.config.fraction_fit)
        self.logger.info(f"Selected clients: {selected_clients}")

        # Distribute global model to clients
        for client_id in selected_clients:
            self.clients[client_id].receive_global_model(self.global_parameters)

        # Collect client updates (asynchronous)
        client_updates = self.collect_client_updates(selected_clients, round_num)

        # Aggregate updates
        if client_updates:
            new_parameters = self.aggregate_updates(client_updates)
            self.global_parameters = new_parameters
            self.set_model_parameters(new_parameters)

        # Evaluate global model
        evaluation_metrics = self.evaluate_global_model()

        # Record round history
        round_info = {
            'round': round_num,
            'start_time': datetime.now().isoformat(),
            'selected_clients': selected_clients,
            'participating_clients': [update['client_id'] for update in client_updates],
            'num_participants': len(client_updates),
            'aggregation_method': self.config.aggregation_strategy,
            'evaluation_metrics': evaluation_metrics,
            'round_duration': time.time() - start_time
        }

        self.round_history.append(round_info)

        self.logger.info(f"Completed round {round_num} with {len(client_updates)} participants")
        return round_info

    def collect_client_updates(self, selected_clients: List[str], round_num: int) -> List[Dict[str, Any]]:
        """Collect model updates from clients asynchronously"""
        import concurrent.futures

        client_updates = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(selected_clients)) as executor:
            # Submit training tasks
            future_to_client = {
                executor.submit(self.clients[client_id].train_round, round_num): client_id
                for client_id in selected_clients
            }

            # Collect results
            for future in concurrent.futures.as_completed(future_to_client, timeout=self.config.round_timeout):
                client_id = future_to_client[future]
                try:
                    result = future.result()
                    if result['status'] == 'success':
                        client_updates.append(result)
                    else:
                        self.logger.warning(f"Client {client_id} training failed: {result.get('error')}")
                except Exception as e:
                    self.logger.error(f"Client {client_id} raised exception: {e}")

        return client_updates

    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on available test data"""
        # For simplicity, evaluate on first client's test data
        if not self.clients:
            return {}

        first_client = list(self.clients.values())[0]
        return first_client.evaluate_model()

    def run_federated_training(self) -> List[Dict[str, Any]]:
        """Run complete federated training process"""
        self.logger.info(f"Starting federated training for {self.config.num_rounds} rounds")

        round_results = []

        for round_num in range(self.config.num_rounds):
            round_result = self.run_federated_round(round_num)
            round_results.append(round_result)

            # Early stopping if performance degrades
            if len(round_results) > 5:
                recent_accuracies = [
                    r['evaluation_metrics'].get('accuracy', 0)
                    for r in round_results[-5:]
                ]
                if np.mean(recent_accuracies) < 0.5:  # Arbitrary threshold
                    self.logger.warning("Early stopping due to poor performance")
                    break

        self.logger.info("Federated training completed")
        return round_results

    def save_training_results(self, output_path: str):
        """Save training results to file"""
        results = {
            'config': self.config.__dict__,
            'round_history': self.round_history,
            'final_parameters': {k: v.cpu().numpy() for k, v in self.global_parameters.items()},
            'num_clients': len(self.clients),
            'total_rounds': len(self.round_history)
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Training results saved to {output_path}")

class PrivacyPreservingFL:
    """
    Enhanced federated learning with privacy preservation.
    """

    def __init__(self, server: FederatedServer, privacy_config: Dict[str, Any]):
        self.server = server
        self.privacy_config = privacy_config
        self.secure_aggregator = SecureAggregator(privacy_config)

    def apply_secure_aggregation(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Apply secure aggregation to preserve privacy"""
        if self.privacy_config.get('secure_aggregation', False):
            return self.secure_aggregator.aggregate_securely(client_updates)
        else:
            return self.server.aggregate_updates(client_updates)

class SecureAggregator:
    """
    Secure aggregation implementation for federated learning.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.secret_sharing = SecretSharing(config.get('secret_sharing_threshold', 2))

    def aggregate_securely(self, client_updates: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates securely using secret sharing"""
        if len(client_updates) < self.config.get('secret_sharing_threshold', 2):
            raise ValueError("Insufficient clients for secure aggregation")

        # Apply secret sharing to each parameter
        aggregated_parameters = {}

        for param_name in client_updates[0]['model_update'].keys():
            # Collect parameter values from all clients
            param_values = [
                update['model_update'][param_name]
                for update in client_updates
                if param_name in update['model_update']
            ]

            if param_values:
                # Apply secret sharing
                shares = []
                for value in param_values:
                    client_shares = self.secret_sharing.share_secret(value)
                    shares.append(client_shares)

                # Reconstruct aggregate
                aggregate = self.secret_sharing.reconstruct_aggregate(shares)
                aggregated_parameters[param_name] = aggregate

        return aggregated_parameters

class SecretSharing:
    """
    Secret sharing for secure aggregation.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self.prime = 2**61 - 1  # Large prime number

    def share_secret(self, secret: torch.Tensor) -> List[torch.Tensor]:
        """Share secret using Shamir's Secret Sharing"""
        # Convert tensor to numerical values
        secret_flat = secret.flatten()
        shares = []

        for _ in range(self.threshold):
            # Generate random polynomial coefficients
            coefficients = torch.randn(self.threshold)
            coefficients[0] = secret_flat[0]  # Secret is constant term

            # Evaluate polynomial for each client
            client_share = []
            for i in range(len(secret_flat)):
                # Simple linear sharing for demonstration
                share = coefficients[0] + torch.randn(1) * 0.1
                client_share.append(share)

            shares.append(torch.stack(client_share).reshape(secret.shape))

        return shares

    def reconstruct_aggregate(self, all_shares: List[List[torch.Tensor]]) -> torch.Tensor:
        """Reconstruct aggregate from shared secrets"""
        # Simple averaging for demonstration
        aggregate = torch.zeros_like(all_shares[0][0])

        for client_shares in all_shares:
            for share in client_shares:
                aggregate += share / len(all_shares) / len(client_shares)

        return aggregate

# Utility functions for federated learning
def create_federated_datasets(dataset: Dataset, num_clients: int,
                             non_iid: bool = False) -> List[Dataset]:
    """Split dataset into federated client datasets"""
    dataset_size = len(dataset)
    client_sizes = [dataset_size // num_clients] * num_clients

    # Distribute remaining data
    remainder = dataset_size % num_clients
    for i in range(remainder):
        client_sizes[i] += 1

    if non_iid:
        # Non-IID data distribution
        indices = np.random.permutation(dataset_size)
        client_datasets = []

        start_idx = 0
        for size in client_sizes:
            end_idx = start_idx + size
            client_indices = indices[start_idx:end_idx]
            client_dataset = torch.utils.data.Subset(dataset, client_indices)
            client_datasets.append(client_dataset)
            start_idx = end_idx

        return client_datasets
    else:
        # IID data distribution
        return torch.utils.data.random_split(dataset, client_sizes)

def calculate_fairness_metrics(client_updates: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate fairness metrics across clients"""
    if not client_updates:
        return {}

    # Extract client accuracies
    accuracies = [
        update['metrics']['accuracy']
        for update in client_updates
        if 'metrics' in update and 'accuracy' in update['metrics']
    ]

    if not accuracies:
        return {}

    # Calculate fairness metrics
    accuracy_disparity = max(accuracies) - min(accuracies)
    accuracy_mean = np.mean(accuracies)
    accuracy_std = np.std(accuracies)
    coefficient_of_variation = accuracy_std / accuracy_mean if accuracy_mean > 0 else 0

    return {
        'accuracy_disparity': accuracy_disparity,
        'accuracy_mean': accuracy_mean,
        'accuracy_std': accuracy_std,
        'coefficient_of_variation': coefficient_of_variation,
        'num_clients': len(accuracies)
    }

# Example usage
if __name__ == "__main__":
    # Create simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self, input_size=784, hidden_size=128, num_classes=10):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Create federated configuration
    config = FederatedConfig(
        num_rounds=50,
        num_clients=5,
        fraction_fit=0.8,
        local_epochs=2,
        batch_size=32,
        learning_rate=0.01,
        aggregation_strategy='fedavg',
        privacy_mechanism='dp',
        privacy_budget=1.0,
        secure_aggregation=True
    )

    # Initialize server
    global_model = SimpleModel()
    server = FederatedServer(global_model, config)

    # Create and register clients (simplified)
    for i in range(config.num_clients):
        # In practice, each client would have its own local dataset
        client_model = SimpleModel()

        # Create dummy datasets
        from torch.utils.data import TensorDataset
        dummy_data = torch.randn(1000, 784)
        dummy_labels = torch.randint(0, 10, (1000,))
        train_dataset = TensorDataset(dummy_data, dummy_labels)
        test_dataset = TensorDataset(dummy_data[:200], dummy_labels[:200])

        client = FederatedClient(
            client_id=f"client_{i}",
            model=client_model,
            train_data=train_dataset,
            test_data=test_dataset,
            config=config
        )
        server.register_client(client)

    # Run federated training
    training_results = server.run_federated_training()

    # Save results
    server.save_training_results("federated_training_results.json")

    print(f"Federated training completed with {len(training_results)} rounds")
    print(f"Final accuracy: {training_results[-1]['evaluation_metrics']['accuracy']:.4f}")

    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics([
        {'metrics': {'accuracy': 0.85}},
        {'metrics': {'accuracy': 0.82}},
        {'metrics': {'accuracy': 0.88}},
        {'metrics': {'accuracy': 0.79}},
        {'metrics': {'accuracy': 0.84}}
    ])
    print("Fairness Metrics:", fairness_metrics)
```

## Edge Device Management and Monitoring

```python
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import sqlite3
import threading
import queue
import time

@dataclass
class DeviceMetrics:
    """Device performance metrics"""
    device_id: str
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    temperature: float
    network_latency: float
    prediction_count: int
    error_count: int
    battery_level: Optional[float] = None

@dataclass
class ModelMetrics:
    """Model performance metrics on edge device"""
    device_id: str
    model_version: str
    timestamp: str
    inference_latency_ms: float
    throughput: float
    accuracy: Optional[float] = None
    memory_footprint_mb: float
    cpu_utilization: float
    error_rate: float

class EdgeMonitoringService:
    """
    Monitor and manage edge devices in distributed ML deployments.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = self.setup_logger()
        self.device_registry = {}
        self.metrics_db = MetricsDatabase(config.get('database_path', 'edge_metrics.db'))
        self.alert_manager = AlertManager(config.get('alert_config', {}))
        self.model_updater = ModelUpdater(config.get('update_config', {}))
        self.running = False

    def setup_logger(self):
        """Setup monitoring service logger"""
        logger = logging.getLogger('edge_monitoring')
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    async def register_device(self, device_info: Dict[str, Any]) -> bool:
        """Register new edge device"""
        try:
            device_id = device_info.get('device_id')
            if not device_id:
                raise ValueError("Device ID is required")

            # Validate device information
            required_fields = ['device_type', 'ip_address', 'port', 'supported_models']
            for field in required_fields:
                if field not in device_info:
                    raise ValueError(f"Missing required field: {field}")

            # Check if device is reachable
            is_reachable = await self.check_device_connectivity(device_info)
            if not is_reachable:
                raise ValueError(f"Device {device_id} is not reachable")

            # Register device
            self.device_registry[device_id] = {
                **device_info,
                'registered_at': datetime.now().isoformat(),
                'status': 'active',
                'last_heartbeat': datetime.now().isoformat()
            }

            self.logger.info(f"Successfully registered device: {device_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register device: {e}")
            return False

    async def check_device_connectivity(self, device_info: Dict[str, Any]) -> bool:
        """Check if device is reachable"""
        try:
            url = f"http://{device_info['ip_address']}:{device_info['port']}/health"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    return response.status == 200

        except Exception as e:
            self.logger.warning(f"Connectivity check failed: {e}")
            return False

    async def collect_device_metrics(self, device_id: str) -> Optional[DeviceMetrics]:
        """Collect metrics from edge device"""
        try:
            device = self.device_registry.get(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")

            url = f"http://{device['ip_address']}:{device['port']}/metrics"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        metrics_data = await response.json()

                        return DeviceMetrics(
                            device_id=device_id,
                            timestamp=datetime.now().isoformat(),
                            cpu_usage=metrics_data.get('cpu_usage', 0),
                            memory_usage=metrics_data.get('memory_usage', 0),
                            disk_usage=metrics_data.get('disk_usage', 0),
                            temperature=metrics_data.get('temperature', 0),
                            network_latency=metrics_data.get('network_latency', 0),
                            prediction_count=metrics_data.get('prediction_count', 0),
                            error_count=metrics_data.get('error_count', 0),
                            battery_level=metrics_data.get('battery_level')
                        )
                    else:
                        raise Exception(f"Device returned status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to collect metrics from device {device_id}: {e}")
            return None

    async def collect_model_metrics(self, device_id: str, model_version: str) -> Optional[ModelMetrics]:
        """Collect model performance metrics"""
        try:
            device = self.device_registry.get(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")

            url = f"http://{device['ip_address']}:{device['port']}/model_metrics/{model_version}"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        metrics_data = await response.json()

                        return ModelMetrics(
                            device_id=device_id,
                            model_version=model_version,
                            timestamp=datetime.now().isoformat(),
                            inference_latency_ms=metrics_data.get('inference_latency_ms', 0),
                            throughput=metrics_data.get('throughput', 0),
                            accuracy=metrics_data.get('accuracy'),
                            memory_footprint_mb=metrics_data.get('memory_footprint_mb', 0),
                            cpu_utilization=metrics_data.get('cpu_utilization', 0),
                            error_rate=metrics_data.get('error_rate', 0)
                        )
                    else:
                        raise Exception(f"Device returned status {response.status}")

        except Exception as e:
            self.logger.error(f"Failed to collect model metrics from device {device_id}: {e}")
            return None

    async def monitor_all_devices(self):
        """Monitor all registered devices"""
        self.logger.info("Starting device monitoring")

        while self.running:
            try:
                tasks = []
                for device_id in self.device_registry.keys():
                    task = asyncio.create_task(self.monitor_device(device_id))
                    tasks.append(task)

                # Wait for all monitoring tasks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

                # Wait before next monitoring cycle
                await asyncio.sleep(self.config.get('monitoring_interval', 60))

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait before retrying

    async def monitor_device(self, device_id: str):
        """Monitor individual device"""
        try:
            # Collect device metrics
            device_metrics = await self.collect_device_metrics(device_id)
            if device_metrics:
                await self.metrics_db.store_device_metrics(device_metrics)

                # Check for alerts
                await self.alert_manager.check_device_alerts(device_metrics)

            # Collect model metrics for each model on the device
            device = self.device_registry.get(device_id, {})
            for model_version in device.get('deployed_models', []):
                model_metrics = await self.collect_model_metrics(device_id, model_version)
                if model_metrics:
                    await self.metrics_db.store_model_metrics(model_metrics)
                    await self.alert_manager.check_model_alerts(model_metrics)

            # Update device heartbeat
            if device_id in self.device_registry:
                self.device_registry[device_id]['last_heartbeat'] = datetime.now().isoformat()

        except Exception as e:
            self.logger.error(f"Error monitoring device {device_id}: {e}")

    async def update_device_model(self, device_id: str, model_version: str,
                                 model_path: str) -> bool:
        """Update model on edge device"""
        try:
            device = self.device_registry.get(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")

            # Pre-update checks
            compatibility_result = await self.check_model_compatibility(device_id, model_version)
            if not compatibility_result['compatible']:
                raise ValueError(f"Model not compatible: {compatibility_result['issues']}")

            # Transfer model to device
            transfer_result = await self.model_updater.transfer_model(
                device_id, model_path, model_version
            )
            if not transfer_result['success']:
                raise Exception(f"Model transfer failed: {transfer_result['error']}")

            # Install and validate model
            install_result = await self.model_updater.install_model(
                device_id, model_version
            )
            if not install_result['success']:
                raise Exception(f"Model installation failed: {install_result['error']}")

            # Update device registry
            if 'deployed_models' not in device:
                device['deployed_models'] = []
            device['deployed_models'].append(model_version)

            self.logger.info(f"Successfully updated model {model_version} on device {device_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update model on device {device_id}: {e}")
            return False

    async def check_model_compatibility(self, device_id: str, model_version: str) -> Dict[str, Any]:
        """Check if model is compatible with device"""
        try:
            device = self.device_registry.get(device_id)
            if not device:
                return {'compatible': False, 'issues': ['Device not found']}

            # Get model requirements
            model_requirements = await self.get_model_requirements(model_version)
            if not model_requirements:
                return {'compatible': False, 'issues': ['Model requirements not found']}

            issues = []

            # Check hardware requirements
            if 'min_memory_mb' in model_requirements:
                device_memory = device.get('memory_mb', 0)
                if device_memory < model_requirements['min_memory_mb']:
                    issues.append(f"Insufficient memory: {device_memory}MB < {model_requirements['min_memory_mb']}MB")

            if 'requires_gpu' in model_requirements and model_requirements['requires_gpu']:
                if not device.get('has_gpu', False):
                    issues.append("Model requires GPU but device doesn't have one")

            # Check software requirements
            if 'required_frameworks' in model_requirements:
                device_frameworks = device.get('frameworks', [])
                for framework in model_requirements['required_frameworks']:
                    if framework not in device_frameworks:
                        issues.append(f"Missing framework: {framework}")

            return {
                'compatible': len(issues) == 0,
                'issues': issues
            }

        except Exception as e:
            return {
                'compatible': False,
                'issues': [f"Compatibility check failed: {str(e)}"]
            }

    async def get_model_requirements(self, model_version: str) -> Optional[Dict[str, Any]]:
        """Get model requirements from registry"""
        # This would typically query a model registry
        # For demonstration, return dummy data
        return {
            'min_memory_mb': 512,
            'requires_gpu': False,
            'required_frameworks': ['tensorflow-lite', 'onnx'],
            'model_size_mb': 25.6
        }

    async def generate_device_report(self, device_id: str,
                                   start_date: str, end_date: str) -> Dict[str, Any]:
        """Generate comprehensive device report"""
        try:
            # Get device information
            device = self.device_registry.get(device_id)
            if not device:
                raise ValueError(f"Device {device_id} not found")

            # Get metrics from database
            device_metrics = await self.metrics_db.get_device_metrics(
                device_id, start_date, end_date
            )
            model_metrics = await self.metrics_db.get_model_metrics(
                device_id, start_date, end_date
            )

            # Calculate statistics
            report = {
                'device_info': device,
                'report_period': {
                    'start': start_date,
                    'end': end_date
                },
                'device_statistics': self.calculate_device_statistics(device_metrics),
                'model_statistics': self.calculate_model_statistics(model_metrics),
                'alerts': await self.alert_manager.get_device_alerts(device_id, start_date, end_date),
                'recommendations': self.generate_device_recommendations(device_metrics, model_metrics)
            }

            return report

        except Exception as e:
            self.logger.error(f"Failed to generate device report: {e}")
            raise

    def calculate_device_statistics(self, metrics: List[DeviceMetrics]) -> Dict[str, Any]:
        """Calculate device performance statistics"""
        if not metrics:
            return {}

        metrics_df = pd.DataFrame([asdict(m) for m in metrics])

        statistics = {
            'uptime_percentage': len(metrics) / (24 * 60) * 100,  # Assuming 24-hour period
            'avg_cpu_usage': metrics_df['cpu_usage'].mean(),
            'max_cpu_usage': metrics_df['cpu_usage'].max(),
            'avg_memory_usage': metrics_df['memory_usage'].mean(),
            'max_memory_usage': metrics_df['memory_usage'].max(),
            'avg_temperature': metrics_df['temperature'].mean(),
            'max_temperature': metrics_df['temperature'].max(),
            'total_predictions': metrics_df['prediction_count'].sum(),
            'total_errors': metrics_df['error_count'].sum(),
            'error_rate': metrics_df['error_count'].sum() / max(metrics_df['prediction_count'].sum(), 1) * 100
        }

        return statistics

    def calculate_model_statistics(self, metrics: List[ModelMetrics]) -> Dict[str, Any]:
        """Calculate model performance statistics"""
        if not metrics:
            return {}

        metrics_df = pd.DataFrame([asdict(m) for m in metrics])

        statistics = {}

        # Group by model version
        for model_version in metrics_df['model_version'].unique():
            version_metrics = metrics_df[metrics_df['model_version'] == model_version]

            statistics[model_version] = {
                'avg_inference_latency': version_metrics['inference_latency_ms'].mean(),
                'p95_inference_latency': version_metrics['inference_latency_ms'].quantile(0.95),
                'avg_throughput': version_metrics['throughput'].mean(),
                'avg_memory_footprint': version_metrics['memory_footprint_mb'].mean(),
                'avg_cpu_utilization': version_metrics['cpu_utilization'].mean(),
                'avg_error_rate': version_metrics['error_rate'].mean(),
                'avg_accuracy': version_metrics['accuracy'].mean() if 'accuracy' in version_metrics.columns else None,
                'total_inferences': version_metrics['throughput'].sum()  # Approximate
            }

        return statistics

    def generate_device_recommendations(self, device_metrics: List[DeviceMetrics],
                                       model_metrics: List[ModelMetrics]) -> List[str]:
        """Generate recommendations based on device performance"""
        recommendations = []

        if device_metrics:
            metrics_df = pd.DataFrame([asdict(m) for m in device_metrics])

            # CPU usage recommendations
            avg_cpu = metrics_df['cpu_usage'].mean()
            if avg_cpu > 80:
                recommendations.append("Consider upgrading device CPU or optimizing model complexity")

            # Memory usage recommendations
            avg_memory = metrics_df['memory_usage'].mean()
            if avg_memory > 85:
                recommendations.append("High memory usage detected - consider memory optimization")

            # Temperature recommendations
            max_temp = metrics_df['temperature'].max()
            if max_temp > 70:
                recommendations.append("Device temperature is high - improve cooling or reduce workload")

            # Error rate recommendations
            error_rate = metrics_df['error_count'].sum() / max(metrics_df['prediction_count'].sum(), 1) * 100
            if error_rate > 5:
                recommendations.append("High error rate detected - investigate model or data issues")

        # Model-specific recommendations
        if model_metrics:
            model_df = pd.DataFrame([asdict(m) for m in model_metrics])

            for model_version in model_df['model_version'].unique():
                version_metrics = model_df[model_df['model_version'] == model_version]

                avg_latency = version_metrics['inference_latency_ms'].mean()
                if avg_latency > 100:  # 100ms threshold
                    recommendations.append(f"Model {model_version} has high latency - consider optimization")

                avg_error_rate = version_metrics['error_rate'].mean()
                if avg_error_rate > 2:  # 2% threshold
                    recommendations.append(f"Model {model_version} has high error rate - consider retraining")

        return recommendations

    def start_monitoring(self):
        """Start the monitoring service"""
        self.running = True
        self.logger.info("Edge monitoring service started")

        # Start monitoring in asyncio event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.monitor_all_devices())

    def stop_monitoring(self):
        """Stop the monitoring service"""
        self.running = False
        self.logger.info("Edge monitoring service stopped")

class MetricsDatabase:
    """Database for storing edge device and model metrics"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Device metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                cpu_usage REAL,
                memory_usage REAL,
                disk_usage REAL,
                temperature REAL,
                network_latency REAL,
                prediction_count INTEGER,
                error_count INTEGER,
                battery_level REAL
            )
        ''')

        # Model metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                inference_latency_ms REAL,
                throughput REAL,
                accuracy REAL,
                memory_footprint_mb REAL,
                cpu_utilization REAL,
                error_rate REAL
            )
        ''')

        # Alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                model_version TEXT,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TEXT
            )
        ''')

        conn.commit()
        conn.close()

    async def store_device_metrics(self, metrics: DeviceMetrics):
        """Store device metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO device_metrics
            (device_id, timestamp, cpu_usage, memory_usage, disk_usage, temperature,
             network_latency, prediction_count, error_count, battery_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.device_id, metrics.timestamp, metrics.cpu_usage, metrics.memory_usage,
            metrics.disk_usage, metrics.temperature, metrics.network_latency,
            metrics.prediction_count, metrics.error_count, metrics.battery_level
        ))

        conn.commit()
        conn.close()

    async def store_model_metrics(self, metrics: ModelMetrics):
        """Store model metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO model_metrics
            (device_id, model_version, timestamp, inference_latency_ms, throughput,
             accuracy, memory_footprint_mb, cpu_utilization, error_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.device_id, metrics.model_version, metrics.timestamp,
            metrics.inference_latency_ms, metrics.throughput, metrics.accuracy,
            metrics.memory_footprint_mb, metrics.cpu_utilization, metrics.error_rate
        ))

        conn.commit()
        conn.close()

    async def get_device_metrics(self, device_id: str, start_date: str, end_date: str) -> List[DeviceMetrics]:
        """Get device metrics for a time period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT device_id, timestamp, cpu_usage, memory_usage, disk_usage, temperature,
                   network_latency, prediction_count, error_count, battery_level
            FROM device_metrics
            WHERE device_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (device_id, start_date, end_date))

        rows = cursor.fetchall()
        conn.close()

        return [
            DeviceMetrics(
                device_id=row[0], timestamp=row[1], cpu_usage=row[2], memory_usage=row[3],
                disk_usage=row[4], temperature=row[5], network_latency=row[6],
                prediction_count=row[7], error_count=row[8], battery_level=row[9]
            )
            for row in rows
        ]

    async def get_model_metrics(self, device_id: str, start_date: str, end_date: str) -> List[ModelMetrics]:
        """Get model metrics for a time period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT device_id, model_version, timestamp, inference_latency_ms, throughput,
                   accuracy, memory_footprint_mb, cpu_utilization, error_rate
            FROM model_metrics
            WHERE device_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp
        ''', (device_id, start_date, end_date))

        rows = cursor.fetchall()
        conn.close()

        return [
            ModelMetrics(
                device_id=row[0], model_version=row[1], timestamp=row[2],
                inference_latency_ms=row[3], throughput=row[4], accuracy=row[5],
                memory_footprint_mb=row[6], cpu_utilization=row[7], error_rate=row[8]
            )
            for row in rows
        ]

class AlertManager:
    """Manage alerts for edge devices"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alert_rules = config.get('alert_rules', {})
        self.database = MetricsDatabase(config.get('database_path', 'edge_metrics.db'))

    async def check_device_alerts(self, metrics: DeviceMetrics):
        """Check for device-related alerts"""
        alerts = []

        # CPU usage alert
        if metrics.cpu_usage > self.alert_rules.get('cpu_usage_threshold', 90):
            alerts.append({
                'type': 'high_cpu_usage',
                'severity': 'warning',
                'message': f"High CPU usage: {metrics.cpu_usage:.1f}%"
            })

        # Memory usage alert
        if metrics.memory_usage > self.alert_rules.get('memory_usage_threshold', 90):
            alerts.append({
                'type': 'high_memory_usage',
                'severity': 'warning',
                'message': f"High memory usage: {metrics.memory_usage:.1f}%"
            })

        # Temperature alert
        if metrics.temperature > self.alert_rules.get('temperature_threshold', 75):
            alerts.append({
                'type': 'high_temperature',
                'severity': 'critical',
                'message': f"High temperature: {metrics.temperature:.1f}°C"
            })

        # Error rate alert
        if metrics.prediction_count > 0:
            error_rate = (metrics.error_count / metrics.prediction_count) * 100
            if error_rate > self.alert_rules.get('error_rate_threshold', 5):
                alerts.append({
                    'type': 'high_error_rate',
                    'severity': 'warning',
                    'message': f"High error rate: {error_rate:.2f}%"
                })

        # Store alerts
        for alert in alerts:
            await self.store_alert(metrics.device_id, None, alert)

    async def check_model_alerts(self, metrics: ModelMetrics):
        """Check for model-related alerts"""
        alerts = []

        # Latency alert
        if metrics.inference_latency_ms > self.alert_rules.get('latency_threshold', 200):
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"High inference latency: {metrics.inference_latency_ms:.1f}ms"
            })

        # Error rate alert
        if metrics.error_rate > self.alert_rules.get('model_error_rate_threshold', 3):
            alerts.append({
                'type': 'high_model_error_rate',
                'severity': 'warning',
                'message': f"High model error rate: {metrics.error_rate:.2f}%"
            })

        # Memory footprint alert
        if metrics.memory_footprint_mb > self.alert_rules.get('memory_footprint_threshold', 100):
            alerts.append({
                'type': 'high_memory_footprint',
                'severity': 'warning',
                'message': f"High memory footprint: {metrics.memory_footprint_mb:.1f}MB"
            })

        # Store alerts
        for alert in alerts:
            await self.store_alert(metrics.device_id, metrics.model_version, alert)

    async def store_alert(self, device_id: str, model_version: Optional[str], alert: Dict[str, Any]):
        """Store alert in database"""
        conn = sqlite3.connect(self.config.get('database_path', 'edge_metrics.db'))
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO alerts (device_id, model_version, alert_type, severity, message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            device_id, model_version, alert['type'], alert['severity'],
            alert['message'], datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    async def get_device_alerts(self, device_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get alerts for a device"""
        conn = sqlite3.connect(self.config.get('database_path', 'edge_metrics.db'))
        cursor = conn.cursor()

        cursor.execute('''
            SELECT device_id, model_version, alert_type, severity, message, timestamp, resolved, resolved_at
            FROM alerts
            WHERE device_id = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (device_id, start_date, end_date))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'device_id': row[0],
                'model_version': row[1],
                'alert_type': row[2],
                'severity': row[3],
                'message': row[4],
                'timestamp': row[5],
                'resolved': bool(row[6]),
                'resolved_at': row[7]
            }
            for row in rows
        ]

class ModelUpdater:
    """Handle model updates on edge devices"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    async def transfer_model(self, device_id: str, model_path: str, model_version: str) -> Dict[str, Any]:
        """Transfer model to edge device"""
        # This is a simplified implementation
        # In practice, you would use secure file transfer protocols
        try:
            # Simulate transfer
            import time
            start_time = time.time()

            # Calculate file size and estimate transfer time
            file_size = os.path.getsize(model_path)
            network_speed = 10  # Mbps (simulated)
            transfer_time = (file_size * 8) / (network_speed * 1024 * 1024)

            # Simulate transfer delay
            await asyncio.sleep(min(transfer_time, 2))

            transfer_time_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'transfer_time_ms': transfer_time_ms,
                'file_size_bytes': file_size,
                'remote_path': f"/models/{model_version}.tflite"
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    async def install_model(self, device_id: str, model_version: str) -> Dict[str, Any]:
        """Install model on edge device"""
        try:
            # Simulate installation process
            install_steps = [
                "Extracting model files",
                "Setting up model configuration",
                "Running validation tests",
                "Updating model registry"
            ]

            for step in install_steps:
                await asyncio.sleep(0.1)  # Simulate processing time

            return {
                'success': True,
                'install_time_ms': 400,  # Simulated
                'installation_steps': install_steps
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# Usage example
if __name__ == "__main__":
    # Initialize edge monitoring service
    config = {
        'monitoring_interval': 30,  # seconds
        'database_path': 'edge_metrics.db',
        'alert_config': {
            'alert_rules': {
                'cpu_usage_threshold': 90,
                'memory_usage_threshold': 90,
                'temperature_threshold': 75,
                'error_rate_threshold': 5,
                'latency_threshold': 200,
                'model_error_rate_threshold': 3,
                'memory_footprint_threshold': 100
            }
        },
        'update_config': {}
    }

    monitoring_service = EdgeMonitoringService(config)

    # Example device registration
    device_info = {
        'device_id': 'edge_pi_001',
        'device_type': 'raspberry_pi',
        'ip_address': '192.168.1.100',
        'port': 8000,
        'supported_models': ['image_classifier', 'object_detector'],
        'memory_mb': 1024,
        'has_gpu': False,
        'frameworks': ['tensorflow-lite', 'onnx']
    }

    # In a real application, you would run this as a service
    # monitoring_service.start_monitoring()
```

## Quick Reference

### Edge AI Optimization Techniques

1. **Model Compression**
   - Quantization (INT8, FP16)
   - Pruning (magnitude-based, structured)
   - Knowledge distillation
   - Architecture search (NAS)

2. **Edge Deployment Strategies**
   - Static deployment (pre-installed models)
   - Dynamic deployment (on-demand model loading)
   - Hierarchical deployment (edge + cloud)
   - Federated deployment (distributed training)

3. **Performance Optimization**
   - Hardware acceleration (GPU, TPU, NPU)
   - Model caching and preloading
   - Batch processing
   - Input preprocessing optimization

### Federated Learning Algorithms

1. **Aggregation Methods**
   - FedAvg (Federated Averaging)
   - FedProx (Proximal optimization)
   - FedBN (Batch Normalization)
   - FedMA (Matched Averaging)

2. **Privacy Preservation**
   - Differential Privacy
   - Secure Aggregation
   - Homomorphic Encryption
   - Secret Sharing

3. **Optimization Strategies**
   - Client selection strategies
   - Adaptive learning rates
   - Personalization techniques
   - Fairness-aware aggregation

### Edge Device Management

1. **Monitoring Metrics**
   - Device health (CPU, memory, temperature)
   - Model performance (latency, accuracy, throughput)
   - Network connectivity and bandwidth
   - Battery life and power consumption

2. **Update Management**
   - Over-the-air (OTA) updates
   - A/B testing for new models
   - Rollback mechanisms
   - Version control and compatibility

## Summary

This module provides comprehensive coverage of Edge AI and Federated Learning, including:

- **Model optimization** techniques for edge deployment
- **Federated learning** frameworks with privacy preservation
- **Edge device management** and monitoring systems
- **Distributed training** strategies for privacy-preserving ML
- **Real-world implementations** with performance optimization

The implementation demonstrates how to deploy machine learning models on edge devices while maintaining data privacy through federated learning approaches.

**Next**: [Module 9: AIOps and Automation](09_AIOps_and_Automation.md)