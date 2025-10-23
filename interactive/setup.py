#!/usr/bin/env python3
"""
Setup script for Interactive AI Learning Notebooks

This script automates the setup process for the interactive notebook environment,
including dependency installation, dataset downloads, and configuration.
"""

import os
import sys
import subprocess
import platform
import importlib.util
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class SetupManager:
    """Manages the setup process for interactive AI learning notebooks"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.utils_dir = self.project_root / "utils"
        self.notebooks_dir = self.project_root / "notebooks"
        self.datasets_dir = self.project_root / "datasets"
        self.models_dir = self.project_root / "models"

        # Setup status tracking
        self.setup_status = {
            'python_version': False,
            'requirements': False,
            'directories': False,
            'utils': False,
            'datasets': False,
            'gpu_support': False
        }

    def print_header(self):
        """Print setup header"""
        print("="*60)
        print("🚀 Interactive AI Learning Notebooks Setup")
        print("="*60)
        print()

    def print_section(self, title):
        """Print section header"""
        print(f"\n📋 {title}")
        print("-" * 40)

    def check_python_version(self):
        """Check if Python version is compatible"""
        self.print_section("Checking Python Version")

        version = sys.version_info
        if version.major >= 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
            self.setup_status['python_version'] = True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not compatible")
            print("   Required: Python 3.8 or higher")
            return False

        return True

    def create_directories(self):
        """Create necessary directories"""
        self.print_section("Creating Directories")

        directories = [
            self.datasets_dir,
            self.models_dir,
            self.utils_dir,
            self.notebooks_dir / "logs",
            self.notebooks_dir / "checkpoints"
        ]

        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"✅ Created: {directory.relative_to(self.project_root)}")
            except Exception as e:
                print(f"❌ Failed to create {directory}: {e}")
                return False

        self.setup_status['directories'] = True
        return True

    def install_requirements(self):
        """Install Python requirements"""
        self.print_section("Installing Requirements")

        if not self.requirements_file.exists():
            print(f"❌ Requirements file not found: {self.requirements_file}")
            return False

        try:
            # Upgrade pip first
            print("📦 Upgrading pip...")
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                          check=True, capture_output=True)

            # Install requirements
            print("📦 Installing packages from requirements.txt...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)
            ], check=True, capture_output=True, text=True)

            print("✅ Requirements installed successfully")
            self.setup_status['requirements'] = True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install requirements: {e}")
            print("   Trying to install essential packages individually...")

            # Essential packages for basic functionality
            essential_packages = [
                "numpy", "pandas", "matplotlib", "seaborn", "scikit-learn",
                "jupyter", "ipywidgets", "torch", "transformers"
            ]

            for package in essential_packages:
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package],
                                  check=True, capture_output=True)
                    print(f"✅ Installed: {package}")
                except subprocess.CalledProcessError:
                    print(f"⚠️  Failed to install: {package}")

        return True

    def check_gpu_support(self):
        """Check for GPU support"""
        self.print_section("Checking GPU Support")

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"✅ GPU detected: {gpu_name}")
                print(f"   GPU Memory: {gpu_memory:.1f} GB")
                self.setup_status['gpu_support'] = True
            else:
                print("ℹ️  No GPU detected - CPU mode will be used")
                print("   GPU acceleration will not be available")

        except ImportError:
            print("⚠️  PyTorch not installed - skipping GPU check")

        return True

    def create_utility_modules(self):
        """Create utility modules if they don't exist"""
        self.print_section("Setting Up Utility Modules")

        # Check if utils directory exists and has files
        if not self.utils_dir.exists():
            print("❌ Utils directory not found")
            return False

        util_files = list(self.utils_dir.glob("*.py"))
        if not util_files:
            print("ℹ️  No utility files found - this is normal for fresh setup")

        print(f"✅ Utility directory ready: {len(util_files)} utility files found")
        self.setup_status['utils'] = True
        return True

    def download_sample_datasets(self):
        """Download sample datasets for testing"""
        self.print_section("Downloading Sample Datasets")

        try:
            import pandas as pd
            from sklearn.datasets import load_iris, load_wine, load_breast_cancer

            # Create datasets directory
            datasets_path = self.datasets_dir / "sample"
            datasets_path.mkdir(exist_ok=True)

            # Download and save classic datasets
            datasets = {
                'iris.csv': load_iris(),
                'wine.csv': load_wine(),
                'breast_cancer.csv': load_breast_cancer()
            }

            for filename, dataset in datasets.items():
                filepath = datasets_path / filename
                df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
                df['target'] = dataset.target
                df.to_csv(filepath, index=False)
                print(f"✅ Downloaded: {filename}")

            print("✅ Sample datasets downloaded successfully")
            self.setup_status['datasets'] = True

        except Exception as e:
            print(f"⚠️  Could not download sample datasets: {e}")
            print("   You can download datasets manually when running notebooks")
            return True

        return True

    def create_config_file(self):
        """Create configuration file"""
        self.print_section("Creating Configuration")

        config = {
            'project_root': str(self.project_root),
            'datasets_dir': str(self.datasets_dir),
            'models_dir': str(self.models_dir),
            'utils_dir': str(self.utils_dir),
            'gpu_available': self.setup_status['gpu_support'],
            'setup_date': pd.Timestamp.now().isoformat(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }

        config_file = self.project_root / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"✅ Configuration saved to: {config_file}")
        return True

    def run_setup_tests(self):
        """Run basic setup tests"""
        self.print_section("Running Setup Tests")

        test_results = []

        # Test essential imports
        essential_modules = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
            'torch', 'tensorflow', 'transformers', 'jupyter', 'ipywidgets'
        ]

        for module in essential_modules:
            try:
                importlib.import_module(module)
                print(f"✅ {module}")
                test_results.append(True)
            except ImportError:
                print(f"❌ {module}")
                test_results.append(False)

        # Test directory structure
        required_dirs = [self.datasets_dir, self.models_dir, self.utils_dir]
        for directory in required_dirs:
            if directory.exists():
                print(f"✅ {directory.name} directory")
                test_results.append(True)
            else:
                print(f"❌ {directory.name} directory")
                test_results.append(False)

        success_rate = sum(test_results) / len(test_results)
        print(f"\n📊 Setup Success Rate: {success_rate:.1%}")

        if success_rate >= 0.8:
            print("🎉 Setup completed successfully!")
            return True
        else:
            print("⚠️  Setup completed with some issues")
            print("   You may need to install some packages manually")
            return True

    def print_final_instructions(self):
        """Print final setup instructions"""
        self.print_section("Next Steps")

        print("🚀 Setup complete! Here's how to get started:")
        print()
        print("1. Launch Jupyter Notebook:")
        print("   jupyter notebook")
        print()
        print("2. Or launch Jupyter Lab:")
        print("   jupyter lab")
        print()
        print("3. Navigate to the notebooks directory")
        print("4. Start with '01_Foundational_Machine_Learning'")
        print()
        print("📚 Recommended Learning Path:")
        print("   • Machine Learning Foundations")
        print("   • Deep Learning Fundamentals")
        print("   • Large Language Models")
        print("   • Advanced Topics")
        print()
        print("💡 Tips:")
        print("   • Read the README.md file first")
        print("   • Check system requirements for deep learning notebooks")
        print("   • Use GPU acceleration if available")
        print("   • Join our community for support")
        print()

    def run_setup(self):
        """Run the complete setup process"""
        self.print_header()

        # Run setup steps
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Directory Creation", self.create_directories),
            ("Requirements Installation", self.install_requirements),
            ("GPU Support Check", self.check_gpu_support),
            ("Utility Modules Setup", self.create_utility_modules),
            ("Sample Datasets Download", self.download_sample_datasets),
            ("Configuration File Creation", self.create_config_file),
            ("Setup Tests", self.run_setup_tests)
        ]

        for step_name, step_function in steps:
            try:
                success = step_function()
                if not success and step_name == "Python Version Check":
                    print(f"❌ Critical failure in {step_name}")
                    return False
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                if step_name == "Python Version Check":
                    return False

        self.print_final_instructions()
        return True

def main():
    """Main setup function"""
    setup_manager = SetupManager()

    try:
        success = setup_manager.run_setup()
        if success:
            print("\n🎉 Setup completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()