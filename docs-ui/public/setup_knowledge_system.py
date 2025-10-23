#!/usr/bin/env python3
"""
Setup and Installation Script for Advanced Knowledge System

This script handles:
- Environment validation
- Dependency installation
- Configuration setup
- Initial system initialization
- Testing and validation
"""

import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeSystemSetup:
    """Setup and installation manager for the knowledge system"""

    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent
        self.python_version = sys.version_info
        self.platform = sys.platform

        # Installation tracking
        self.installation_log = []
        self.errors = []
        self.warnings = []

        # Required Python version
        self.min_python_version = (3, 7)

        # Essential packages that must be installed
        self.essential_packages = [
            'numpy',
            'pandas',
            'networkx',
            'scikit-learn',
            'plotly',
            'nltk',
            'spacy'
        ]

        # Optional packages with enhanced functionality
        self.optional_packages = {
            'transformers': 'Advanced NLP and embeddings',
            'torch': 'Deep learning support',
            'sentence-transformers': 'Sentence embeddings',
            'matplotlib': 'Additional visualization options',
            'seaborn': 'Statistical visualizations',
            'schedule': 'Background task scheduling',
            'fastapi': 'Web API interface',
            'redis': 'Advanced caching',
            'jupyter': 'Interactive notebooks'
        }

    def log_message(self, message: str, level: str = 'INFO'):
        """Log a message"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.installation_log.append(log_entry)

        if level == 'ERROR':
            logger.error(message)
            self.errors.append(message)
        elif level == 'WARNING':
            logger.warning(message)
            self.warnings.append(message)
        else:
            logger.info(message)

    def check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements"""
        self.log_message("Checking system requirements...")

        # Check Python version
        if self.python_version < self.min_python_version:
            self.log_message(
                f"Python {self.min_python_version[0]}.{self.min_python_version[1]}+ required, "
                f"found {self.python_version.major}.{self.python_version.minor}",
                'ERROR'
            )
            return False

        self.log_message(f"✓ Python version: {self.python_version.major}.{self.python_version.minor}")

        # Check available memory (basic check)
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2:
                self.log_message(
                    f"Low memory detected: {available_memory_gb:.1f}GB available. "
                    "4GB+ recommended for optimal performance.",
                    'WARNING'
                )
            else:
                self.log_message(f"✓ Available memory: {available_memory_gb:.1f}GB")
        except ImportError:
            self.log_message("Cannot check memory usage (psutil not available)", 'WARNING')

        # Check disk space
        try:
            disk_usage = self.base_path.statvfs
            if hasattr(self.base_path, 'statvfs'):  # Unix-like systems
                stat = self.base_path.statvfs('.')
                free_space_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                if free_space_gb < 1:
                    self.log_message(
                        f"Low disk space: {free_space_gb:.1f}GB available. "
                        "2GB+ recommended.",
                        'WARNING'
                    )
                else:
                    self.log_message(f"✓ Available disk space: {free_space_gb:.1f}GB")
        except:
            self.log_message("Cannot check disk space", 'WARNING')

        return len(self.errors) == 0

    def install_dependencies(self, upgrade: bool = False) -> bool:
        """Install required dependencies"""
        self.log_message("Installing dependencies...")

        # Read requirements file
        requirements_file = self.base_path / "knowledge_system_requirements.txt"
        if not requirements_file.exists():
            self.log_message("Requirements file not found!", 'ERROR')
            return False

        try:
            # Install essential packages first
            self.log_message("Installing essential packages...")
            for package in self.essential_packages:
                self._install_package(package, upgrade)

            # Install remaining requirements
            self.log_message("Installing additional requirements...")
            cmd = [sys.executable, '-m', 'pip', 'install']
            if upgrade:
                cmd.append('--upgrade')
            cmd.extend(['-r', str(requirements_file)])

            result = subprocess.run(
                cmd,
                cwd=self.base_path,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                self.log_message("✓ Dependencies installed successfully")
            else:
                self.log_message(f"Error installing dependencies: {result.stderr}", 'ERROR')
                return False

            # Install spacy model
            self._install_spacy_model()

            # Download NLTK data
            self._download_nltk_data()

        except Exception as e:
            self.log_message(f"Error during dependency installation: {e}", 'ERROR')
            return False

        return True

    def _install_package(self, package: str, upgrade: bool = False):
        """Install a single package"""
        try:
            cmd = [sys.executable, '-m', 'pip', 'install']
            if upgrade:
                cmd.append('--upgrade')
            cmd.append(package)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )

            if result.returncode == 0:
                self.log_message(f"✓ {package} installed successfully")
            else:
                self.log_message(f"Warning: Could not install {package}: {result.stderr}", 'WARNING')

        except subprocess.TimeoutExpired:
            self.log_message(f"Timeout installing {package}", 'WARNING')
        except Exception as e:
            self.log_message(f"Error installing {package}: {e}", 'WARNING')

    def _install_spacy_model(self):
        """Install spacy English model"""
        try:
            import spacy
            try:
                spacy.load('en_core_web_sm')
                self.log_message("✓ spacy English model already installed")
            except OSError:
                self.log_message("Installing spacy English model...")
                subprocess.run([
                    sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'
                ], check=True)
                self.log_message("✓ spacy English model installed")
        except ImportError:
            self.log_message("spacy not available, skipping model installation", 'WARNING')

    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            import nltk
            import ssl

            # Handle SSL certificate issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context

            required_data = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']

            for data in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data}')
                    self.log_message(f"✓ NLTK {data} already downloaded")
                except LookupError:
                    self.log_message(f"Downloading NLTK {data}...")
                    nltk.download(data, quiet=True)

        except ImportError:
            self.log_message("NLTK not available, skipping data download", 'WARNING')

    def create_directories(self):
        """Create necessary directories"""
        self.log_message("Creating directory structure...")

        directories = [
            'visualizations',
            'cache',
            'temp',
            'logs',
            'data',
            'exports',
            'backups'
        ]

        for directory in directories:
            dir_path = self.base_path / directory
            dir_path.mkdir(exist_ok=True)
            self.log_message(f"✓ Created directory: {directory}")

    def create_configuration(self):
        """Create configuration files"""
        self.log_message("Creating configuration files...")

        # Main configuration
        config = {
            'system': {
                'base_path': str(self.base_path),
                'version': '1.0.0',
                'debug': False,
                'log_level': 'INFO'
            },
            'knowledge_graph': {
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'similarity_threshold': 0.3,
                'max_concepts_per_file': 100,
                'cache_embeddings': True
            },
            'cross_referencer': {
                'min_confidence_threshold': 0.3,
                'max_suggestions_per_file': 20,
                'auto_insert_threshold': 0.8,
                'enable_auto_insertion': False
            },
            'content_discovery': {
                'min_topic_coherence': 0.3,
                'max_topics': 20,
                'min_cluster_size': 3,
                'trending_threshold': 0.1
            },
            'visualization': {
                'default_width': 1200,
                'default_height': 800,
                'color_scheme': 'plotly',
                'interactive': True,
                'save_format': 'html'
            },
            'scheduler': {
                'enable_auto_updates': True,
                'update_interval_hours': 24,
                'enable_maintenance': True
            }
        }

        config_file = self.base_path / "knowledge_system_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        self.log_message(f"✓ Configuration saved to {config_file}")

        # Create .gitignore entries
        gitignore_entries = [
            '# Knowledge System',
            'cache/',
            'temp/',
            'logs/',
            '*.log',
            'visualizations/*.html',
            'exports/',
            'backups/',
            'knowledge_graph_cache.pkl',
            'cross_reference_cache.json',
            'content_discovery_cache.json',
            'orchestrator_state.json',
            '__pycache__/',
            '*.pyc',
            '.env'
        ]

        gitignore_file = self.base_path / ".gitignore"
        existing_entries = set()
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                existing_entries = set(line.strip() for line in f if line.strip())

        new_entries = [entry for entry in gitignore_entries if entry not in existing_entries]
        if new_entries:
            with open(gitignore_file, 'a') as f:
                f.write('\n'.join(new_entries) + '\n')
            self.log_message("✓ Updated .gitignore file")

    def test_installation(self) -> bool:
        """Test if installation was successful"""
        self.log_message("Testing installation...")

        try:
            # Test imports
            import_test_results = {}

            # Essential imports
            essential_imports = {
                'numpy': 'numpy',
                'pandas': 'pandas',
                'networkx': 'networkx',
                'sklearn': 'scikit-learn',
                'plotly': 'plotly'
            }

            for display_name, module_name in essential_imports.items():
                try:
                    __import__(module_name)
                    import_test_results[display_name] = '✓'
                    self.log_message(f"✓ {display_name} imported successfully")
                except ImportError as e:
                    import_test_results[display_name] = f'✗ ({e})'
                    self.log_message(f"✗ Could not import {display_name}: {e}", 'ERROR')

            # Test optional imports
            optional_imports = {
                'nltk': 'nltk',
                'spacy': 'spacy',
                'transformers': 'transformers',
                'torch': 'torch'
            }

            for display_name, module_name in optional_imports.items():
                try:
                    __import__(module_name)
                    self.log_message(f"✓ {display_name} (optional) available")
                except ImportError:
                    self.log_message(f"○ {display_name} (optional) not available", 'WARNING')

            # Test our modules
            try:
                from knowledge_graph_system import AIKnowledgeGraph
                self.log_message("✓ Knowledge graph module imported successfully")
            except ImportError as e:
                self.log_message(f"✗ Could not import knowledge graph module: {e}", 'ERROR')

            try:
                from intelligent_cross_referencer import IntelligentCrossReferencer
                self.log_message("✓ Cross-referencer module imported successfully")
            except ImportError as e:
                self.log_message(f"✗ Could not import cross-referencer module: {e}", 'ERROR')

            try:
                from content_discovery_system import ContentDiscoverySystem
                self.log_message("✓ Content discovery module imported successfully")
            except ImportError as e:
                self.log_message(f"✗ Could not import content discovery module: {e}", 'ERROR')

            try:
                from knowledge_visualization_system import KnowledgeVisualizationSystem
                self.log_message("✓ Visualization module imported successfully")
            except ImportError as e:
                self.log_message(f"✗ Could not import visualization module: {e}", 'ERROR')

            try:
                from ai_knowledge_orchestrator import AIKnowledgeOrchestrator
                self.log_message("✓ Orchestrator module imported successfully")
            except ImportError as e:
                self.log_message(f"✗ Could not import orchestrator module: {e}", 'ERROR')

            return len([e for e in self.errors if 'import' in e.lower()]) == 0

        except Exception as e:
            self.log_message(f"Error during installation test: {e}", 'ERROR')
            return False

    def initialize_system(self) -> bool:
        """Initialize the knowledge system"""
        self.log_message("Initializing knowledge system...")

        try:
            # Import and initialize the orchestrator
            from ai_knowledge_orchestrator import AIKnowledgeOrchestrator

            orchestrator = AIKnowledgeOrchestrator(str(self.base_path))

            # Run initialization
            init_result = orchestrator.initialize_system(force_rebuild=True)

            if init_result['status'] == 'success':
                self.log_message("✓ Knowledge system initialized successfully")
                self.log_message(f"  - Concepts: {init_result['knowledge_graph']['concepts']}")
                self.log_message(f"  - Relationships: {init_result['knowledge_graph']['relationships']}")
                self.log_message(f"  - Documents: {init_result['content_discovery']['documents']}")
                self.log_message(f"  - Topics: {init_result['content_discovery']['topics']}")
                self.log_message(f"  - System health: {init_result['system_health']:.2f}")
                return True
            else:
                self.log_message(f"✗ System initialization failed: {init_result.get('error_message', 'Unknown error')}", 'ERROR')
                return False

        except Exception as e:
            self.log_message(f"Error during system initialization: {e}", 'ERROR')
            return False

    def save_installation_log(self):
        """Save installation log"""
        log_file = self.base_path / "installation_log.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'python_version': f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            'platform': self.platform,
            'base_path': str(self.base_path),
            'installation_log': self.installation_log,
            'errors': self.errors,
            'warnings': self.warnings,
            'success': len(self.errors) == 0
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        self.log_message(f"✓ Installation log saved to {log_file}")

    def run_setup(self, force reinstall: bool = False) -> bool:
        """Run complete setup process"""
        print("=" * 60)
        print("AI Knowledge System Setup")
        print("=" * 60)

        start_time = datetime.now()

        # Step 1: Check system requirements
        if not self.check_system_requirements():
            print("\n❌ System requirements not met. Please address the errors above.")
            return False

        # Step 2: Install dependencies
        if not self.install_dependencies(upgrade=reinstall):
            print("\n❌ Dependency installation failed. Please check the errors above.")
            return False

        # Step 3: Create directories
        self.create_directories()

        # Step 4: Create configuration
        self.create_configuration()

        # Step 5: Test installation
        if not self.test_installation():
            print("\n❌ Installation test failed. Please check the errors above.")
            return False

        # Step 6: Initialize system
        if not self.initialize_system():
            print("\n❌ System initialization failed. Please check the errors above.")
            return False

        # Step 7: Save log
        self.save_installation_log()

        # Calculate setup time
        setup_time = datetime.now() - start_time

        print("\n" + "=" * 60)
        print("✅ Setup completed successfully!")
        print("=" * 60)
        print(f"Setup time: {setup_time.total_seconds():.2f} seconds")
        print(f"Installation log: {self.base_path}/installation_log.json")
        print(f"Configuration file: {self.base_path}/knowledge_system_config.json")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} warning(s) occurred during setup")
            for warning in self.warnings:
                print(f"   - {warning}")

        print("\nNext steps:")
        print("1. Review the configuration file if needed")
        print("2. Run 'python ai_knowledge_orchestrator.py diagnostics' to verify system health")
        print("3. Run 'python ai_knowledge_orchestrator.py report' to see system analytics")
        print("4. Start the scheduler with 'python ai_knowledge_orchestrator.py scheduler'")

        return True

def main():
    """Main setup function"""
    import argparse

    parser = argparse.ArgumentParser(description='Setup AI Knowledge System')
    parser.add_argument('--reinstall', action='store_true', help='Reinstall all dependencies')
    parser.add_argument('--skip-init', action='store_true', help='Skip system initialization')
    parser.add_argument('--base-path', type=str, help='Base path for installation')

    args = parser.parse_args()

    setup = KnowledgeSystemSetup(args.base_path)

    try:
        if args.skip_init:
            # Run setup without system initialization
            success = (
                setup.check_system_requirements() and
                setup.install_dependencies(upgrade=args.reinstall) and
                setup.test_installation()
            )

            if success:
                setup.create_directories()
                setup.create_configuration()
                setup.save_installation_log()
                print("\n✅ Dependencies installed successfully!")
                print("Run with --skip-init=False to initialize the system.")
            else:
                print("\n❌ Setup failed. Check the errors above.")
                return 1
        else:
            success = setup.run_setup(force reinstall=args.reinstall)
            return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n❌ Setup interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Setup failed with unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())