"""
Data Loading Utilities for Interactive Notebooks

This module provides standardized data loading functions for all interactive notebooks,
ensuring consistency and ease of use across different sections.

Author: AI Documentation Project
Date: September 2025
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.datasets import fetch_openml, load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataRegistry:
    """Central registry for all datasets used in interactive notebooks"""

    def __init__(self):
        self.datasets = {}
        self._initialize_datasets()

    def _initialize_datasets(self):
        """Initialize built-in datasets"""

        # Classic ML datasets
        self.datasets['iris'] = {
            'name': 'Iris Dataset',
            'description': 'Classic flower classification dataset with 3 species',
            'type': 'classification',
            'features': 4,
            'classes': 3,
            'samples': 150,
            'loader': self._load_iris
        }

        self.datasets['wine'] = {
            'name': 'Wine Dataset',
            'description': 'Wine classification based on chemical analysis',
            'type': 'classification',
            'features': 13,
            'classes': 3,
            'samples': 178,
            'loader': self._load_wine
        }

        self.datasets['breast_cancer'] = {
            'name': 'Breast Cancer Dataset',
            'description': 'Breast cancer tumor classification',
            'type': 'classification',
            'features': 30,
            'classes': 2,
            'samples': 569,
            'loader': self._load_breast_cancer
        }

        # OpenML datasets
        self.datasets['titanic'] = {
            'name': 'Titanic Dataset',
            'description': 'Titanic passenger survival prediction',
            'type': 'classification',
            'features': 10,
            'classes': 2,
            'samples': 1309,
            'loader': self._load_titanic
        }

        self.datasets['boston'] = {
            'name': 'Boston Housing Dataset',
            'description': 'House price prediction in Boston',
            'type': 'regression',
            'features': 13,
            'target': 1,
            'samples': 506,
            'loader': self._load_boston
        }

        # Synthetic datasets
        self.datasets['synthetic_classification'] = {
            'name': 'Synthetic Classification Dataset',
            'description': 'Synthetic dataset for classification experiments',
            'type': 'classification',
            'features': 10,
            'classes': 3,
            'samples': 1000,
            'loader': self._load_synthetic_classification
        }

        self.datasets['synthetic_regression'] = {
            'name': 'Synthetic Regression Dataset',
            'description': 'Synthetic dataset for regression experiments',
            'type': 'regression',
            'features': 8,
            'target': 1,
            'samples': 1000,
            'loader': self._load_synthetic_regression
        }

    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available datasets: {list(self.datasets.keys())}")
        return self.datasets[dataset_name]

    def list_datasets(self) -> pd.DataFrame:
        """List all available datasets as a DataFrame"""
        data = []
        for name, info in self.datasets.items():
            data.append({
                'Name': name,
                'Description': info['description'],
                'Type': info['type'],
                'Features': info.get('features', 'N/A'),
                'Classes': info.get('classes', 'N/A'),
                'Samples': info.get('samples', 'N/A')
            })
        return pd.DataFrame(data)

    def load_dataset(self, dataset_name: str, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        return self.datasets[dataset_name]['loader'](**kwargs)

    def _load_iris(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the Iris dataset"""
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name='target')
        return df, target

    def _load_wine(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the Wine dataset"""
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name='target')
        return df, target

    def _load_breast_cancer(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the Breast Cancer dataset"""
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        target = pd.Series(data.target, name='target')
        return df, target

    def _load_titanic(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the Titanic dataset from OpenML"""
        data = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
        df = data.frame
        target = df['survived']
        df = df.drop('survived', axis=1)
        return df, target

    def _load_boston(self, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the Boston Housing dataset"""
        data = fetch_openml('boston', version=1, as_frame=True, parser='auto')
        df = data.frame
        target = df['MEDV']
        df = df.drop('MEDV', axis=1)
        return df, target

    def _load_synthetic_classification(self, n_samples=1000, n_features=10, n_classes=3, random_state=42, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load synthetic classification dataset"""
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            random_state=random_state,
            **kwargs
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        target = pd.Series(y, name='target')
        return df, target

    def _load_synthetic_regression(self, n_samples=1000, n_features=8, random_state=42, **kwargs) -> Tuple[pd.DataFrame, pd.Series]:
        """Load synthetic regression dataset"""
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            random_state=random_state,
            **kwargs
        )
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        target = pd.Series(y, name='target')
        return df, target

class DataLoader:
    """Main data loader class for interactive notebooks"""

    def __init__(self):
        self.registry = DataRegistry()
        self.cache = {}

    def load(self, dataset_name: str, split: bool = False, test_size: float = 0.2, random_state: int = 42, **kwargs) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        """
        Load dataset with optional train/test split

        Args:
            dataset_name: Name of the dataset to load
            split: Whether to split into train/test sets
            test_size: Proportion of test set if split=True
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for dataset loading

        Returns:
            If split=False: (X, y)
            If split=True: (X_train, X_test, y_train, y_test)
        """
        cache_key = f"{dataset_name}_{split}_{test_size}_{random_state}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Load dataset
        X, y = self.registry.load_dataset(dataset_name, **kwargs)

        if split:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if self.registry.datasets[dataset_name]['type'] == 'classification' else None
            )
            result = (X_train, X_test, y_train, y_test)
        else:
            result = (X, y)

        # Cache result
        self.cache[cache_key] = result
        return result

    def get_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get information about a dataset"""
        return self.registry.get_dataset_info(dataset_name)

    def list_datasets(self) -> pd.DataFrame:
        """List all available datasets"""
        return self.registry.list_datasets()

    def quick_explore(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """
        Quick exploration of dataset statistics and visualizations

        Args:
            dataset_name: Name of the dataset to explore
            **kwargs: Additional parameters

        Returns:
            Dictionary containing exploration results
        """
        X, y = self.load(dataset_name, **kwargs)
        info = self.get_info(dataset_name)

        exploration = {
            'dataset_info': info,
            'data_shape': X.shape,
            'missing_values': X.isnull().sum().sum(),
            'feature_types': X.dtypes.value_counts().to_dict(),
            'target_distribution': y.value_counts().to_dict() if info['type'] == 'classification' else y.describe().to_dict(),
            'correlation_matrix': X.corr() if X.select_dtypes(include=[np.number]).shape[1] > 1 else None
        }

        return exploration

    def create_visualization_report(self, dataset_name: str, save_path: Optional[str] = None, **kwargs):
        """Create comprehensive visualization report for a dataset"""
        X, y = self.load(dataset_name, **kwargs)
        info = self.get_info(dataset_name)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Dataset Exploration: {info["name"]}', fontsize=16)

        # Plot 1: Target distribution
        if info['type'] == 'classification':
            ax = axes[0, 0]
            y.value_counts().plot(kind='bar', ax=ax)
            ax.set_title('Target Distribution')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
        else:
            ax = axes[0, 0]
            y.plot(kind='hist', bins=30, ax=ax)
            ax.set_title('Target Distribution')
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Count')

        # Plot 2: Feature correlations
        ax = axes[0, 1]
        if X.select_dtypes(include=[np.number]).shape[1] > 1:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            sns.heatmap(X[numeric_cols].corr(), ax=ax, cmap='coolwarm', center=0)
            ax.set_title('Feature Correlations')
        else:
            ax.text(0.5, 0.5, 'Not enough numeric features for correlation',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Correlations')

        # Plot 3: Feature distributions
        ax = axes[1, 0]
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:4]  # Show first 4 numeric features
        if len(numeric_cols) > 0:
            X[numeric_cols].hist(ax=ax, bins=20, alpha=0.7)
            ax.set_title('Feature Distributions')
        else:
            ax.text(0.5, 0.5, 'No numeric features to display',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Distributions')

        # Plot 4: Missing values
        ax = axes[1, 1]
        missing_data = X.isnull().sum()
        if missing_data.sum() > 0:
            missing_data[missing_data > 0].plot(kind='bar', ax=ax)
            ax.set_title('Missing Values')
            ax.set_ylabel('Count')
        else:
            ax.text(0.5, 0.5, 'No missing values',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Missing Values')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

        return fig

# Global instance
data_loader = DataLoader()

# Convenience functions
def load_dataset(dataset_name: str, **kwargs):
    """Convenience function to load a dataset"""
    return data_loader.load(dataset_name, **kwargs)

def get_dataset_info(dataset_name: str):
    """Convenience function to get dataset information"""
    return data_loader.get_info(dataset_name)

def list_datasets():
    """Convenience function to list all datasets"""
    return data_loader.list_datasets()

def quick_explore(dataset_name: str, **kwargs):
    """Convenience function for quick dataset exploration"""
    return data_loader.quick_explore(dataset_name, **kwargs)

def create_visualization_report(dataset_name: str, save_path: Optional[str] = None, **kwargs):
    """Convenience function to create visualization report"""
    return data_loader.create_visualization_report(dataset_name, save_path, **kwargs)