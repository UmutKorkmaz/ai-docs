"""
Data preprocessing module for customer churn prediction.
Handles data loading, cleaning, validation, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from pathlib import Path
import yaml
import pandera as pa
from pandera import Check, Column

from ..utils.logging import get_logger
from ..utils.database import DatabaseManager

logger = get_logger(__name__)


class ChurnDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for customer churn prediction.
    """

    def __init__(self, config_path: str = "config/preprocessing_config.yaml"):
        """
        Initialize the preprocessor with configuration.

        Args:
            config_path: Path to preprocessing configuration file
        """
        self.config = self._load_config(config_path)
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.label_encoders = {}
        self.feature_names = []
        self.target_column = self.config.get("target_column", "churn")

    def _load_config(self, config_path: str) -> Dict:
        """Load preprocessing configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default preprocessing configuration."""
        return {
            "target_column": "churn",
            "numeric_features": [
                "age", "tenure", "monthly_charges", "total_charges",
                "senior_citizen", "dependents", "partner"
            ],
            "categorical_features": [
                "gender", "contract_type", "payment_method",
                "internet_service", "online_security", "tech_support"
            ],
            "feature_engineering": {
                "create_ratio_features": True,
                "create_interaction_features": True,
                "create_polynomial_features": False,
                "create_bins": True
            },
            "outlier_handling": {
                "method": "iqr",  # Options: "iqr", "zscore", "none"
                "threshold": 1.5
            },
            "missing_value_handling": {
                "numeric_strategy": "median",
                "categorical_strategy": "most_frequent"
            }
        }

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from various sources.

        Args:
            data_path: Path to data file or database connection string

        Returns:
            Loaded DataFrame
        """
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            elif data_path.startswith(('postgresql://', 'mysql://', 'sqlite://')):
                db_manager = DatabaseManager(data_path)
                df = db_manager.load_table("customer_data")
            else:
                raise ValueError(f"Unsupported data source: {data_path}")

            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate data using Pandera schema.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            schema = self._create_validation_schema()
            validated_df = schema.validate(df, lazy=True)
            return True, []
        except pa.errors.SchemaErrors as err:
            error_messages = [f"Column {error['column']}: {error['error']}"
                            for error in err.failure_cases]
            return False, error_messages

    def _create_validation_schema(self) -> pa.DataFrameSchema:
        """Create Pandera validation schema."""
        return pa.DataFrameSchema({
            "customer_id": Column(str, Check.str_length(min=1, max=50)),
            "age": Column(int, Check.greater_than(0), Check.less_than(120)),
            "tenure": Column(int, Check.greater_than_or_equal_to(0)),
            "monthly_charges": Column(float, Check.greater_than_or_equal_to(0)),
            "total_charges": Column(float, Check.greater_than_or_equal_to(0)),
            "gender": Column(str, Check.isin(["Male", "Female"])),
            "contract_type": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
            "payment_method": Column(str),
            "churn": Column(int, Check.isin([0, 1]))
        })

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the data.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Handle missing values
        df = self._handle_missing_values(df)

        # Handle outliers
        df = self._handle_outliers(df)

        # Convert data types
        df = self._convert_data_types(df)

        logger.info("Data cleaning completed")
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        logger.info(f"Missing values:\n{missing_counts[missing_counts > 0]}")

        # Drop columns with >70% missing values
        cols_to_drop = missing_percentages[missing_percentages > 70].index.tolist()
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped columns with >70% missing values: {cols_to_drop}")

        # Fill missing values based on configuration
        config = self.config["missing_value_handling"]

        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy=config["numeric_strategy"])
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy=config["categorical_strategy"])
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using configured method."""
        method = self.config["outlier_handling"]["method"]
        threshold = self.config["outlier_handling"]["threshold"]

        if method == "none":
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_column]

        for col in numeric_cols:
            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    logger.info(f"Found {len(outliers)} outliers in {col}")
                    # Cap outliers instead of removing them
                    df[col] = df[col].clip(lower_bound, upper_bound)

            elif method == "zscore":
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold]
                if len(outliers) > 0:
                    logger.info(f"Found {len(outliers)} outliers in {col}")
                    # Cap outliers
                    median_val = df[col].median()
                    df.loc[z_scores > threshold, col] = median_val

        return df

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types appropriately."""
        # Convert boolean columns
        bool_cols = ['senior_citizen', 'partner', 'dependents', 'phone_service',
                    'paperless_billing', 'churn']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)

        # Convert numeric columns
        numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.

        Args:
            df: DataFrame with basic features

        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")

        config = self.config["feature_engineering"]

        if config.get("create_ratio_features", True):
            df = self._create_ratio_features(df)

        if config.get("create_interaction_features", True):
            df = self._create_interaction_features(df)

        if config.get("create_polynomial_features", False):
            df = self._create_polynomial_features(df)

        if config.get("create_bins", True):
            df = self._create_binned_features(df)

        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df

    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio-based features."""
        # Monthly to total charges ratio
        if 'monthly_charges' in df.columns and 'total_charges' in df.columns:
            df['monthly_to_total_ratio'] = df['monthly_charges'] / (df['total_charges'] + 1)

        # Tenure to age ratio
        if 'tenure' in df.columns and 'age' in df.columns:
            df['tenure_to_age_ratio'] = df['tenure'] / (df['age'] + 1)

        # Average monthly charges over tenure
        if 'total_charges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charges'] = np.where(
                df['tenure'] > 0,
                df['total_charges'] / df['tenure'],
                df['monthly_charges']
            )

        return df

    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features."""
        # Service count interactions
        service_cols = ['phone_service', 'internet_service', 'online_security', 'tech_support']
        available_services = [col for col in service_cols if col in df.columns]

        if len(available_services) >= 2:
            df['multiple_services'] = df[available_services].sum(axis=1)

        # Contract and payment method interaction
        if 'contract_type' in df.columns and 'payment_method' in df.columns:
            df['contract_payment_interaction'] = (
                df['contract_type'].astype(str) + '_' + df['payment_method'].astype(str)
            )

        return df

    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for numeric columns."""
        from sklearn.preprocessing import PolynomialFeatures

        numeric_cols = ['age', 'tenure', 'monthly_charges', 'total_charges']
        available_numeric = [col for col in numeric_cols if col in df.columns]

        if len(available_numeric) >= 2:
            poly = PolynomialFeatures(degree=2, include_bias=False)
            poly_features = poly.fit_transform(df[available_numeric])
            poly_feature_names = poly.get_feature_names_out(available_numeric)

            # Exclude original features (already in df)
            new_features = poly_feature_names[len(available_numeric):]
            poly_df = pd.DataFrame(poly_features[:, len(available_numeric):],
                                 columns=new_features, index=df.index)

            df = pd.concat([df, poly_df], axis=1)

        return df

    def _create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binned features."""
        # Age bins
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                   bins=[0, 30, 50, 70, 120],
                                   labels=['young', 'middle', 'senior', 'elderly'])

        # Tenure bins
        if 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(df['tenure'],
                                      bins=[0, 12, 24, 48, 100],
                                      labels=['new', 'short', 'medium', 'long'])

        # Monthly charges bins
        if 'monthly_charges' in df.columns:
            df['charge_group'] = pd.cut(df['monthly_charges'],
                                      bins=[0, 30, 60, 90, 200],
                                      labels=['low', 'medium', 'high', 'premium'])

        return df

    def create_preprocessing_pipeline(self, df: pd.DataFrame) -> ColumnTransformer:
        """
        Create sklearn preprocessing pipeline.

        Args:
            df: DataFrame to analyze for feature types

        Returns:
            Configured ColumnTransformer
        """
        logger.info("Creating preprocessing pipeline...")

        # Identify numeric and categorical features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target column if present
        if self.target_column in numeric_features:
            numeric_features.remove(self.target_column)
        if self.target_column in categorical_features:
            categorical_features.remove(self.target_column)

        self.feature_names = numeric_features + categorical_features

        # Create numeric transformer
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Create categorical transformer
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, numeric_features),
                ('cat', self.categorical_transformer, categorical_features)
            ])

        logger.info(f"Preprocessing pipeline created for {len(self.feature_names)} features")
        return self.preprocessor

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit preprocessing pipeline and transform data.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (X, y) features and target
        """
        logger.info("Fitting and transforming data...")

        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Create and fit preprocessing pipeline
        self.preprocessor = self.create_preprocessing_pipeline(X)
        X_processed = self.preprocessor.fit_transform(X)

        logger.info(f"Data transformed. Feature shape: {X_processed.shape}")
        return X_processed, y

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted preprocessing pipeline.

        Args:
            df: Input DataFrame

        Returns:
            Transformed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        X = df.drop(columns=[self.target_column], errors='ignore')
        X_processed = self.preprocessor.transform(X)

        return X_processed

    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted.")

        return self.preprocessor.get_feature_names_out()

    def save_preprocessor(self, path: str) -> None:
        """Save fitted preprocessor to file."""
        import joblib
        joblib.dump(self.preprocessor, path)
        logger.info(f"Preprocessor saved to {path}")

    def load_preprocessor(self, path: str) -> None:
        """Load preprocessor from file."""
        import joblib
        self.preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")

    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate data summary statistics."""
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }

        # Categorical summaries
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = df[col].value_counts().to_dict()

        return summary


def create_sample_dataset(n_samples: int = 1000, save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Create a sample customer churn dataset for testing.

    Args:
        n_samples: Number of samples to generate
        save_path: Path to save the dataset

    Returns:
        Generated DataFrame
    """
    np.random.seed(42)

    # Generate customer IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(n_samples)]

    # Generate demographic features
    age = np.random.normal(45, 15, n_samples).astype(int)
    age = np.clip(age, 18, 80)

    gender = np.random.choice(['Male', 'Female'], n_samples)
    senior_citizen = (age >= 65).astype(int)

    # Generate service-related features
    tenure = np.random.exponential(30, n_samples).astype(int)
    tenure = np.clip(tenure, 0, 72)

    monthly_charges = np.random.normal(70, 25, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 150)

    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_samples)
    total_charges = np.maximum(total_charges, 0)

    # Generate categorical features
    contract_types = ['Month-to-month', 'One year', 'Two year']
    contract_probs = [0.5, 0.3, 0.2]
    contract_type = np.random.choice(contract_types, n_samples, p=contract_probs)

    payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    payment_method = np.random.choice(payment_methods, n_samples)

    # Generate service features
    phone_service = np.random.choice([1, 0], n_samples, p=[0.9, 0.1])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.5, 0.2])
    online_security = np.random.choice([1, 0], n_samples, p=[0.4, 0.6])
    tech_support = np.random.choice([1, 0], n_samples, p=[0.3, 0.7])

    # Generate churn based on features
    churn_prob = (
        0.1 +  # Base churn rate
        (age < 30) * 0.1 +  # Younger customers more likely to churn
        (tenure < 12) * 0.2 +  # New customers more likely to churn
        (monthly_charges > 100) * 0.15 +  # High-cost customers more likely to churn
        (contract_type == 'Month-to-month') * 0.2 +  # Month-to-month more likely to churn
        (internet_service == 'Fiber optic') * 0.1 +  # Fiber optic more likely to churn
        (online_security == 0) * 0.1  # No security more likely to churn
    )

    churn_prob = np.clip(churn_prob, 0.05, 0.8)
    churn = np.random.binomial(1, churn_prob)

    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'age': age,
        'gender': gender,
        'senior_citizen': senior_citizen,
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'payment_method': payment_method,
        'phone_service': phone_service,
        'internet_service': internet_service,
        'online_security': online_security,
        'tech_support': tech_support,
        'churn': churn
    })

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info(f"Sample dataset saved to {save_path}")

    return df


if __name__ == "__main__":
    # Example usage
    preprocessor = ChurnDataPreprocessor()

    # Create sample dataset
    sample_df = create_sample_dataset(1000, "data/raw/customer_churn_sample.csv")

    # Load and preprocess data
    df = preprocessor.load_data("data/raw/customer_churn_sample.csv")

    # Validate data
    is_valid, errors = preprocessor.validate_data(df)
    if not is_valid:
        logger.error(f"Data validation failed: {errors}")

    # Clean data
    cleaned_df = preprocessor.clean_data(df)

    # Engineer features
    featured_df = preprocessor.engineer_features(cleaned_df)

    # Create preprocessing pipeline
    X, y = preprocessor.fit_transform(featured_df)

    print(f"Final feature shape: {X.shape}")
    print(f"Target distribution: {np.bincount(y)}")

    # Save preprocessor
    preprocessor.save_preprocessor("models/preprocessor.joblib")