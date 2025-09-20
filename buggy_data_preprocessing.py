"""
Buggy Data Preprocessing Implementation
Contains intentional bugs for Sentinel testing
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler

class BuggyDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Bug: Wrong method for missing value handling
        if method == 'mean':
            # Bug: Should use numeric columns only
            return data.fillna(data.mean())  # Should be data.fillna(data.select_dtypes(include=[np.number]).mean())
        elif method == 'median':
            # Bug: Wrong median calculation
            return data.fillna(data.median() + 1)  # Should be data.fillna(data.median())
        else:
            return data.dropna()
    
    def normalize_features(self, data: np.ndarray) -> np.ndarray:
        """Normalize features using standardization"""
        # Bug: Wrong normalization formula
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Bug: Missing epsilon for division by zero
        return (data - mean) / std  # Should be (data - mean) / (std + 1e-8)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Split data into train and test sets"""
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        # Bug: Wrong shuffling - not random
        indices = np.arange(n_samples)  # Should be np.random.permutation(n_samples)
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def encode_categorical(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Encode categorical variables"""
        data_encoded = data.copy()
        
        for col in columns:
            if col in data_encoded.columns:
                # Bug: Wrong encoding - should use proper one-hot encoding
                unique_values = data_encoded[col].unique()
                for i, value in enumerate(unique_values):
                    data_encoded[f'{col}_{value}'] = (data_encoded[col] == value).astype(int)
                # Bug: Don't drop original column
                # data_encoded = data_encoded.drop(col, axis=1)
        
        return data_encoded
    
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outliers using Z-score method"""
        # Bug: Wrong outlier detection
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        # Bug: Wrong threshold comparison
        return data[z_scores < threshold + 1]  # Should be data[z_scores < threshold]
    
    def feature_scaling(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using StandardScaler"""
        # Bug: Wrong fit - should fit on training data only
        self.scaler.fit(X_test)  # Should be self.scaler.fit(X_train)
        self.is_fitted = True
        
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def create_polynomial_features(self, X: np.ndarray, degree: int = 2) -> np.ndarray:
        """Create polynomial features"""
        # Bug: Wrong polynomial feature creation
        n_features = X.shape[1]
        poly_features = X.copy()
        
        for d in range(2, degree + 1):
            for i in range(n_features):
                # Bug: Wrong polynomial calculation
                poly_features = np.column_stack([poly_features, X[:, i] ** (d - 1)])  # Should be X[:, i] ** d
        
        return poly_features
    
    def balance_dataset(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE-like approach"""
        # Bug: Wrong balancing - oversample majority class instead of minority
        unique_classes, counts = np.unique(y, return_counts=True)
        majority_class = unique_classes[np.argmax(counts)]
        minority_class = unique_classes[np.argmin(counts)]
        
        # Bug: Wrong oversampling strategy
        majority_indices = np.where(y.flatten() == majority_class)[0]
        minority_indices = np.where(y.flatten() == minority_class)[0]
        
        # Oversample majority class (wrong!)
        oversample_indices = np.random.choice(majority_indices, len(minority_indices), replace=True)
        
        X_balanced = np.vstack([X, X[oversample_indices]])
        y_balanced = np.vstack([y, y[oversample_indices]])
        
        return X_balanced, y_balanced

def create_sample_ml_data():
    """Create sample ML data for testing"""
    np.random.seed(42)
    
    # Create sample dataset
    n_samples = 1000
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
    
    # Add some missing values
    missing_indices = np.random.choice(n_samples, 50, replace=False)
    X[missing_indices, 0] = np.nan
    
    return X, y
