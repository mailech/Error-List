"""
Test file for Buggy Data Preprocessing
These tests will fail due to intentional bugs
"""

import pytest
import numpy as np
import pandas as pd
from buggy_data_preprocessing import BuggyDataPreprocessor, create_sample_ml_data

def test_missing_value_handling():
    """Test missing value handling"""
    preprocessor = BuggyDataPreprocessor()
    
    # Create data with missing values
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [np.nan, 2, 3, 4, 5],
        'C': [1, 2, 3, 4, np.nan]
    })
    
    result = preprocessor.handle_missing_values(data, method='mean')
    
    # These tests will fail due to bugs
    assert not result.isnull().any().any()  # Will fail - wrong method for numeric columns
    assert result['A'].mean() == 3.0  # Will fail - wrong mean calculation

def test_normalization():
    """Test feature normalization"""
    preprocessor = BuggyDataPreprocessor()
    
    data = np.array([[1, 2], [3, 4], [5, 6]])
    result = preprocessor.normalize_features(data)
    
    # These tests will fail due to bugs
    assert np.allclose(result.mean(axis=0), 0)  # Will fail - missing epsilon
    assert np.allclose(result.std(axis=0), 1)  # Will fail - missing epsilon

def test_data_splitting():
    """Test data splitting"""
    preprocessor = BuggyDataPreprocessor()
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.4)
    
    # These tests will fail due to bugs
    assert len(X_train) == 3  # Will fail - wrong shuffling
    assert len(X_test) == 2  # Will fail - wrong shuffling
    assert not np.array_equal(X_train, X[:3])  # Will fail - no random shuffling

def test_categorical_encoding():
    """Test categorical encoding"""
    preprocessor = BuggyDataPreprocessor()
    
    data = pd.DataFrame({
        'category': ['A', 'B', 'A', 'C', 'B'],
        'value': [1, 2, 3, 4, 5]
    })
    
    result = preprocessor.encode_categorical(data, ['category'])
    
    # These tests will fail due to bugs
    assert 'category_A' in result.columns  # Will fail - original column not dropped
    assert 'category_B' in result.columns  # Will fail - original column not dropped
    assert 'category_C' in result.columns  # Will fail - original column not dropped

def test_outlier_removal():
    """Test outlier removal"""
    preprocessor = BuggyDataPreprocessor()
    
    # Create data with outliers
    data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9, 10])
    result = preprocessor.remove_outliers(data, threshold=2.0)
    
    # These tests will fail due to bugs
    assert len(result) < len(data)  # Will fail - wrong threshold
    assert 100 not in result  # Will fail - wrong outlier detection

def test_feature_scaling():
    """Test feature scaling"""
    preprocessor = BuggyDataPreprocessor()
    
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[7, 8], [9, 10]])
    
    X_train_scaled, X_test_scaled = preprocessor.feature_scaling(X_train, X_test)
    
    # These tests will fail due to bugs
    assert preprocessor.is_fitted  # Will fail - wrong fit data
    assert X_train_scaled.shape == X_train.shape  # Will fail - wrong scaling
    assert X_test_scaled.shape == X_test.shape  # Will fail - wrong scaling

def test_polynomial_features():
    """Test polynomial feature creation"""
    preprocessor = BuggyDataPreprocessor()
    
    X = np.array([[1, 2], [3, 4]])
    result = preprocessor.create_polynomial_features(X, degree=2)
    
    # These tests will fail due to bugs
    assert result.shape[1] > X.shape[1]  # Will fail - wrong polynomial calculation
    assert result.shape[1] == 5  # Will fail - wrong polynomial calculation

def test_dataset_balancing():
    """Test dataset balancing"""
    preprocessor = BuggyDataPreprocessor()
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 0, 0, 1, 1]).reshape(-1, 1)
    
    X_balanced, y_balanced = preprocessor.balance_dataset(X, y)
    
    # These tests will fail due to bugs
    unique_classes, counts = np.unique(y_balanced, return_counts=True)
    assert len(counts) == 2  # Will fail - wrong balancing strategy
    assert counts[0] == counts[1]  # Will fail - oversampling majority instead of minority

def test_sample_data_creation():
    """Test sample data creation"""
    X, y = create_sample_ml_data()
    
    # These tests will fail due to bugs
    assert X.shape == (1000, 5)  # Will fail - missing values not handled
    assert y.shape == (1000, 1)  # Will fail - missing values not handled
    assert not np.isnan(X).any()  # Will fail - missing values present

if __name__ == "__main__":
    pytest.main([__file__])
