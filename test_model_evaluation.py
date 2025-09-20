"""
Test file for Buggy Model Evaluation
These tests will fail due to intentional bugs
"""

import pytest
import numpy as np
from buggy_model_evaluation import BuggyModelEvaluator, create_sample_evaluation_data

def test_accuracy_calculation():
    """Test accuracy calculation"""
    evaluator = BuggyModelEvaluator()
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    accuracy = evaluator.calculate_accuracy(y_true, y_pred)
    
    # This test will fail due to bug
    expected_accuracy = 4 / 5  # Should be 0.8
    assert abs(accuracy - expected_accuracy) < 1e-6  # Will fail - wrong calculation

def test_precision_calculation():
    """Test precision calculation"""
    evaluator = BuggyModelEvaluator()
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    precision = evaluator.calculate_precision(y_true, y_pred)
    
    # This test will fail due to bug
    expected_precision = 2 / (2 + 0)  # Should be 1.0
    assert abs(precision - expected_precision) < 1e-6  # Will fail - wrong calculation

def test_recall_calculation():
    """Test recall calculation"""
    evaluator = BuggyModelEvaluator()
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    recall = evaluator.calculate_recall(y_true, y_pred)
    
    # This test will fail due to bug
    expected_recall = 2 / (2 + 1)  # Should be 0.667
    assert abs(recall - expected_recall) < 1e-6  # Will fail - wrong calculation

def test_f1_score_calculation():
    """Test F1 score calculation"""
    evaluator = BuggyModelEvaluator()
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    f1 = evaluator.calculate_f1_score(y_true, y_pred)
    
    # This test will fail due to bug
    precision = evaluator.calculate_precision(y_true, y_pred)
    recall = evaluator.calculate_recall(y_true, y_pred)
    expected_f1 = 2 * precision * recall / (precision + recall)
    assert abs(f1 - expected_f1) < 1e-6  # Will fail - wrong calculation

def test_confusion_matrix():
    """Test confusion matrix calculation"""
    evaluator = BuggyModelEvaluator()
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    
    matrix = evaluator.calculate_confusion_matrix(y_true, y_pred)
    
    # These tests will fail due to bugs
    assert matrix.shape == (2, 2)  # Will fail - wrong indexing
    assert matrix[0, 0] == 2  # Will fail - wrong indexing (should be TN)
    assert matrix[1, 1] == 2  # Will fail - wrong indexing (should be TP)

def test_cross_validation():
    """Test cross-validation"""
    evaluator = BuggyModelEvaluator()
    
    # Mock model class
    class MockModel:
        def fit(self, X, y):
            pass
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
    
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    model = MockModel()
    
    scores = evaluator.cross_validation_score(model, X, y, cv=5)
    
    # These tests will fail due to bugs
    assert len(scores) == 5  # Will fail - wrong fold splitting
    assert all(0 <= score <= 1 for score in scores)  # Will fail - wrong score calculation

def test_roc_auc_calculation():
    """Test ROC AUC calculation"""
    evaluator = BuggyModelEvaluator()
    
    y_true = np.array([1, 0, 1, 0, 1])
    y_scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
    
    auc = evaluator.calculate_roc_auc(y_true, y_scores)
    
    # This test will fail due to bug
    assert 0 <= auc <= 1  # Will fail - wrong AUC calculation

def test_feature_importance():
    """Test feature importance calculation"""
    evaluator = BuggyModelEvaluator()
    
    # Mock model with feature importances
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.array([0.3, 0.5, 0.2])
    
    model = MockModel()
    feature_names = ['feature1', 'feature2', 'feature3']
    
    importance = evaluator.calculate_feature_importance(model, feature_names)
    
    # These tests will fail due to bugs
    assert len(importance) == 3  # Will fail - wrong importance scaling
    assert sum(importance.values()) == 1.0  # Will fail - wrong importance scaling

def test_model_evaluation():
    """Test comprehensive model evaluation"""
    evaluator = BuggyModelEvaluator()
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
    
    model = MockModel()
    X_test = np.random.randn(100, 5)
    y_test = np.random.randint(0, 2, 100)
    
    metrics = evaluator.evaluate_model(model, X_test, y_test)
    
    # These tests will fail due to bugs
    assert 'accuracy' in metrics  # Will fail - wrong evaluation
    assert 'precision' in metrics  # Will fail - wrong evaluation
    assert 'recall' in metrics  # Will fail - wrong evaluation
    assert 'f1_score' in metrics  # Will fail - wrong evaluation

def test_best_model_selection():
    """Test best model selection"""
    evaluator = BuggyModelEvaluator()
    
    # Mock models
    class MockModel:
        def __init__(self, accuracy):
            self.accuracy = accuracy
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
    
    models = [MockModel(0.8), MockModel(0.9), MockModel(0.7)]
    X_val = np.random.randn(50, 5)
    y_val = np.random.randint(0, 2, 50)
    
    best_model = evaluator.get_best_model(models, X_val, y_val)
    
    # This test will fail due to bug
    assert best_model is not None  # Will fail - wrong best model selection

def test_sample_data_creation():
    """Test sample data creation"""
    y_true, y_pred, y_scores = create_sample_evaluation_data()
    
    # These tests will fail due to bugs
    assert len(y_true) == 1000  # Will fail - wrong data creation
    assert len(y_pred) == 1000  # Will fail - wrong data creation
    assert len(y_scores) == 1000  # Will fail - wrong data creation
    assert np.all(y_true >= 0) and np.all(y_true <= 1)  # Will fail - wrong data creation

if __name__ == "__main__":
    pytest.main([__file__])
