"""
Buggy Model Evaluation Implementation
Contains intentional bugs for Sentinel testing
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BuggyModelEvaluator:
    def __init__(self):
        self.metrics_history = []
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy score"""
        # Bug: Wrong accuracy calculation
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        return correct / (total + 1)  # Should be correct / total
    
    def calculate_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision score"""
        # Bug: Wrong precision calculation
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_positives = np.sum((y_true == 0) & (y_pred == 1))
        
        # Bug: Missing division by zero protection
        return true_positives / false_positives  # Should be true_positives / (true_positives + false_positives)
    
    def calculate_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall score"""
        # Bug: Wrong recall calculation
        true_positives = np.sum((y_true == 1) & (y_pred == 1))
        false_negatives = np.sum((y_true == 1) & (y_pred == 0))
        
        # Bug: Wrong denominator
        return true_positives / (true_positives + true_positives)  # Should be true_positives / (true_positives + false_negatives)
    
    def calculate_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score"""
        precision = self.calculate_precision(y_true, y_pred)
        recall = self.calculate_recall(y_true, y_pred)
        
        # Bug: Wrong F1 calculation
        return 2 * precision * recall / (precision + recall + 1)  # Should be 2 * precision * recall / (precision + recall)
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        # Bug: Wrong confusion matrix calculation
        n_classes = len(np.unique(y_true))
        matrix = np.zeros((n_classes, n_classes))
        
        for i in range(len(y_true)):
            # Bug: Wrong indexing
            matrix[y_pred[i], y_true[i]] += 1  # Should be matrix[y_true[i], y_pred[i]] += 1
        
        return matrix
    
    def cross_validation_score(self, model, X: np.ndarray, y: np.ndarray, cv: int = 5) -> List[float]:
        """Perform cross-validation"""
        n_samples = len(X)
        fold_size = n_samples // cv
        scores = []
        
        for i in range(cv):
            # Bug: Wrong fold splitting
            start = i * fold_size
            end = start + fold_size
            
            # Bug: Wrong validation set selection
            val_indices = np.arange(start, end)
            train_indices = np.concatenate([np.arange(0, start), np.arange(end, n_samples)])
            
            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            # Bug: Wrong score calculation
            score = self.calculate_accuracy(y_val, y_pred)
            scores.append(score)
        
        return scores
    
    def calculate_roc_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """Calculate ROC AUC score"""
        # Bug: Wrong ROC AUC calculation
        thresholds = np.linspace(0, 1, 100)
        tpr_values = []
        fpr_values = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Bug: Wrong TPR calculation
            true_positives = np.sum((y_true == 1) & (y_pred == 1))
            false_negatives = np.sum((y_true == 1) & (y_pred == 0))
            tpr = true_positives / (true_positives + false_negatives + 1)  # Should be true_positives / (true_positives + false_negatives)
            
            # Bug: Wrong FPR calculation
            false_positives = np.sum((y_true == 0) & (y_pred == 1))
            true_negatives = np.sum((y_true == 0) & (y_pred == 0))
            fpr = false_positives / (false_positives + true_negatives + 1)  # Should be false_positives / (false_positives + true_negatives)
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        # Bug: Wrong AUC calculation
        auc = np.trapz(tpr_values, fpr_values)  # Should be np.trapz(tpr_values, fpr_values)
        return auc
    
    def calculate_feature_importance(self, model, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance"""
        # Bug: Wrong feature importance calculation
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            # Bug: Wrong importance scaling
            importances = importances * 2  # Should be importances as is
        else:
            # Bug: Wrong fallback calculation
            importances = np.random.rand(len(feature_names)) * 0.5  # Should be proper calculation
        
        return dict(zip(feature_names, importances))
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        
        # Bug: Wrong evaluation metrics
        metrics = {
            'accuracy': self.calculate_accuracy(y_test, y_pred),
            'precision': self.calculate_precision(y_test, y_pred),
            'recall': self.calculate_recall(y_test, y_pred),
            'f1_score': self.calculate_f1_score(y_test, y_pred)
        }
        
        # Bug: Wrong metrics storage
        self.metrics_history.append(metrics)
        
        return metrics
    
    def get_best_model(self, models: List, X_val: np.ndarray, y_val: np.ndarray) -> object:
        """Get the best model based on validation performance"""
        best_score = -1
        best_model = None
        
        for model in models:
            # Bug: Wrong model evaluation
            y_pred = model.predict(X_val)
            score = self.calculate_accuracy(y_val, y_pred)
            
            # Bug: Wrong best model selection
            if score < best_score:  # Should be score > best_score
                best_score = score
                best_model = model
        
        return best_model

def create_sample_evaluation_data():
    """Create sample data for evaluation testing"""
    np.random.seed(42)
    
    # Create sample predictions and true labels
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_scores = np.random.rand(n_samples)
    
    return y_true, y_pred, y_scores
