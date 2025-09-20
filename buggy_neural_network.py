"""
Buggy Neural Network Implementation
Contains intentional bugs for Sentinel testing
"""

import numpy as np
from typing import List, Tuple

class BuggyNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Bug: Wrong weight initialization - should be random, not zeros
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        # Bug: Missing overflow protection
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x: np.ndarray) -> np.ndarray:
        # Bug: Wrong derivative formula
        return x * (1 - x) * (1 - x)  # Should be x * (1 - x)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Bug: Missing bias addition
        self.z1 = np.dot(X, self.W1)  # Should be np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Bug: Missing bias addition
        self.z2 = np.dot(self.a1, self.W2)  # Should be np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2, self.a1
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, hidden: np.ndarray):
        m = X.shape[0]
        
        # Bug: Wrong cost derivative
        dz2 = output - y  # Should be (output - y) / m
        
        # Bug: Wrong gradient calculation
        dW2 = np.dot(hidden.T, dz2)  # Should be np.dot(hidden.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True)  # Should be np.sum(dz2, axis=0, keepdims=True) / m
        
        # Bug: Wrong backpropagation
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(hidden)
        dW1 = np.dot(X.T, dz1)  # Should be np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True)  # Should be np.sum(dz1, axis=0, keepdims=True) / m
        
        return dW1, db1, dW2, db2
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, learning_rate: float = 0.1):
        # Bug: Wrong learning rate - too high
        learning_rate = 10.0  # Should be 0.1
        
        for epoch in range(epochs):
            # Forward pass
            output, hidden = self.forward(X)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y, output, hidden)
            
            # Bug: Wrong weight update - missing learning rate
            self.W1 -= dW1  # Should be self.W1 -= learning_rate * dW1
            self.b1 -= db1  # Should be self.b1 -= learning_rate * db1
            self.W2 -= dW2  # Should be self.W2 -= learning_rate * dW2
            self.b2 -= db2  # Should be self.b2 -= learning_rate * db2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        output, _ = self.forward(X)
        # Bug: Wrong prediction threshold
        return (output > 0.8).astype(int)  # Should be (output > 0.5).astype(int)

def create_sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)
    return X, y
