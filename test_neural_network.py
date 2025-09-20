"""
Test file for Buggy Neural Network
These tests will fail due to intentional bugs
"""

import pytest
import numpy as np
from buggy_neural_network import BuggyNeuralNetwork, create_sample_data

def test_neural_network_initialization():
    """Test neural network initialization"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    
    # These tests will fail due to bugs
    assert nn.W1.shape == (2, 3)
    assert nn.W2.shape == (3, 1)
    assert not np.all(nn.W1 == 0)  # Will fail - weights are zeros instead of random
    assert not np.all(nn.W2 == 0)  # Will fail - weights are zeros instead of random

def test_sigmoid_function():
    """Test sigmoid function"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    
    # Test sigmoid with normal values
    x = np.array([0, 1, -1])
    result = nn.sigmoid(x)
    
    # These tests will fail due to bugs
    assert np.all(result >= 0) and np.all(result <= 1)  # Will fail - no overflow protection
    assert result[0] == 0.5  # Will fail - sigmoid(0) should be 0.5

def test_sigmoid_derivative():
    """Test sigmoid derivative"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    
    x = np.array([0.5])
    result = nn.sigmoid_derivative(x)
    
    # This test will fail due to bug
    expected = 0.5 * (1 - 0.5)  # Should be 0.25
    assert abs(result[0] - expected) < 1e-6  # Will fail - wrong derivative formula

def test_forward_pass():
    """Test forward pass"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    X = np.array([[1, 2], [3, 4]])
    
    output, hidden = nn.forward(X)
    
    # These tests will fail due to bugs
    assert output.shape == (2, 1)  # Will fail - missing bias
    assert hidden.shape == (2, 3)  # Will fail - missing bias
    assert not np.all(output == 0.5)  # Will fail - all outputs are 0.5 due to zero weights

def test_backward_pass():
    """Test backward pass"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    X = np.array([[1, 2]])
    y = np.array([[1]])
    
    output, hidden = nn.forward(X)
    dW1, db1, dW2, db2 = nn.backward(X, y, output, hidden)
    
    # These tests will fail due to bugs
    assert dW1.shape == (2, 3)  # Will fail - wrong gradient calculation
    assert dW2.shape == (3, 1)  # Will fail - wrong gradient calculation
    assert not np.all(dW1 == 0)  # Will fail - gradients are wrong

def test_training():
    """Test training process"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    X, y = create_sample_data()
    
    # Train the network
    nn.train(X, y, epochs=10, learning_rate=0.1)
    
    # These tests will fail due to bugs
    assert not np.all(nn.W1 == 0)  # Will fail - weights not updated properly
    assert not np.all(nn.W2 == 0)  # Will fail - weights not updated properly
    
    # Test prediction
    predictions = nn.predict(X[:5])
    assert predictions.shape == (5, 1)  # Will fail - wrong prediction threshold

def test_prediction():
    """Test prediction functionality"""
    nn = BuggyNeuralNetwork(2, 3, 1)
    X = np.array([[0.5, 0.5]])
    
    prediction = nn.predict(X)
    
    # This test will fail due to bug
    assert prediction.shape == (1, 1)  # Will fail - wrong threshold
    assert prediction[0, 0] in [0, 1]  # Will fail - wrong threshold

if __name__ == "__main__":
    pytest.main([__file__])
