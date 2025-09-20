# Error-List: Machine Learning Buggy Code

This folder contains intentionally buggy Machine Learning code for testing the Self-Healing Codebase Sentinel system.

## Files Overview

### Buggy ML Modules
- **`buggy_neural_network.py`** - Neural network implementation with bugs in:
  - Weight initialization (zeros instead of random)
  - Sigmoid function (no overflow protection)
  - Sigmoid derivative (wrong formula)
  - Forward pass (missing bias)
  - Backward pass (wrong gradients)
  - Training (wrong learning rate and updates)
  - Prediction (wrong threshold)

- **`buggy_data_preprocessing.py`** - Data preprocessing with bugs in:
  - Missing value handling (wrong method)
  - Normalization (missing epsilon)
  - Data splitting (no shuffling)
  - Categorical encoding (wrong approach)
  - Outlier removal (wrong threshold)
  - Feature scaling (wrong fit data)
  - Polynomial features (wrong calculation)
  - Dataset balancing (oversampling majority)

- **`buggy_model_evaluation.py`** - Model evaluation with bugs in:
  - Accuracy calculation (wrong formula)
  - Precision calculation (wrong denominator)
  - Recall calculation (wrong denominator)
  - F1 score calculation (wrong formula)
  - Confusion matrix (wrong indexing)
  - Cross-validation (wrong splitting)
  - ROC AUC (wrong calculation)
  - Feature importance (wrong scaling)
  - Best model selection (wrong comparison)

### Test Files
- **`test_neural_network.py`** - Tests for neural network bugs
- **`test_data_preprocessing.py`** - Tests for preprocessing bugs
- **`test_model_evaluation.py`** - Tests for evaluation bugs

## Expected Sentinel Behavior

When these files are pushed to GitHub and tests are run:

1. **CI Pipeline Fails** - Tests will fail due to intentional bugs
2. **Sentinel Detects Failure** - Webhook triggers Sentinel
3. **AI Analysis** - Sentinel analyzes the ML code bugs
4. **Patch Generation** - Creates fixes for ML algorithms
5. **Pull Request Creation** - Automated PR with ML bug fixes
6. **Validation** - Tests pass after fixes are applied

## ML Bug Categories

### Neural Network Bugs
- Initialization errors
- Activation function bugs
- Gradient calculation errors
- Training loop issues
- Prediction threshold problems

### Data Preprocessing Bugs
- Missing value handling errors
- Normalization issues
- Feature engineering bugs
- Data splitting problems
- Scaling errors

### Model Evaluation Bugs
- Metric calculation errors
- Cross-validation issues
- Performance measurement bugs
- Model selection problems

## Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (will fail)
pytest test_*.py -v

# Run specific test file
pytest test_neural_network.py -v
pytest test_data_preprocessing.py -v
pytest test_model_evaluation.py -v
```

## Sentinel Integration

This code is designed to work with the Self-Healing Codebase Sentinel system:

1. Push these files to your GitHub repository
2. Ensure CI pipeline is configured
3. Run tests to trigger failures
4. Watch Sentinel automatically fix the ML bugs
5. Review the generated Pull Requests with fixes

## Note

All bugs in this code are **intentional** and designed to test the Sentinel system's ability to:
- Detect ML-specific bugs
- Generate appropriate fixes
- Handle complex algorithm corrections
- Maintain code quality in ML pipelines
