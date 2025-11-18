# Meta-Evaluation System for PN Classifiers

## Overview

The meta-evaluation system provides a modular framework for computing performance metrics on positive-dropout cross-validation. This system is implemented in `train_cls_PN_base.py`.

## Architecture

### Key Components

1. **Metric Functions**: Individual functions that compute specific metrics
2. **Function Registry**: `META_EVALUATION_FUNCTIONS` dictionary mapping metric names to functions
3. **Dispatcher**: `compute_meta_evaluation_metric()` that routes to appropriate functions
4. **Training Integration**: Automatic computation during `train_cls_1_PN_PosDrop()`
5. **Cached Evaluation**: Load and evaluate from saved predictions via `load_and_evaluate_existing_predictions()`

## Built-in Metrics

### PosDrop_Acc (Positive Drop Accuracy)
**Purpose**: Measures how well the model predicts on positives that were dropped during training.

**Formula**: `mean(predictions[dropped_positives])`

**Interpretation**: 
- Higher is better (range: 0.0 to 1.0)
- Evaluates cross-validation performance
- Indicates generalization to unseen positives

**Function**: `compute_posdrop_acc()`

**Required Parameters**:
- `predictions_mean`: Array of mean predictions
- `all_pos_idx`: List of all positive sample indices
- `pos_indices_this_iter`: Set of positives used in training

---

### Focus
**Purpose**: Measures model selectivity/sparsity - how focused the predictions are.

**Formula**: `1.0 - (sum(predictions) / total_predictions)`

**Interpretation**:
- Higher is better (range: 0.0 to 1.0)
- Higher focus = more selective predictions
- Lower focus = model predicts more areas as positive

**Function**: `compute_focus()`

**Required Parameters**:
- `predictions_mean`: Array of mean predictions

---

## Adding New Metrics

### Step 1: Create the Metric Function

Create a new function in `train_cls_PN_base.py` following this template:

```python
def compute_your_metric_name(
    *,
    predictions_mean: np.ndarray,
    predictions_std: Optional[np.ndarray] = None,
    all_pos_idx: Optional[List[int]] = None,
    pos_indices_this_iter: Optional[set] = None,
    # Add any other parameters you need
) -> float:
    """
    Compute YourMetricName: Brief description of what it measures.
    
    Formula explanation and interpretation.
    
    Args:
        predictions_mean: Mean predictions array (always provided)
        predictions_std: Standard deviation predictions (optional)
        all_pos_idx: List of all positive indices (optional)
        pos_indices_this_iter: Set of positives used in training (optional)
    
    Returns:
        Metric value as float
    """
    # Your computation logic here
    result = ...  # compute your metric
    return float(result)
```

### Step 2: Register the Function

Add your function to the `META_EVALUATION_FUNCTIONS` registry:

```python
META_EVALUATION_FUNCTIONS: Dict[str, Callable] = {
    "PosDrop_Acc": compute_posdrop_acc,
    "Focus": compute_focus,
    "YourMetricName": compute_your_metric_name,  # Add this line
}
```

### Step 3: Update the Dispatcher (if needed)

If your metric requires special parameters, update `compute_meta_evaluation_metric()`:

```python
def compute_meta_evaluation_metric(...):
    # ... existing code ...
    
    if metric_name == "YourMetricName":
        if some_required_param is None:
            raise ValueError(f"Metric '{metric_name}' requires some_required_param")
        kwargs["some_required_param"] = some_required_param
    
    # ... rest of the code ...
```

### Step 4: Update the Default Set (optional)

To include your metric by default:

```python
DEFAULT_META_EVALUATION = {"PosDrop_Acc", "Focus", "YourMetricName"}
```

### Step 5: Use Your Metric

```bash
# Command line usage
python -m Methods.1_UniversalGFM4MPM_v2_module.scripts.train \
    --config ./config.json \
    --meta-evaluation PosDrop_Acc Focus YourMetricName
```

## Example: Adding Uncertainty-Based Metric

Here's a complete example of adding a new metric that uses prediction uncertainty:

```python
# Step 1: Create the function
def compute_uncertainty_score(
    *,
    predictions_mean: np.ndarray,
    predictions_std: np.ndarray,
) -> float:
    """
    Compute UncertaintyScore: Average prediction uncertainty across all samples.
    
    Uses standard deviation of Monte Carlo dropout predictions.
    Lower uncertainty indicates more confident predictions.
    
    Args:
        predictions_mean: Mean predictions array
        predictions_std: Standard deviation of predictions
    
    Returns:
        Average uncertainty score
    """
    avg_uncertainty = float(np.mean(predictions_std))
    return avg_uncertainty


# Step 2: Register it
META_EVALUATION_FUNCTIONS["UncertaintyScore"] = compute_uncertainty_score

# Step 3: Update dispatcher (if predictions_std needs special handling)
def compute_meta_evaluation_metric(...):
    # ... existing code ...
    
    if metric_name == "UncertaintyScore":
        if predictions_std is None:
            raise ValueError(f"Metric '{metric_name}' requires predictions_std")
        kwargs["predictions_std"] = predictions_std
    
    # ... rest of the code ...

# Step 4: Use it
# python ... --meta-evaluation PosDrop_Acc Focus UncertaintyScore
```

## API Reference

### compute_meta_evaluation_metric()
Central dispatcher that computes any registered metric.

**Parameters**:
- `metric_name` (str): Name of metric to compute
- `predictions_mean` (np.ndarray): Mean predictions (always required)
- `predictions_std` (np.ndarray, optional): Prediction uncertainty
- `all_pos_idx` (List[int], optional): All positive sample indices
- `pos_indices_this_iter` (set, optional): Positives used in this iteration

**Returns**: float - Computed metric value

**Raises**: ValueError if metric is unknown or required parameters are missing

---

### META_EVALUATION_FUNCTIONS
Dictionary registry mapping metric names to their computation functions.

**Type**: `Dict[str, Callable]`

**Usage**:
```python
# Check available metrics
available_metrics = list(META_EVALUATION_FUNCTIONS.keys())

# Get a specific function
posdrop_func = META_EVALUATION_FUNCTIONS["PosDrop_Acc"]
```

---

## Best Practices

1. **Naming**: Use descriptive PascalCase names (e.g., `PosDrop_Acc`, `Focus`)
2. **Docstrings**: Include formula, interpretation, and parameter descriptions
3. **Type Hints**: Use proper type annotations for all parameters
4. **Return Type**: Always return `float` (not numpy types)
5. **Error Handling**: Raise `ValueError` with clear messages for missing requirements
6. **Defaults**: Use `Optional` for parameters that aren't always needed
7. **Keyword-only**: Use `*` to make all parameters keyword-only for clarity

## Testing New Metrics

```python
# Test your metric function directly
predictions_mean = np.array([0.1, 0.8, 0.3, 0.9])
result = compute_your_metric_name(predictions_mean=predictions_mean)
print(f"Test result: {result}")

# Test through the dispatcher
result = compute_meta_evaluation_metric(
    metric_name="YourMetricName",
    predictions_mean=predictions_mean
)
print(f"Dispatcher result: {result}")
```

## Common Parameter Patterns

### Always Available
- `predictions_mean`: Mean predictions from Monte Carlo dropout

### Often Available  
- `predictions_std`: Uncertainty estimates from MC dropout
- `all_pos_idx`: Indices of all positive samples
- `pos_indices_this_iter`: Which positives were used in training

### Iteration Context
- Can access dropped positives: `[idx for idx in all_pos_idx if idx not in pos_indices_this_iter]`
- Can access trained positives: `pos_indices_this_iter`

## Command-Line Usage

```bash
# Use default metrics (PosDrop_Acc, Focus)
python -m Methods.1_UniversalGFM4MPM_v2_module.scripts.train --config ./config.json

# Specify custom metrics
python -m Methods.1_UniversalGFM4MPM_v2_module.scripts.train \
    --config ./config.json \
    --meta-evaluation PosDrop_Acc Focus CustomMetric

# Use only one metric
python -m Methods.1_UniversalGFM4MPM_v2_module.scripts.train \
    --config ./config.json \
    --meta-evaluation Focus
```

## File Locations

- **Main module**: `Common/cls/training/train_cls_PN_base.py`
- **Training script**: `Methods/1_UniversalGFM4MPM_v2_module/scripts/train.py`
- **Results**: `{output_dir}/meta_evaluation_results/{tag}_meta_evaluation.json`

## Migration Notes

The old `train_cls_1.py` has been moved and refactored to `Common/cls/training/train_cls_PN_base.py` with the following improvements:

1. ✅ Modular metric functions with clear separation of concerns
2. ✅ Function registry for easy metric addition
3. ✅ Central dispatcher with parameter validation
4. ✅ Support for both training-time and cached evaluation
5. ✅ Command-line configurable metric selection
6. ✅ Better documentation and error messages
