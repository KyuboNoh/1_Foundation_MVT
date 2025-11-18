# Enhanced Metrics Extraction System

## Overview
The `save_training_results()` function in `train.py` has been enhanced to automatically extract, save, and display key training metrics for easy comparison across experiments.

## Features Added

### 1. **Automatic Metric Extraction** (`_extract_key_metrics`)
Extracts comprehensive metrics from training results:
- **Final Metrics**: accuracy, AUCPR, precision, recall, F1, loss
- **Per-Epoch Metrics**: train/val loss, train/val accuracy
- **PosDrop-Specific**: pos_accuracy, neg_accuracy, focus_score
- **Best Epoch Metrics**: automatically finds best epoch by validation loss
- **Early Stopping Info**: whether early stopping triggered, patience used
- **Configuration**: drop_rate, filter_top_pct, negs_per_pos

### 2. **Best Epoch Detection** (`_find_best_epoch_metrics`)
Automatically identifies the best training epoch based on validation loss and extracts:
- Best epoch number
- Validation loss/accuracy at best epoch
- PosDrop metrics (pos_acc, neg_acc, focus) at best epoch

### 3. **CSV Export** (`_save_metrics_csv`)
Saves all metrics to `all_experiments_metrics.csv` for:
- Easy comparison across experiments
- Quick filtering/sorting in Excel/pandas
- Time-series analysis of training runs

### 4. **Console Logging** (`_log_key_metrics`)
Displays formatted key metrics after each training run:
```
================================================================================
KEY METRICS FOR: base_cls1_iter0_drop0.2
================================================================================
✅ Training completed
Final Accuracy: 0.8543
Final AUCPR: 0.7821
Pos Accuracy: 0.9123
Neg Accuracy: 0.7965
Focus Score: 0.1158
Best Epoch: 42 (Val Loss: 0.2341)
Epochs: 50
Drop Rate: 0.2
Negs per Pos: 5
================================================================================
```

## Usage

The enhanced system works automatically - no code changes needed in your training loops:

```python
# Your existing code works as-is
result = train_and_save_result(
    cls, train_labels, train_loader, val_labels, val_loader,
    tag=f"{tag}_iter{rotation_i}_drop{drop_rate}",
    cfg=cfg, run_logger=run_logger, output_dir=output_dir, device=device
)

# save_training_results now automatically:
# 1. Extracts key metrics
# 2. Saves to JSON (full results)
# 3. Saves to CSV (for comparison)
# 4. Displays summary in console
save_training_results(result, tag, output_dir, run_logger)
```

## Output Files

### 1. Full Results: `{tag}_results.json`
Complete training data including:
- Model configuration
- Full epoch history
- Inference results
- **NEW**: `key_metrics` section with extracted metrics

### 2. Summary: `{tag}_summary.json`
Quick overview with:
- Tag, timestamp, model type
- Number of epochs
- **NEW**: `key_metrics` section
- Final metrics summary
- Early stopping info

### 3. Comparison CSV: `all_experiments_metrics.csv`
Aggregated metrics from all runs:
```csv
tag,extraction_timestamp,best_epoch_num,best_focus,best_neg_acc,best_pos_acc,...
base_cls1_iter0_drop0.2,2024-01-15T10:30:00,42,0.1158,0.7965,0.9123,...
base_cls1_iter1_drop0.2,2024-01-15T10:45:00,38,0.1245,0.8021,0.9087,...
```

## Comparing Experiments

### Quick Comparison (CSV)
```python
import pandas as pd

# Load all metrics
df = pd.read_csv('cls_1_training_results/all_experiments_metrics.csv')

# Compare by drop rate
df.groupby('drop_rate')[['final_aucpr', 'best_focus']].mean()

# Find best configuration
best = df.loc[df['final_aucpr'].idxmax()]
print(f"Best: {best['tag']} with AUCPR={best['final_aucpr']:.4f}")
```

### Detailed Analysis (JSON)
```python
import json

# Load specific experiment
with open('cls_1_training_results/base_cls1_iter0_drop0.2_results.json') as f:
    result = json.load(f)

# Access extracted metrics
metrics = result['key_metrics']
print(f"Pos Accuracy: {metrics['last_epoch_pos_acc']}")
print(f"Focus Score: {metrics['last_epoch_focus']}")
```

## Metrics Reference

| Metric | Source | Description |
|--------|--------|-------------|
| `final_accuracy` | metrics_summary | Final test accuracy |
| `final_aucpr` | metrics_summary | Area under precision-recall curve |
| `last_epoch_pos_acc` | epoch_history | Positive sample accuracy (last epoch) |
| `last_epoch_neg_acc` | epoch_history | Negative sample accuracy (last epoch) |
| `last_epoch_focus` | epoch_history | Focus score: pos_acc - neg_acc |
| `best_epoch_num` | epoch_history | Epoch with lowest validation loss |
| `best_val_loss` | epoch_history | Best validation loss achieved |
| `best_pos_acc` | epoch_history | Positive accuracy at best epoch |
| `best_focus` | epoch_history | Focus score at best epoch |
| `num_epochs_trained` | epoch_history | Total epochs before stopping |
| `early_stopped` | early_stopping_summary | Whether early stopping triggered |
| `drop_rate` | config | Positive dropout rate |
| `negs_per_pos` | config | Negative-to-positive ratio |

## Error Handling

All helper functions include try-except blocks:
- Extraction errors logged to `key_metrics['extraction_error']`
- Failed saves logged but don't crash training
- Graceful handling of missing/malformed data

## Implementation Details

### Location
- File: `Methods/1_UniversalGFM4MPM_v2_module/scripts/train.py`
- Main function: `save_training_results()` (lines ~870-1040)
- Helper functions: `_extract_key_metrics()`, `_find_best_epoch_metrics()`, `_save_metrics_csv()`, `_log_key_metrics()` (lines ~1044-1195)

### Design Principles
1. **Non-intrusive**: Works with existing training code
2. **Robust**: Handles missing data gracefully
3. **Extensible**: Easy to add new metrics
4. **Efficient**: Extraction happens after training completes

## Future Enhancements

Potential additions:
- [ ] Plot generation (training curves, metric comparisons)
- [ ] Statistical significance testing between runs
- [ ] Automatic hyperparameter tuning based on metrics
- [ ] Integration with experiment tracking tools (MLflow, Weights & Biases)
