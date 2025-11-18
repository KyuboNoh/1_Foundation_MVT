"""
Example: How to use the enhanced metrics extraction system
"""
import json
import pandas as pd
from pathlib import Path

# ============================================================================
# EXAMPLE 1: Automatic metrics during training
# ============================================================================
def example_training_workflow():
    """Your existing training code works unchanged"""
    
    # After training completes, save_training_results automatically:
    # 1. Extracts key metrics (accuracy, AUCPR, focus, etc.)
    # 2. Saves full results to JSON
    # 3. Appends to CSV for comparison
    # 4. Logs summary to console
    
    result = {
        'tag': 'base_cls1_iter0_drop0.2',
        'metrics_summary': {
            'accuracy': 0.8543,
            'aucpr': 0.7821,
            'precision': 0.8234,
            'recall': 0.7654,
        },
        'epoch_history': [
            {'epoch': 1, 'train_loss': 0.45, 'val_loss': 0.42, 'pos_accuracy': 0.85, 'neg_accuracy': 0.75, 'focus_score': 0.10},
            {'epoch': 2, 'train_loss': 0.38, 'val_loss': 0.35, 'pos_accuracy': 0.88, 'neg_accuracy': 0.78, 'focus_score': 0.10},
            # ... more epochs
        ],
        'early_stopping_summary': {
            'early_stopped': True,
            'best_epoch': 42,
            'best_val_loss': 0.2341,
        }
    }
    
    # This call now does everything:
    # save_training_results(result, tag, output_dir, run_logger)
    
    print("✅ Metrics extracted, saved, and logged automatically!")


# ============================================================================
# EXAMPLE 2: Compare experiments using CSV
# ============================================================================
def compare_experiments(results_dir: Path):
    """Compare multiple experiments easily"""
    
    # Load aggregated metrics
    csv_path = results_dir / "all_experiments_metrics.csv"
    df = pd.read_csv(csv_path)
    
    # Compare different drop rates
    print("\n📊 Average AUCPR by Drop Rate:")
    print(df.groupby('drop_rate')['final_aucpr'].agg(['mean', 'std', 'count']))
    
    # Find best configuration
    best_idx = df['final_aucpr'].idxmax()
    best = df.loc[best_idx]
    print(f"\n🏆 Best Configuration:")
    print(f"  Tag: {best['tag']}")
    print(f"  AUCPR: {best['final_aucpr']:.4f}")
    print(f"  Focus: {best['best_focus']:.4f}")
    print(f"  Pos Acc: {best['best_pos_acc']:.4f}")
    
    # Compare base vs Method2 vs Method3
    df['method'] = df['tag'].str.extract(r'(base|Method2|Method3)')
    print("\n📈 Comparison by Method:")
    print(df.groupby('method')[['final_aucpr', 'best_focus']].mean())


# ============================================================================
# EXAMPLE 3: Analyze specific experiment
# ============================================================================
def analyze_single_experiment(results_dir: Path, tag: str):
    """Deep dive into a specific experiment"""
    
    # Load full results
    json_path = results_dir / f"{tag}_results.json"
    with open(json_path) as f:
        result = json.load(f)
    
    # Access extracted metrics
    metrics = result['key_metrics']
    
    print(f"\n🔍 Analysis of {tag}:")
    print(f"  Final AUCPR: {metrics.get('final_aucpr', 'N/A')}")
    print(f"  Best Epoch: {metrics.get('best_epoch_num', 'N/A')}")
    print(f"  Pos Accuracy (best): {metrics.get('best_pos_acc', 'N/A'):.4f}")
    print(f"  Focus Score (best): {metrics.get('best_focus', 'N/A'):.4f}")
    print(f"  Early Stopped: {metrics.get('early_stopped', False)}")
    
    # Access full epoch history for plotting
    epoch_history = result['epoch_history']
    print(f"  Total Epochs: {len(epoch_history)}")


# ============================================================================
# EXAMPLE 4: Monitor training progress
# ============================================================================
def monitor_training_progress(results_dir: Path):
    """Track recent experiments"""
    
    csv_path = results_dir / "all_experiments_metrics.csv"
    df = pd.read_csv(csv_path)
    
    # Sort by timestamp
    df['extraction_timestamp'] = pd.to_datetime(df['extraction_timestamp'])
    recent = df.sort_values('extraction_timestamp', ascending=False).head(5)
    
    print("\n🕒 Recent Experiments:")
    for _, row in recent.iterrows():
        print(f"  {row['tag']}")
        print(f"    AUCPR: {row['final_aucpr']:.4f}")
        print(f"    Focus: {row.get('best_focus', 'N/A')}")
        print(f"    Time: {row['extraction_timestamp']}")
        print()


# ============================================================================
# EXAMPLE 5: Feature combination comparison
# ============================================================================
def compare_feature_combinations(results_dir: Path):
    """Compare different feature concatenation strategies"""
    
    csv_path = results_dir / "all_experiments_metrics.csv"
    df = pd.read_csv(csv_path)
    
    # Extract feature method from tag
    # e.g., "base_cls1" vs "Method2_cls1" vs "Method3_cls1"
    df['features'] = df['tag'].str.extract(r'^(base|Method2|Method3)')[0]
    
    print("\n🧪 Feature Combination Comparison:")
    comparison = df.groupby('features').agg({
        'final_aucpr': ['mean', 'std'],
        'best_focus': ['mean', 'std'],
        'best_pos_acc': ['mean', 'std'],
        'num_epochs_trained': 'mean',
    }).round(4)
    print(comparison)
    
    # Statistical test
    from scipy import stats
    base = df[df['features'] == 'base']['final_aucpr']
    method2 = df[df['features'] == 'Method2']['final_aucpr']
    if len(base) > 1 and len(method2) > 1:
        t_stat, p_value = stats.ttest_ind(base, method2)
        print(f"\n📊 t-test (base vs Method2): p={p_value:.4f}")
        if p_value < 0.05:
            print("  ✅ Statistically significant difference!")


if __name__ == "__main__":
    # Example usage
    results_dir = Path("outputs/training_results")
    
    print("=" * 80)
    print("ENHANCED METRICS EXTRACTION - USAGE EXAMPLES")
    print("=" * 80)
    
    # These functions show how to use the extracted metrics
    # compare_experiments(results_dir)
    # analyze_single_experiment(results_dir, "base_cls1_iter0_drop0.2")
    # monitor_training_progress(results_dir)
    # compare_feature_combinations(results_dir)
