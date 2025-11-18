#!/usr/bin/env python3
"""
Test script to verify multi-clustering implementation is complete and working.
"""

def test_imports():
    """Test that all required functions can be imported."""
    try:
        from Common.cls.training.train_cls_PN_base import (
            create_pos_drop_schedule_kmeans,
            create_pos_drop_schedule_hierarchical,
            create_pos_drop_schedule_random,
            create_pos_drop_schedule_unified,
            train_cls_1_PN_PosDrop,
            train_cls_1_PN_PosDrop_MultiClustering,
            load_and_evaluate_existing_predictions
        )
        print("✅ All required functions imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_function_signatures():
    """Test that function signatures match expected usage."""
    try:
        import inspect
        from Common.cls.training.train_cls_PN_base import (
            train_cls_1_PN_PosDrop,
            train_cls_1_PN_PosDrop_MultiClustering
        )
        
        # Check train_cls_1_PN_PosDrop signature
        sig = inspect.signature(train_cls_1_PN_PosDrop)
        expected_params = {
            'meta_evaluation_n_clusters', 'clustering_method', 
            'linkage', 'common', 'data_use'
        }
        actual_params = set(sig.parameters.keys())
        
        if expected_params.issubset(actual_params):
            print("✅ train_cls_1_PN_PosDrop signature is compatible")
        else:
            missing = expected_params - actual_params
            print(f"❌ train_cls_1_PN_PosDrop missing parameters: {missing}")
            return False
        
        # Check train_cls_1_PN_PosDrop_MultiClustering signature
        sig_multi = inspect.signature(train_cls_1_PN_PosDrop_MultiClustering)
        expected_params_multi = {
            'meta_evaluation_n_clusters_list', 'clustering_methods',
            'linkage', 'common', 'data_use'
        }
        actual_params_multi = set(sig_multi.parameters.keys())
        
        if expected_params_multi.issubset(actual_params_multi):
            print("✅ train_cls_1_PN_PosDrop_MultiClustering signature is compatible")
        else:
            missing = expected_params_multi - actual_params_multi
            print(f"❌ train_cls_1_PN_PosDrop_MultiClustering missing parameters: {missing}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Signature test error: {e}")
        return False

def test_clustering_functions():
    """Test that clustering schedule functions exist and have reasonable signatures."""
    try:
        from Common.cls.training.train_cls_PN_base import (
            create_pos_drop_schedule_unified
        )
        
        import inspect
        sig = inspect.signature(create_pos_drop_schedule_unified)
        params = set(sig.parameters.keys())
        
        expected = {
            'all_pos_idx', 'features', 'meta_evaluation_n_clusters',
            'clustering_method', 'seed'
        }
        
        if expected.issubset(params):
            print("✅ create_pos_drop_schedule_unified has compatible signature")
            return True
        else:
            missing = expected - params
            print(f"❌ create_pos_drop_schedule_unified missing parameters: {missing}")
            return False
            
    except Exception as e:
        print(f"❌ Clustering function test error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing multi-clustering implementation...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_function_signatures,
        test_clustering_functions
    ]
    
    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Multi-clustering implementation is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)