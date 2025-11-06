"""
Save All Model Weights to .pkl Files
Saves trained model weights for:
1. XGBoost V1 (hierarchical)
2. XGBoost V2 (hierarchical)
3. LSTM V1 (hierarchical)
4. LSTM V2 (hierarchical)
"""

import os
import sys
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

def save_xgboost_v1_weights():
    """Save XGBoost V1 hierarchical model weights"""
    print("\n" + "=" * 80)
    print("SAVING XGBOOST V1 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/hierarchical/'
    
    # Stage 1 Model
    print("\n📦 Loading XGBoost V1 Stage 1 Model...")
    model_stage1 = xgb.XGBClassifier()
    model_stage1.load_model(os.path.join(base_dir, 'hierarchical_stage1_model.json'))
    print("✅ Stage 1 model loaded")
    
    # Save Stage 1 as .pkl
    output_file1 = os.path.join(base_dir, 'hierarchical_stage1_model.pkl')
    joblib.dump(model_stage1, output_file1)
    print(f"✅ Stage 1 model saved to: {output_file1}")
    print(f"   - Model type: {type(model_stage1).__name__}")
    print(f"   - Number of estimators: {model_stage1.n_estimators}")
    
    # Stage 2 Model
    print("\n📦 Loading XGBoost V1 Stage 2 Model...")
    model_stage2 = xgb.XGBClassifier()
    model_stage2.load_model(os.path.join(base_dir, 'hierarchical_stage2_model.json'))
    print("✅ Stage 2 model loaded")
    
    # Save Stage 2 as .pkl
    output_file2 = os.path.join(base_dir, 'hierarchical_stage2_model.pkl')
    joblib.dump(model_stage2, output_file2)
    print(f"✅ Stage 2 model saved to: {output_file2}")
    print(f"   - Model type: {type(model_stage2).__name__}")
    print(f"   - Number of estimators: {model_stage2.n_estimators}")
    
    return True

def save_xgboost_v2_weights():
    """Save XGBoost V2 hierarchical model weights"""
    print("\n" + "=" * 80)
    print("SAVING XGBOOST V2 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/v2_hierarchical/'
    
    # Models are already saved as .pkl files
    if os.path.exists(os.path.join(base_dir, 'xgboost_stage1.pkl')):
        print("✅ XGBoost V2 Stage 1 model already saved as .pkl")
        model_s1 = joblib.load(os.path.join(base_dir, 'xgboost_stage1.pkl'))
        print(f"   - Model type: {type(model_s1).__name__}")
        print(f"   - Number of estimators: {model_s1.n_estimators}")
    else:
        print("⚠️  XGBoost V2 Stage 1 model not found")
        return False
    
    if os.path.exists(os.path.join(base_dir, 'xgboost_stage2.pkl')):
        print("✅ XGBoost V2 Stage 2 model already saved as .pkl")
        model_s2 = joblib.load(os.path.join(base_dir, 'xgboost_stage2.pkl'))
        print(f"   - Model type: {type(model_s2).__name__}")
        print(f"   - Number of estimators: {model_s2.n_estimators}")
    else:
        print("⚠️  XGBoost V2 Stage 2 model not found")
        return False
    
    return True

def save_lstm_v1_weights():
    """Save LSTM V1 hierarchical model weights"""
    print("\n" + "=" * 80)
    print("SAVING LSTM V1 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/lstm/'
    
    # Stage 1 Model
    print("\n📦 Loading LSTM V1 Stage 1 Model...")
    h5_file1 = os.path.join(base_dir, 'lstm_hierarchical_stage1_model.h5')
    pkl_file1 = os.path.join(base_dir, 'lstm_hierarchical_stage1_model.pkl')
    
    if os.path.exists(h5_file1):
        model_stage1 = load_model(h5_file1)
        print("✅ Stage 1 model loaded from .h5")
        
        # Save as .pkl
        joblib.dump(model_stage1, pkl_file1)
        print(f"✅ Stage 1 model saved to: {pkl_file1}")
        print(f"   - Model type: {type(model_stage1).__name__}")
        print(f"   - Total parameters: {model_stage1.count_params():,}")
        print(f"   - Input shape: {model_stage1.input_shape}")
        print(f"   - Output shape: {model_stage1.output_shape}")
    else:
        print("⚠️  LSTM V1 Stage 1 model not found")
        return False
    
    # Stage 2 Model
    print("\n📦 Loading LSTM V1 Stage 2 Model...")
    h5_file2 = os.path.join(base_dir, 'lstm_hierarchical_stage2_model.h5')
    pkl_file2 = os.path.join(base_dir, 'lstm_hierarchical_stage2_model.pkl')
    
    if os.path.exists(h5_file2):
        model_stage2 = load_model(h5_file2)
        print("✅ Stage 2 model loaded from .h5")
        
        # Save as .pkl
        joblib.dump(model_stage2, pkl_file2)
        print(f"✅ Stage 2 model saved to: {pkl_file2}")
        print(f"   - Model type: {type(model_stage2).__name__}")
        print(f"   - Total parameters: {model_stage2.count_params():,}")
        print(f"   - Input shape: {model_stage2.input_shape}")
        print(f"   - Output shape: {model_stage2.output_shape}")
    else:
        print("⚠️  LSTM V1 Stage 2 model not found")
        return False
    
    return True

def save_lstm_v2_weights():
    """Save LSTM V2 hierarchical model weights"""
    print("\n" + "=" * 80)
    print("SAVING LSTM V2 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/v2_lstm/'
    
    # Stage 1 Model
    print("\n📦 Loading LSTM V2 Stage 1 Model...")
    h5_file1 = os.path.join(base_dir, 'lstm_stage1_model.h5')
    pkl_file1 = os.path.join(base_dir, 'lstm_stage1_model.pkl')
    
    if os.path.exists(h5_file1):
        model_stage1 = load_model(h5_file1)
        print("✅ Stage 1 model loaded from .h5")
        
        # Save as .pkl
        joblib.dump(model_stage1, pkl_file1)
        print(f"✅ Stage 1 model saved to: {pkl_file1}")
        print(f"   - Model type: {type(model_stage1).__name__}")
        print(f"   - Total parameters: {model_stage1.count_params():,}")
        print(f"   - Input shape: {model_stage1.input_shape}")
        print(f"   - Output shape: {model_stage1.output_shape}")
    else:
        print("⚠️  LSTM V2 Stage 1 model not found")
        return False
    
    # Stage 2 Model
    print("\n📦 Loading LSTM V2 Stage 2 Model...")
    h5_file2 = os.path.join(base_dir, 'lstm_stage2_model.h5')
    pkl_file2 = os.path.join(base_dir, 'lstm_stage2_model.pkl')
    
    if os.path.exists(h5_file2):
        model_stage2 = load_model(h5_file2)
        print("✅ Stage 2 model loaded from .h5")
        
        # Save as .pkl
        joblib.dump(model_stage2, pkl_file2)
        print(f"✅ Stage 2 model saved to: {pkl_file2}")
        print(f"   - Model type: {type(model_stage2).__name__}")
        print(f"   - Total parameters: {model_stage2.count_params():,}")
        print(f"   - Input shape: {model_stage2.input_shape}")
        print(f"   - Output shape: {model_stage2.output_shape}")
    else:
        print("⚠️  LSTM V2 Stage 2 model not found")
        return False
    
    return True

def main():
    """Save all model weights"""
    print("\n" + "=" * 80)
    print("SAVING ALL MODEL WEIGHTS TO .PKL FILES")
    print("=" * 80)
    print("\nThis script will save weights for:")
    print("1. XGBoost V1 (hierarchical)")
    print("2. XGBoost V2 (hierarchical)")
    print("3. LSTM V1 (hierarchical)")
    print("4. LSTM V2 (hierarchical)")
    print("=" * 80)
    
    results = {
        'xgboost_v1': False,
        'xgboost_v2': False,
        'lstm_v1': False,
        'lstm_v2': False
    }
    
    # Save XGBoost V1
    try:
        results['xgboost_v1'] = save_xgboost_v1_weights()
    except Exception as e:
        print(f"❌ Error saving XGBoost V1: {e}")
    
    # Save XGBoost V2
    try:
        results['xgboost_v2'] = save_xgboost_v2_weights()
    except Exception as e:
        print(f"❌ Error saving XGBoost V2: {e}")
    
    # Save LSTM V1
    try:
        results['lstm_v1'] = save_lstm_v1_weights()
    except Exception as e:
        print(f"❌ Error saving LSTM V1: {e}")
    
    # Save LSTM V2
    try:
        results['lstm_v2'] = save_lstm_v2_weights()
    except Exception as e:
        print(f"❌ Error saving LSTM V2: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{model_name.upper():<20} {status}")
    
    total_success = sum(results.values())
    print(f"\n{total_success}/4 models saved successfully")
    
    print("\n" + "=" * 80)
    print("✅ MODEL WEIGHT SAVING COMPLETE")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()

