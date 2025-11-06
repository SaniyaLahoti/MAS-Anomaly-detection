"""
Save XGBoost Model Weights as .pkl files
Loads the trained hierarchical XGBoost models and saves them as reusable .pkl files
"""

import xgboost as xgb
import joblib
import os

def save_model_weights():
    """
    Load XGBoost models and save them as .pkl files
    """
    base_dir = '../models/hierarchical/'
    
    print("=" * 80)
    print("SAVING XGBOOST MODEL WEIGHTS AS .PKL FILES")
    print("=" * 80)
    
    # Stage 1 Model
    print("\n📦 Loading Stage 1 Model...")
    model_stage1 = xgb.XGBClassifier()
    model_stage1.load_model(os.path.join(base_dir, 'hierarchical_stage1_model.json'))
    print("✅ Stage 1 model loaded")
    
    # Save Stage 1 as .pkl
    output_file1 = os.path.join(base_dir, 'hierarchical_stage1_model.pkl')
    joblib.dump(model_stage1, output_file1)
    print(f"✅ Stage 1 model saved to: {output_file1}")
    print(f"   - Model type: {type(model_stage1)}")
    print(f"   - Number of estimators: {model_stage1.n_estimators}")
    print(f"   - Max depth: {model_stage1.max_depth}")
    
    # Stage 2 Model
    print("\n📦 Loading Stage 2 Model...")
    model_stage2 = xgb.XGBClassifier()
    model_stage2.load_model(os.path.join(base_dir, 'hierarchical_stage2_model.json'))
    print("✅ Stage 2 model loaded")
    
    # Save Stage 2 as .pkl
    output_file2 = os.path.join(base_dir, 'hierarchical_stage2_model.pkl')
    joblib.dump(model_stage2, output_file2)
    print(f"✅ Stage 2 model saved to: {output_file2}")
    print(f"   - Model type: {type(model_stage2)}")
    print(f"   - Number of estimators: {model_stage2.n_estimators}")
    print(f"   - Max depth: {model_stage2.max_depth}")
    
    print("\n" + "=" * 80)
    print("✅ MODEL WEIGHTS SAVED SUCCESSFULLY AS .PKL FILES")
    print("=" * 80)
    print(f"\n📁 Saved files:")
    print(f"   - {output_file1}")
    print(f"   - {output_file2}")
    print("\n💡 These .pkl files can be loaded using:")
    print("   model = joblib.load('hierarchical_stage1_model.pkl')")
    print("=" * 80)

if __name__ == "__main__":
    save_model_weights()
