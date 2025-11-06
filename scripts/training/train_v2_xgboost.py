"""
Train Hierarchical XGBoost Model on V2 Dataset
Reuses the V1 training structure with V2 preprocessed data
"""

import sys
import os
import numpy as np
import joblib

# Import the V1 training functions (they are dataset-agnostic)
from hierarchical_classification import (
    load_and_engineer_features,
    create_stage1_labels,
    balance_dataset_stage1,
    prepare_stage2_data,
    train_stage1_model,
    train_stage2_model
)

def main():
    """
    Train hierarchical XGBoost on V2 dataset
    """
    print("\n" + "=" * 80)
    print("TRAINING HIERARCHICAL XGBOOST ON V2 DATASET")
    print("=" * 80)
    print("\nUsing preprocessed V2 data: NF-BoT-IoT-v2-preprocessed.csv")
    print("Same architecture as V1: Stage 1 (4-class) + Stage 2 (DDoS vs DoS)")
    print("=" * 80 + "\n")
    
    # Paths for V2
    input_file = "../../datasets/v2_dataset/NF-BoT-IoT-v2-preprocessed.csv"
    output_dir = "../../models/v2_hierarchical/"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Stage 1: 4-class classification
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING 4-CLASS CLASSIFIER")
    print("=" * 80)
    
    # Load and engineer features
    df = load_and_engineer_features(input_file)
    
    # Create Stage 1 labels
    df_stage1 = create_stage1_labels(df)
    
    # Balance dataset
    df_stage1_balanced = balance_dataset_stage1(df_stage1, target_size=25000)
    
    # Train Stage 1 model
    model_stage1, scaler_stage1, le_stage1, stage1_cv_results = train_stage1_model(df_stage1_balanced)
    
    # Save Stage 1 model
    import joblib
    joblib.dump(model_stage1, f"{output_dir}/xgboost_stage1.pkl")
    joblib.dump(scaler_stage1, f"{output_dir}/scaler_stage1.pkl")
    joblib.dump(le_stage1, f"{output_dir}/label_encoder_stage1.pkl")
    
    print("\n✅ Stage 1 Training Complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Average F1-Score: {np.mean(stage1_cv_results['f1']):.4f}")
    
    # Stage 2: DDoS vs DoS classification
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING DDoS vs DoS CLASSIFIER")
    print("=" * 80)
    
    # Prepare DOS subset
    df_dos = prepare_stage2_data(df)
    
    # Train Stage 2 model
    model_stage2, scaler_stage2, le_stage2, stage2_cv_results = train_stage2_model(df_dos)
    
    # Save Stage 2 model
    joblib.dump(model_stage2, f"{output_dir}/xgboost_stage2.pkl")
    joblib.dump(scaler_stage2, f"{output_dir}/scaler_stage2.pkl")
    joblib.dump(le_stage2, f"{output_dir}/label_encoder_stage2.pkl")
    
    print("\n✅ Stage 2 Training Complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Average F1-Score: {np.mean(stage2_cv_results['f1']):.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ V2 HIERARCHICAL XGBOOST TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Models saved to: {output_dir}")
    print("\nStage 1 (4-class) Results:")
    print(f"   Accuracy:  {np.mean(stage1_cv_results['accuracy']):.4f}")
    print(f"   Precision: {np.mean(stage1_cv_results['precision']):.4f}")
    print(f"   Recall:    {np.mean(stage1_cv_results['recall']):.4f}")
    print(f"   F1-Score:  {np.mean(stage1_cv_results['f1']):.4f}")
    
    print("\nStage 2 (DDoS vs DoS) Results:")
    print(f"   Accuracy:  {np.mean(stage2_cv_results['accuracy']):.4f}")
    print(f"   Precision: {np.mean(stage2_cv_results['precision']):.4f}")
    print(f"   Recall:    {np.mean(stage2_cv_results['recall']):.4f}")
    print(f"   F1-Score:  {np.mean(stage2_cv_results['f1']):.4f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

