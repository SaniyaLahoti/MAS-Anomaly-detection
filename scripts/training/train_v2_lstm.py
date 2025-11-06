"""
Train Hierarchical LSTM Model on V2 Dataset
Reuses the V1 LSTM training structure with V2 preprocessed data
"""

import sys
import os
import numpy as np
sys.path.append('../../models/lstm/')

# Import the V1 LSTM training functions
from lstm_hierarchical_classification import (
    load_and_engineer_features,
    create_stage1_labels,
    balance_dataset_stage1,
    prepare_stage2_data,
    prepare_lstm_data,
    train_stage1_kfold,
    train_stage2_kfold,
    train_final_models
)

def main():
    """
    Train hierarchical LSTM on V2 dataset
    """
    print("\n" + "=" * 80)
    print("TRAINING HIERARCHICAL LSTM ON V2 DATASET")
    print("=" * 80)
    print("\nUsing preprocessed V2 data: NF-BoT-IoT-v2-preprocessed.csv")
    print("Same architecture as V1: Stage 1 (4-class) + Stage 2 (DDoS vs DoS)")
    print("=" * 80 + "\n")
    
    # Paths for V2
    input_file = "../../datasets/v2_dataset/NF-BoT-IoT-v2-preprocessed.csv"
    output_dir = "../../models/v2_lstm/"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and engineer features
    df = load_and_engineer_features(input_file)
    
    # Stage 1: Prepare data
    print("\n" + "=" * 80)
    print("STAGE 1: PREPARING 4-CLASS LSTM DATA")
    print("=" * 80)
    
    df_stage1 = create_stage1_labels(df)
    df_s1_balanced = balance_dataset_stage1(df_stage1)
    X_s1, y_s1, y_s1_enc, le_s1, scaler_s1 = prepare_lstm_data(df_s1_balanced, 'Attack_Stage1')
    
    print(f"✅ Stage 1 data prepared: {X_s1.shape}")
    
    # Stage 2: Prepare data
    print("\n" + "=" * 80)
    print("STAGE 2: PREPARING DDoS vs DoS LSTM DATA")
    print("=" * 80)
    
    df_s2_balanced = prepare_stage2_data(df)
    X_s2, y_s2, y_s2_enc, le_s2, scaler_s2 = prepare_lstm_data(df_s2_balanced, 'Attack')
    
    print(f"✅ Stage 2 data prepared: {X_s2.shape}")
    
    # Train Stage 1 with K-Fold CV
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING WITH K-FOLD CV")
    print("=" * 80)
    
    s1_cv, s1_prec, s1_rec, s1_f1 = train_stage1_kfold(X_s1, y_s1, y_s1_enc, le_s1)
    
    # Train Stage 2 with K-Fold CV
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING WITH K-FOLD CV")
    print("=" * 80)
    
    s2_cv = train_stage2_kfold(X_s2, y_s2, y_s2_enc, le_s2)
    
    # Train final models on all data
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODELS ON FULL DATA")
    print("=" * 80)
    
    model_s1, model_s2 = train_final_models(X_s1, y_s1, X_s2, y_s2)
    
    # Save models
    print("\n" + "=" * 80)
    print("SAVING MODELS AND ARTIFACTS")
    print("=" * 80)
    
    model_s1.save(f'{output_dir}/lstm_stage1_model.h5')
    model_s2.save(f'{output_dir}/lstm_stage2_model.h5')
    
    # Save encoders and scalers
    np.save(f'{output_dir}/label_encoder_stage1.npy', le_s1.classes_)
    np.save(f'{output_dir}/label_encoder_stage2.npy', le_s2.classes_)
    np.save(f'{output_dir}/scaler_stage1_mean.npy', scaler_s1.mean_)
    np.save(f'{output_dir}/scaler_stage1_scale.npy', scaler_s1.scale_)
    np.save(f'{output_dir}/scaler_stage2_mean.npy', scaler_s2.mean_)
    np.save(f'{output_dir}/scaler_stage2_scale.npy', scaler_s2.scale_)
    
    print("✅ Models and artifacts saved")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ V2 HIERARCHICAL LSTM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Models saved to: {output_dir}")
    print("\nStage 1 (4-class) CV Results:")
    print(f"   Accuracy:  {np.mean(s1_cv['accuracy']):.4f} (±{np.std(s1_cv['accuracy']):.4f})")
    print(f"   F1-Score:  {np.mean(s1_cv['f1']):.4f} (±{np.std(s1_cv['f1']):.4f})")
    
    print("\nPer-Class Metrics (Stage 1):")
    for i, cls in enumerate(le_s1.classes_):
        print(f"   {cls:<20} P:{s1_prec[i]:.4f}  R:{s1_rec[i]:.4f}  F1:{s1_f1[i]:.4f}")
    
    print("\nStage 2 (DDoS vs DoS) CV Results:")
    print(f"   Accuracy:  {np.mean(s2_cv['accuracy']):.4f} (±{np.std(s2_cv['accuracy']):.4f})")
    print(f"   F1-Score:  {np.mean(s2_cv['f1']):.4f} (±{np.std(s2_cv['f1']):.4f})")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

