"""
Verify XGBoost Model Weights Saved in .pkl File
Tests the saved .pkl model against the original .json model using V1 test dataset
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """
    Apply same feature engineering as training
    """
    # Remove IP addresses and Label
    if 'IPV4_SRC_ADDR' in df.columns:
        df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'])
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    
    # Engineer features
    df['PACKET_RATE'] = (df['IN_PKTS'] + df['OUT_PKTS']) / (df['FLOW_DURATION_MILLISECONDS'] + 1)
    df['BYTE_RATE'] = (df['IN_BYTES'] + df['OUT_BYTES']) / (df['FLOW_DURATION_MILLISECONDS'] + 1)
    df['AVG_PACKET_SIZE'] = (df['IN_BYTES'] + df['OUT_BYTES']) / (df['IN_PKTS'] + df['OUT_PKTS'] + 1)
    df['AVG_IN_PACKET_SIZE'] = df['IN_BYTES'] / (df['IN_PKTS'] + 1)
    df['AVG_OUT_PACKET_SIZE'] = df['OUT_BYTES'] / (df['OUT_PKTS'] + 1)
    df['BYTE_ASYMMETRY'] = abs(df['IN_BYTES'] - df['OUT_BYTES']) / (df['IN_BYTES'] + df['OUT_BYTES'] + 1)
    df['PACKET_ASYMMETRY'] = abs(df['IN_PKTS'] - df['OUT_PKTS']) / (df['IN_PKTS'] + df['OUT_PKTS'] + 1)
    df['IN_OUT_BYTE_RATIO'] = df['IN_BYTES'] / (df['OUT_BYTES'] + 1)
    df['IN_OUT_PACKET_RATIO'] = df['IN_PKTS'] / (df['OUT_PKTS'] + 1)
    df['PROTOCOL_INTENSITY'] = df['PROTOCOL'] * df['PACKET_RATE']
    df['TCP_PACKET_INTERACTION'] = df['TCP_FLAGS'] * df['IN_PKTS']
    df['PROTOCOL_PORT_COMBO'] = df['PROTOCOL'] * df['L4_DST_PORT']
    df['FLOW_INTENSITY'] = (df['IN_PKTS'] + df['OUT_PKTS']) / (df['FLOW_DURATION_MILLISECONDS'] + 1) * df['AVG_PACKET_SIZE']
    
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def verify_model_weights():
    """
    Verify that the saved .pkl model weights match the original .json model
    """
    print("=" * 80)
    print("VERIFYING XGBOOST MODEL WEIGHTS FROM .PKL FILE")
    print("=" * 80)
    
    base_dir = '../models/hierarchical/'
    
    # Load V1 test dataset
    print("\n📊 Loading V1 test dataset...")
    df = pd.read_csv('../datasets/v1_dataset/NF-BoT-IoT.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare features and labels
    X = df.drop(columns=['Attack'])
    y = df['Attack']
    
    # Split into train/test (80/20) to simulate training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Test dataset prepared: {X_test.shape[0]} samples")
    print(f"   Attack distribution in test set:")
    print(y_test.value_counts())
    
    # Load scalers and encoders
    print("\n📦 Loading scalers and encoders...")
    scaler_stage1 = np.load(os.path.join(base_dir, 'hierarchical_stage1_scaler.npy'), allow_pickle=True).item()
    scaler_stage2 = np.load(os.path.join(base_dir, 'hierarchical_stage2_scaler.npy'), allow_pickle=True).item()
    encoder_stage1_classes = np.load(os.path.join(base_dir, 'hierarchical_stage1_encoder.npy'), allow_pickle=True)
    encoder_stage2_classes = np.load(os.path.join(base_dir, 'hierarchical_stage2_encoder.npy'), allow_pickle=True)
    print("✅ Scalers and encoders loaded")
    
    # Load original .json model
    print("\n📦 Loading ORIGINAL .json model (Stage 1)...")
    model_json_stage1 = xgb.XGBClassifier()
    model_json_stage1.load_model(os.path.join(base_dir, 'hierarchical_stage1_model.json'))
    print("✅ Original .json model loaded")
    
    # Load saved .pkl model
    print("\n📦 Loading SAVED .pkl model (Stage 1)...")
    model_pkl_stage1 = joblib.load(os.path.join(base_dir, 'hierarchical_stage1_model.pkl'))
    print("✅ Saved .pkl model loaded")
    
    # Prepare test data
    X_test_scaled = scaler_stage1.transform(X_test)
    
    # Get predictions from both models
    print("\n🔍 Comparing predictions from both models...")
    print("   Testing on", len(X_test), "samples")
    
    # Predictions from original model
    pred_json = model_json_stage1.predict(X_test_scaled)
    proba_json = model_json_stage1.predict_proba(X_test_scaled)
    
    # Predictions from saved .pkl model
    pred_pkl = model_pkl_stage1.predict(X_test_scaled)
    proba_pkl = model_pkl_stage1.predict_proba(X_test_scaled)
    
    # Compare predictions
    predictions_match = np.array_equal(pred_json, pred_pkl)
    probabilities_match = np.allclose(proba_json, proba_pkl, rtol=1e-10)
    
    print(f"\n✅ Prediction Comparison:")
    print(f"   Predictions match: {predictions_match}")
    print(f"   Probabilities match: {probabilities_match}")
    
    if predictions_match:
        print(f"   ✅ All {len(X_test)} predictions are IDENTICAL!")
    else:
        mismatches = np.sum(pred_json != pred_pkl)
        print(f"   ⚠️  {mismatches} predictions differ out of {len(X_test)}")
    
    if probabilities_match:
        max_diff = np.max(np.abs(proba_json - proba_pkl))
        print(f"   ✅ Probabilities are IDENTICAL (max diff: {max_diff:.2e})")
    else:
        max_diff = np.max(np.abs(proba_json - proba_pkl))
        print(f"   ⚠️  Probabilities differ (max diff: {max_diff:.2e})")
    
    # Compare feature importances
    print(f"\n🔍 Comparing feature importances...")
    feat_imp_json = model_json_stage1.feature_importances_
    feat_imp_pkl = model_pkl_stage1.feature_importances_
    feat_imp_match = np.allclose(feat_imp_json, feat_imp_pkl, rtol=1e-10)
    
    if feat_imp_match:
        print(f"   ✅ Feature importances are IDENTICAL!")
    else:
        max_diff = np.max(np.abs(feat_imp_json - feat_imp_pkl))
        print(f"   ⚠️  Feature importances differ (max diff: {max_diff:.2e})")
    
    # Test accuracy on test set
    print(f"\n📊 Testing model accuracy on test dataset...")
    
    # Create stage 1 labels (4-class)
    stage1_labels = []
    for attack in y_test:
        if attack in ['DDoS', 'DoS']:
            stage1_labels.append('DOS')
        else:
            stage1_labels.append(attack)
    
    y_test_stage1 = np.array(stage1_labels)
    
    # Get accuracy - need to map predictions to labels
    pred_labels = [encoder_stage1_classes[pred] for pred in pred_pkl]
    accuracy = accuracy_score(y_test_stage1, pred_labels)
    print(f"   ✅ Test Accuracy: {accuracy*100:.2f}%")
    
    # Classification report
    print(f"\n📋 Classification Report (Using .pkl model):")
    print(classification_report(y_test_stage1, pred_labels, 
                                target_names=list(encoder_stage1_classes)))
    
    # Test on a few specific samples
    print(f"\n🔬 Detailed Sample Testing:")
    print("-" * 80)
    
    sample_indices = [0, 100, 500, 1000] if len(X_test) > 1000 else [0, 10, 50, 100]
    
    for idx in sample_indices:
        if idx >= len(X_test):
            continue
            
        sample = X_test_scaled[idx:idx+1]
        actual = y_test_stage1[idx]
        
        # Predictions from both models
        pred_json_idx = model_json_stage1.predict(sample)[0]
        proba_json_sample = model_json_stage1.predict_proba(sample)[0]
        pred_json_label = encoder_stage1_classes[pred_json_idx]
        
        pred_pkl_idx = model_pkl_stage1.predict(sample)[0]
        proba_pkl_sample = model_pkl_stage1.predict_proba(sample)[0]
        pred_pkl_label = encoder_stage1_classes[pred_pkl_idx]
        
        print(f"\n   Sample {idx}:")
        print(f"   Actual: {actual}")
        print(f"   .json prediction: {pred_json_label} (conf: {np.max(proba_json_sample):.4f})")
        print(f"   .pkl prediction:  {pred_pkl_label} (conf: {np.max(proba_pkl_sample):.4f})")
        print(f"   Match: {'✅' if pred_json_idx == pred_pkl_idx else '❌'}")
        if pred_json_idx == pred_pkl_idx:
            print(f"   Probabilities: {np.max(np.abs(proba_json_sample - proba_pkl_sample)):.2e} diff")
    
    # Final verification
    print("\n" + "=" * 80)
    if predictions_match and probabilities_match and feat_imp_match:
        print("✅ VERIFICATION SUCCESSFUL!")
        print("   The saved .pkl model weights are IDENTICAL to the original .json model")
        print("   All model weights have been saved correctly!")
    else:
        print("⚠️  VERIFICATION ISSUES DETECTED")
        print("   Some differences found between .json and .pkl models")
    print("=" * 80)
    
    return predictions_match and probabilities_match and feat_imp_match

if __name__ == "__main__":
    verify_model_weights()

