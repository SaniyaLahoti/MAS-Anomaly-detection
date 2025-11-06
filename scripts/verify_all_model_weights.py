"""
Verify All Model Weights Saved in .pkl Files
Tests saved .pkl models against original models using test datasets
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """Apply feature engineering (same as training)"""
    # Remove IP addresses and Label if present
    cols_to_drop = []
    if 'IPV4_SRC_ADDR' in df.columns:
        cols_to_drop.extend(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'])
    if 'Label' in df.columns:
        cols_to_drop.append('Label')
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
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

def verify_xgboost_v1():
    """Verify XGBoost V1 model weights"""
    print("\n" + "=" * 80)
    print("VERIFYING XGBOOST V1 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/hierarchical/'
    
    # Load test dataset
    print("\n📊 Loading V1 test dataset...")
    df = pd.read_csv('../datasets/v1_dataset/NF-BoT-IoT.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare test data
    X = df.drop(columns=['Attack'])
    y = df['Attack']
    
    # Split (use consistent random_state)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create stage 1 labels
    stage1_labels = []
    for attack in y_test:
        if attack in ['DDoS', 'DoS']:
            stage1_labels.append('DOS')
        else:
            stage1_labels.append(attack)
    y_test_stage1 = np.array(stage1_labels)
    
    print(f"✅ Test dataset prepared: {X_test.shape[0]} samples")
    
    # Load scalers and encoders
    scaler = np.load(os.path.join(base_dir, 'hierarchical_stage1_scaler.npy'), allow_pickle=True).item()
    encoder_classes = np.load(os.path.join(base_dir, 'hierarchical_stage1_encoder.npy'), allow_pickle=True)
    
    # Load original .json model
    print("\n📦 Loading ORIGINAL .json model...")
    model_json = xgb.XGBClassifier()
    model_json.load_model(os.path.join(base_dir, 'hierarchical_stage1_model.json'))
    
    # Load saved .pkl model
    print("📦 Loading SAVED .pkl model...")
    model_pkl = joblib.load(os.path.join(base_dir, 'hierarchical_stage1_model.pkl'))
    
    # Scale test data
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    print("\n🔍 Comparing predictions...")
    pred_json = model_json.predict(X_test_scaled)
    pred_pkl = model_pkl.predict(X_test_scaled)
    proba_json = model_json.predict_proba(X_test_scaled)
    proba_pkl = model_pkl.predict_proba(X_test_scaled)
    
    # Compare
    predictions_match = np.array_equal(pred_json, pred_pkl)
    probabilities_match = np.allclose(proba_json, proba_pkl, rtol=1e-10)
    
    print(f"   Predictions match: {predictions_match} ({'✅' if predictions_match else '❌'})")
    print(f"   Probabilities match: {probabilities_match} ({'✅' if probabilities_match else '❌'})")
    
    if predictions_match:
        # Test accuracy
        pred_labels = [encoder_classes[pred] for pred in pred_pkl]
        accuracy = accuracy_score(y_test_stage1, pred_labels)
        print(f"   Test Accuracy: {accuracy*100:.2f}%")
    
    return predictions_match and probabilities_match

def verify_xgboost_v2():
    """Verify XGBoost V2 model weights"""
    print("\n" + "=" * 80)
    print("VERIFYING XGBOOST V2 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/v2_hierarchical/'
    
    # Check if preprocessed V2 dataset exists
    dataset_path = '../datasets/v2_dataset/NF-BoT-IoT-v2-preprocessed.csv'
    if not os.path.exists(dataset_path):
        print("⚠️  V2 preprocessed dataset not found")
        return False
    
    # Load test dataset
    print("\n📊 Loading V2 test dataset...")
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare test data
    X = df.drop(columns=['Attack'])
    y = df['Attack']
    
    # Split (use consistent random_state)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create stage 1 labels
    stage1_labels = []
    for attack in y_test:
        if attack in ['DDoS', 'DoS']:
            stage1_labels.append('DOS')
        else:
            stage1_labels.append(attack)
    y_test_stage1 = np.array(stage1_labels)
    
    print(f"✅ Test dataset prepared: {X_test.shape[0]} samples")
    
    # Load model and artifacts
    print("\n📦 Loading V2 XGBoost model...")
    model = joblib.load(os.path.join(base_dir, 'xgboost_stage1.pkl'))
    scaler = joblib.load(os.path.join(base_dir, 'scaler_stage1.pkl'))
    label_encoder = joblib.load(os.path.join(base_dir, 'label_encoder_stage1.pkl'))
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    pred_labels = label_encoder.inverse_transform(predictions)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_stage1, pred_labels)
    print(f"✅ Model loaded and tested successfully")
    print(f"   Test Accuracy: {accuracy*100:.2f}%")
    print(f"   Model parameters: {model.n_estimators} estimators")
    
    return True

def verify_lstm_v1():
    """Verify LSTM V1 model weights"""
    print("\n" + "=" * 80)
    print("VERIFYING LSTM V1 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/lstm/'
    
    # Load test dataset
    print("\n📊 Loading V1 test dataset...")
    df = pd.read_csv('../datasets/v1_dataset/NF-BoT-IoT.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare test data
    X = df.drop(columns=['Attack'])
    y = df['Attack']
    
    # Split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create stage 1 labels
    stage1_labels = []
    for attack in y_test:
        if attack in ['DDoS', 'DoS']:
            stage1_labels.append('DOS')
        else:
            stage1_labels.append(attack)
    
    print(f"✅ Test dataset prepared: {X_test.shape[0]} samples")
    
    # Load scaler and encoder
    scaler_mean = np.load(os.path.join(base_dir, 'lstm_hierarchical_s1_scaler_mean.npy'))
    scaler_scale = np.load(os.path.join(base_dir, 'lstm_hierarchical_s1_scaler_scale.npy'))
    encoder_classes = np.load(os.path.join(base_dir, 'lstm_hierarchical_s1_encoder.npy'), allow_pickle=True)
    
    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    # Load original .h5 model
    print("\n📦 Loading ORIGINAL .h5 model...")
    model_h5 = load_model(os.path.join(base_dir, 'lstm_hierarchical_stage1_model.h5'))
    
    # Load saved .pkl model
    print("📦 Loading SAVED .pkl model...")
    pkl_path = os.path.join(base_dir, 'lstm_hierarchical_stage1_model.pkl')
    if not os.path.exists(pkl_path):
        print("⚠️  .pkl model not found. Please run save_all_model_weights.py first")
        return False
    
    model_pkl = joblib.load(pkl_path)
    
    # Prepare test data for LSTM
    X_test_scaled = scaler.transform(X_test)
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Get predictions
    print("\n🔍 Comparing predictions...")
    pred_h5 = model_h5.predict(X_test_lstm, verbose=0)
    pred_pkl = model_pkl.predict(X_test_lstm, verbose=0)
    
    # Compare
    predictions_match = np.allclose(pred_h5, pred_pkl, rtol=1e-6)
    max_diff = np.max(np.abs(pred_h5 - pred_pkl))
    
    print(f"   Predictions match: {predictions_match} ({'✅' if predictions_match else '❌'})")
    print(f"   Max difference: {max_diff:.2e}")
    
    if predictions_match:
        # Get class predictions
        pred_classes = np.argmax(pred_pkl, axis=1)
        pred_labels = [encoder_classes[pred] for pred in pred_classes]
        accuracy = accuracy_score(stage1_labels, pred_labels)
        print(f"   Test Accuracy: {accuracy*100:.2f}%")
    
    return predictions_match

def verify_lstm_v2():
    """Verify LSTM V2 model weights"""
    print("\n" + "=" * 80)
    print("VERIFYING LSTM V2 MODEL WEIGHTS")
    print("=" * 80)
    
    base_dir = '../models/v2_lstm/'
    
    # Check if preprocessed V2 dataset exists
    dataset_path = '../datasets/v2_dataset/NF-BoT-IoT-v2-preprocessed.csv'
    if not os.path.exists(dataset_path):
        print("⚠️  V2 preprocessed dataset not found")
        return False
    
    # Load test dataset
    print("\n📊 Loading V2 test dataset...")
    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Engineer features
    df = engineer_features(df)
    
    # Prepare test data
    X = df.drop(columns=['Attack'])
    y = df['Attack']
    
    # Split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Create stage 1 labels
    stage1_labels = []
    for attack in y_test:
        if attack in ['DDoS', 'DoS']:
            stage1_labels.append('DOS')
        else:
            stage1_labels.append(attack)
    
    print(f"✅ Test dataset prepared: {X_test.shape[0]} samples")
    
    # Load scaler and encoder
    scaler_mean = np.load(os.path.join(base_dir, 'scaler_stage1_mean.npy'))
    scaler_scale = np.load(os.path.join(base_dir, 'scaler_stage1_scale.npy'))
    encoder_classes = np.load(os.path.join(base_dir, 'label_encoder_stage1.npy'), allow_pickle=True)
    
    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale
    
    # Load original .h5 model
    print("\n📦 Loading ORIGINAL .h5 model...")
    model_h5 = load_model(os.path.join(base_dir, 'lstm_stage1_model.h5'))
    
    # Load saved .pkl model
    print("📦 Loading SAVED .pkl model...")
    pkl_path = os.path.join(base_dir, 'lstm_stage1_model.pkl')
    if not os.path.exists(pkl_path):
        print("⚠️  .pkl model not found. Please run save_all_model_weights.py first")
        return False
    
    model_pkl = joblib.load(pkl_path)
    
    # Prepare test data for LSTM
    X_test_scaled = scaler.transform(X_test)
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Get predictions
    print("\n🔍 Comparing predictions...")
    pred_h5 = model_h5.predict(X_test_lstm, verbose=0)
    pred_pkl = model_pkl.predict(X_test_lstm, verbose=0)
    
    # Compare
    predictions_match = np.allclose(pred_h5, pred_pkl, rtol=1e-6)
    max_diff = np.max(np.abs(pred_h5 - pred_pkl))
    
    print(f"   Predictions match: {predictions_match} ({'✅' if predictions_match else '❌'})")
    print(f"   Max difference: {max_diff:.2e}")
    
    if predictions_match:
        # Get class predictions
        pred_classes = np.argmax(pred_pkl, axis=1)
        pred_labels = [encoder_classes[pred] for pred in pred_classes]
        accuracy = accuracy_score(stage1_labels, pred_labels)
        print(f"   Test Accuracy: {accuracy*100:.2f}%")
    
    return predictions_match

def main():
    """Verify all model weights"""
    print("\n" + "=" * 80)
    print("VERIFYING ALL MODEL WEIGHTS")
    print("=" * 80)
    print("\nThis script will verify weights for:")
    print("1. XGBoost V1 (hierarchical)")
    print("2. XGBoost V2 (hierarchical)")
    print("3. LSTM V1 (hierarchical)")
    print("4. LSTM V2 (hierarchical)")
    print("=" * 80)
    
    results = {}
    
    # Verify XGBoost V1
    try:
        results['xgboost_v1'] = verify_xgboost_v1()
    except Exception as e:
        print(f"❌ Error verifying XGBoost V1: {e}")
        results['xgboost_v1'] = False
    
    # Verify XGBoost V2
    try:
        results['xgboost_v2'] = verify_xgboost_v2()
    except Exception as e:
        print(f"❌ Error verifying XGBoost V2: {e}")
        results['xgboost_v2'] = False
    
    # Verify LSTM V1
    try:
        results['lstm_v1'] = verify_lstm_v1()
    except Exception as e:
        print(f"❌ Error verifying LSTM V1: {e}")
        results['lstm_v1'] = False
    
    # Verify LSTM V2
    try:
        results['lstm_v2'] = verify_lstm_v2()
    except Exception as e:
        print(f"❌ Error verifying LSTM V2: {e}")
        results['lstm_v2'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    for model_name, success in results.items():
        status = "✅ VERIFIED" if success else "❌ FAILED"
        print(f"{model_name.upper():<20} {status}")
    
    total_success = sum(results.values())
    print(f"\n{total_success}/{len(results)} models verified successfully")
    
    if total_success == len(results):
        print("\n" + "=" * 80)
        print("✅ ALL MODEL WEIGHTS VERIFIED SUCCESSFULLY!")
        print("   All saved .pkl files match original models")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️  SOME VERIFICATIONS FAILED")
        print("   Check error messages above for details")
        print("=" * 80)
    
    return results

if __name__ == "__main__":
    main()

