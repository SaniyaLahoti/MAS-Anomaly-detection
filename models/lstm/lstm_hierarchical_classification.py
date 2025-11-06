"""
Hierarchical LSTM Multi-Class Classification for Anomaly Detection

This module implements a two-stage hierarchical LSTM model:
- Stage 1: 4-class classification (Benign, DOS, Reconnaissance, Theft)
- Stage 2: Binary classification (DDoS vs DoS) for DOS samples

Mirrors the XGBoost hierarchical approach for fair comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def load_and_engineer_features(file_path):
    """
    Load data and engineer features (same as XGBoost)
    """
    print("=" * 80)
    print("HIERARCHICAL LSTM CLASSIFICATION")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    print("\n📊 ORIGINAL ATTACK DISTRIBUTION")
    print("-" * 80)
    print(df['Attack'].value_counts())
    
    # Remove IP addresses and Label
    df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'])
    
    # Engineer features
    print("\n🔧 ENGINEERING FEATURES...")
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
    
    print(f"✅ Total features: {len(df.columns) - 1} (23 features)")
    
    return df

def create_stage1_labels(df):
    """
    Stage 1: Merge DDoS and DoS into 'DOS' class
    """
    print("\n" + "=" * 80)
    print("STAGE 1: 4-CLASS CLASSIFICATION (DDoS+DoS merged)")
    print("=" * 80)
    
    df_stage1 = df.copy()
    df_stage1['Attack_Stage1'] = df_stage1['Attack'].replace({
        'DDoS': 'DOS',
        'DoS': 'DOS'
    })
    
    print("\nStage 1 Attack Distribution:")
    print(df_stage1['Attack_Stage1'].value_counts())
    
    return df_stage1

def balance_dataset_stage1(df, target_size=25000):
    """
    Balance Stage 1 dataset
    """
    print("\n🔄 BALANCING STAGE 1 DATASET")
    print("-" * 80)
    
    balanced_dfs = []
    for attack_type in df['Attack_Stage1'].unique():
        df_class = df[df['Attack_Stage1'] == attack_type]
        
        if len(df_class) > target_size:
            df_resampled = resample(df_class, replace=False, n_samples=target_size, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=target_size, random_state=42)
        
        balanced_dfs.append(df_resampled)
    
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Balanced Stage 1: {df_balanced.shape}")
    print("Class distribution:")
    print(df_balanced['Attack_Stage1'].value_counts())
    
    return df_balanced

def prepare_stage2_data(df_original, target_size=30000):
    """
    Stage 2: Only DDoS vs DoS samples
    """
    print("\n" + "=" * 80)
    print("STAGE 2: BINARY CLASSIFICATION (DDoS vs DoS)")
    print("=" * 80)
    
    df_stage2 = df_original[df_original['Attack'].isin(['DDoS', 'DoS'])].copy()
    
    print(f"Stage 2 samples: {df_stage2.shape}")
    print("\nAttack distribution:")
    print(df_stage2['Attack'].value_counts())
    
    # Balance
    balanced_dfs = []
    for attack_type in df_stage2['Attack'].unique():
        df_class = df_stage2[df_stage2['Attack'] == attack_type]
        
        if len(df_class) > target_size:
            df_resampled = resample(df_class, replace=False, n_samples=target_size, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=target_size, random_state=42)
        
        balanced_dfs.append(df_resampled)
    
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Balanced Stage 2: {df_balanced.shape}")
    print("Class distribution:")
    print(df_balanced['Attack'].value_counts())
    
    return df_balanced

def prepare_lstm_data(df, target_column):
    """
    Prepare data for LSTM
    """
    X = df.drop(columns=['Attack', target_column] if 'Attack' in df.columns else [target_column])
    y = df[target_column]
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM: (samples, timesteps, features)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    return X_lstm, y_categorical, y_encoded, le, scaler

def create_lstm_stage1_model(input_shape, num_classes):
    """
    Create LSTM Stage 1 model
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_lstm_stage2_model(input_shape):
    """
    Create LSTM Stage 2 model (Binary)
    """
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        Dropout(0.2),
        
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_stage1_kfold(X, y, y_encoded, label_encoder, n_splits=5):
    """
    Train Stage 1 with k-fold CV
    """
    print("\n" + "=" * 80)
    print(f"TRAINING STAGE 1 - {n_splits}-FOLD CV")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
                   'per_class_precision': [], 'per_class_recall': [], 'per_class_f1': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'=' * 80}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_enc = y_encoded[train_idx]
        y_test_enc = y_encoded[test_idx]
        
        # Create model
        model = create_lstm_stage1_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=y.shape[1]
        )
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
        
        # Train
        print(f"🔄 Training Fold {fold}...")
        start = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        train_time = time.time() - start
        print(f"✅ Completed in {train_time:.2f}s, Epochs: {len(history.history['loss'])}")
        
        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(y_test_enc, y_pred_classes)
        prec = precision_score(y_test_enc, y_pred_classes, average='weighted', zero_division=0)
        rec = recall_score(y_test_enc, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test_enc, y_pred_classes, average='weighted', zero_division=0)
        
        prec_per = precision_score(y_test_enc, y_pred_classes, average=None, zero_division=0)
        rec_per = recall_score(y_test_enc, y_pred_classes, average=None, zero_division=0)
        f1_per = f1_score(y_test_enc, y_pred_classes, average=None, zero_division=0)
        
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1'].append(f1)
        fold_results['per_class_precision'].append(prec_per)
        fold_results['per_class_recall'].append(rec_per)
        fold_results['per_class_f1'].append(f1_per)
        
        print(f"Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("STAGE 1 CV SUMMARY")
    print(f"{'=' * 80}")
    print(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})")
    print(f"Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})")
    print(f"Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})")
    print(f"F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})")
    
    # Per-class
    print(f"\nPER-CLASS PERFORMANCE:")
    print("-" * 80)
    avg_prec = np.mean(fold_results['per_class_precision'], axis=0)
    avg_rec = np.mean(fold_results['per_class_recall'], axis=0)
    avg_f1 = np.mean(fold_results['per_class_f1'], axis=0)
    
    for i, cls in enumerate(label_encoder.classes_):
        print(f"{cls:<20} P:{avg_prec[i]:.4f}  R:{avg_rec[i]:.4f}  F1:{avg_f1[i]:.4f}")
    
    return fold_results, avg_prec, avg_rec, avg_f1

def train_stage2_kfold(X, y, y_encoded, label_encoder, n_splits=5):
    """
    Train Stage 2 with k-fold CV
    """
    print("\n" + "=" * 80)
    print(f"TRAINING STAGE 2 - {n_splits}-FOLD CV (DDoS vs DoS)")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), 1):
        print(f"\nFold {fold}/{n_splits}...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_enc = y_encoded[train_idx]
        y_test_enc = y_encoded[test_idx]
        
        model = create_lstm_stage2_model(input_shape=(X.shape[1], X.shape[2]))
        
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=0)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        acc = accuracy_score(y_test_enc, y_pred_classes)
        prec = precision_score(y_test_enc, y_pred_classes, average='binary', zero_division=0)
        rec = recall_score(y_test_enc, y_pred_classes, average='binary', zero_division=0)
        f1 = f1_score(y_test_enc, y_pred_classes, average='binary', zero_division=0)
        
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1'].append(f1)
        
        print(f"  Acc={acc:.4f}, F1={f1:.4f}")
    
    print(f"\n{'=' * 80}")
    print("STAGE 2 CV SUMMARY")
    print(f"{'=' * 80}")
    print(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})")
    print(f"Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})")
    print(f"Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})")
    print(f"F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})")
    
    return fold_results

def train_final_models(X_s1, y_s1, X_s2, y_s2):
    """
    Train final models on all data
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODELS")
    print("=" * 80)
    
    # Stage 1
    print("\n🔄 Training final Stage 1 model...")
    model_s1 = create_lstm_stage1_model(
        input_shape=(X_s1.shape[1], X_s1.shape[2]),
        num_classes=y_s1.shape[1]
    )
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=0)
    
    history_s1 = model_s1.fit(
        X_s1, y_s1,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    print(f"✅ Stage 1 trained: Loss={history_s1.history['loss'][-1]:.4f}, Acc={history_s1.history['accuracy'][-1]:.4f}")
    
    # Stage 2
    print("\n🔄 Training final Stage 2 model...")
    model_s2 = create_lstm_stage2_model(input_shape=(X_s2.shape[1], X_s2.shape[2]))
    
    history_s2 = model_s2.fit(
        X_s2, y_s2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    print(f"✅ Stage 2 trained: Loss={history_s2.history['loss'][-1]:.4f}, Acc={history_s2.history['accuracy'][-1]:.4f}")
    
    return model_s1, model_s2

def save_results(stage1_cv, stage2_cv, s1_prec, s1_rec, s1_f1, s1_le):
    """
    Save results
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    with open('lstm_hierarchical_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIERARCHICAL LSTM CLASSIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STAGE 1 (4-Class):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {np.mean(stage1_cv['accuracy']):.4f} (±{np.std(stage1_cv['accuracy']):.4f})\n")
        f.write(f"F1-Score:  {np.mean(stage1_cv['f1']):.4f} (±{np.std(stage1_cv['f1']):.4f})\n\n")
        
        f.write("Per-Class:\n")
        for i, cls in enumerate(s1_le.classes_):
            f.write(f"{cls:<20} P:{s1_prec[i]:.4f}  R:{s1_rec[i]:.4f}  F1:{s1_f1[i]:.4f}\n")
        
        f.write(f"\nSTAGE 2 (DDoS vs DoS):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {np.mean(stage2_cv['accuracy']):.4f} (±{np.std(stage2_cv['accuracy']):.4f})\n")
        f.write(f"F1-Score:  {np.mean(stage2_cv['f1']):.4f} (±{np.std(stage2_cv['f1']):.4f})\n")
    
    print("✅ Results saved to 'lstm_hierarchical_results.txt'")

def main():
    """
    Main execution
    """
    print("\n" + "=" * 80)
    print("HIERARCHICAL LSTM ANOMALY DETECTION")
    print("=" * 80 + "\n")
    
    file_path = "/Users/saniyalahoti/Downloads/MAS-LSTM-1/v1_dataset/NF-BoT-IoT.csv"
    
    # Load data
    df = load_and_engineer_features(file_path)
    
    # Stage 1 data
    df_stage1 = create_stage1_labels(df)
    df_s1_balanced = balance_dataset_stage1(df_stage1)
    X_s1, y_s1, y_s1_enc, le_s1, scaler_s1 = prepare_lstm_data(df_s1_balanced, 'Attack_Stage1')
    
    # Stage 2 data
    df_s2_balanced = prepare_stage2_data(df)
    X_s2, y_s2, y_s2_enc, le_s2, scaler_s2 = prepare_lstm_data(df_s2_balanced, 'Attack')
    
    # Train Stage 1
    s1_cv, s1_prec, s1_rec, s1_f1 = train_stage1_kfold(X_s1, y_s1, y_s1_enc, le_s1)
    
    # Train Stage 2
    s2_cv = train_stage2_kfold(X_s2, y_s2, y_s2_enc, le_s2)
    
    # Train final models
    model_s1, model_s2 = train_final_models(X_s1, y_s1, X_s2, y_s2)
    
    # Save
    model_s1.save('lstm_hierarchical_stage1_model.h5')
    model_s2.save('lstm_hierarchical_stage2_model.h5')
    np.save('lstm_hierarchical_s1_encoder.npy', le_s1.classes_)
    np.save('lstm_hierarchical_s2_encoder.npy', le_s2.classes_)
    np.save('lstm_hierarchical_s1_scaler_mean.npy', scaler_s1.mean_)
    np.save('lstm_hierarchical_s1_scaler_scale.npy', scaler_s1.scale_)
    np.save('lstm_hierarchical_s2_scaler_mean.npy', scaler_s2.mean_)
    np.save('lstm_hierarchical_s2_scaler_scale.npy', scaler_s2.scale_)
    
    print("\n✅ Models saved:")
    print("  - lstm_hierarchical_stage1_model.h5")
    print("  - lstm_hierarchical_stage2_model.h5")
    
    save_results(s1_cv, s2_cv, s1_prec, s1_rec, s1_f1, le_s1)
    
    print("\n" + "=" * 80)
    print("HIERARCHICAL LSTM TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ Stage 1 Accuracy: {np.mean(s1_cv['accuracy'])*100:.2f}%")
    print(f"✅ Stage 2 Accuracy: {np.mean(s2_cv['accuracy'])*100:.2f}%")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

