"""
LSTM Multi-Class Classification for Anomaly Detection

This module implements LSTM-based multi-class classification for network traffic
using the same features and preprocessing as the XGBoost model.

Classes: Benign, DDoS, DoS, Reconnaissance, Theft
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_engineer_features(file_path):
    """
    Load data and create same engineered features as XGBoost model
    """
    print("=" * 80)
    print("LSTM MULTI-CLASS CLASSIFICATION")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    print("\n📊 ORIGINAL ATTACK DISTRIBUTION")
    print("-" * 80)
    print(df['Attack'].value_counts())
    
    # Remove IP addresses and Label column
    df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'])
    
    # Engineer same features as XGBoost
    print("\n🔧 ENGINEERING FEATURES (same as XGBoost)...")
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

def balance_multiclass_data(df, target_size=20000):
    """
    Balance all attack classes
    """
    print("\n🔄 BALANCING DATASET")
    print("-" * 80)
    
    balanced_dfs = []
    for attack_type in df['Attack'].unique():
        df_class = df[df['Attack'] == attack_type]
        
        if len(df_class) > target_size:
            df_resampled = resample(df_class, replace=False, n_samples=target_size, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=target_size, random_state=42)
        
        balanced_dfs.append(df_resampled)
    
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Balanced dataset: {df_balanced.shape}")
    print("\nClass distribution:")
    print(df_balanced['Attack'].value_counts())
    
    return df_balanced

def prepare_data_for_lstm(df):
    """
    Prepare data for LSTM: scale features and encode labels
    """
    print("\n🔧 PREPARING DATA FOR LSTM")
    print("-" * 80)
    
    # Separate features and target
    X = df.drop(columns=['Attack']).values
    y = df['Attack'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM: (samples, timesteps, features)
    # For flow-level data, we treat each flow as a single timestep
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    
    print(f"✅ X shape for LSTM: {X_lstm.shape}")
    print(f"✅ y shape: {y_categorical.shape}")
    print(f"✅ Classes: {list(label_encoder.classes_)}")
    
    return X_lstm, y_categorical, y_encoded, label_encoder, scaler

def create_lstm_model(input_shape, num_classes):
    """
    Create LSTM model architecture
    """
    print("\n🔧 BUILDING LSTM MODEL")
    print("-" * 80)
    
    model = Sequential([
        # First LSTM layer
        LSTM(128, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.3),
        
        # Second LSTM layer
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        # Third LSTM layer
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        # Dense layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ LSTM Model Architecture:")
    model.summary()
    
    return model

def perform_kfold_cross_validation(X, y, y_encoded, label_encoder, n_splits=5):
    """
    Perform k-fold cross-validation for LSTM
    """
    print("\n" + "=" * 80)
    print(f"PERFORMING {n_splits}-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'per_class_precision': [],
        'per_class_recall': [],
        'per_class_f1': []
    }
    
    fold_num = 1
    
    for train_idx, test_idx in skf.split(X, y_encoded):
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_num}/{n_splits}")
        print(f"{'=' * 80}")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_encoded = y_encoded[train_idx]
        y_test_encoded = y_encoded[test_idx]
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Create model
        model = create_lstm_model(
            input_shape=(X.shape[1], X.shape[2]),
            num_classes=y.shape[1]
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
        
        # Train model
        print(f"\n🔄 Training Fold {fold_num}...")
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=50,
            batch_size=64,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"   Epochs trained: {len(history.history['loss'])}")
        
        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_classes)
        precision = precision_score(y_test_encoded, y_pred_classes, average='weighted', zero_division=0)
        recall = recall_score(y_test_encoded, y_pred_classes, average='weighted', zero_division=0)
        f1 = f1_score(y_test_encoded, y_pred_classes, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test_encoded, y_pred_classes, average=None, zero_division=0)
        recall_per_class = recall_score(y_test_encoded, y_pred_classes, average=None, zero_division=0)
        f1_per_class = f1_score(y_test_encoded, y_pred_classes, average=None, zero_division=0)
        
        # Store results
        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        fold_results['per_class_precision'].append(precision_per_class)
        fold_results['per_class_recall'].append(recall_per_class)
        fold_results['per_class_f1'].append(f1_per_class)
        
        print(f"\n📊 Fold {fold_num} Results:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        
        fold_num += 1
    
    # Calculate average metrics
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 OVERALL METRICS (Mean ± Std):")
    print(f"   Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})")
    print(f"   Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})")
    print(f"   Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})")
    print(f"   F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})")
    
    # Per-class averages
    print(f"\n📊 PER-CLASS PERFORMANCE:")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    avg_precision_per_class = np.mean(fold_results['per_class_precision'], axis=0)
    avg_recall_per_class = np.mean(fold_results['per_class_recall'], axis=0)
    avg_f1_per_class = np.mean(fold_results['per_class_f1'], axis=0)
    
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{class_name:<20} {avg_precision_per_class[i]:<12.4f} {avg_recall_per_class[i]:<12.4f} {avg_f1_per_class[i]:<12.4f}")
    
    return fold_results, avg_precision_per_class, avg_recall_per_class, avg_f1_per_class

def train_final_model(X, y, y_encoded):
    """
    Train final model on all data
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL ON ALL DATA")
    print("=" * 80)
    
    # Create model
    model = create_lstm_model(
        input_shape=(X.shape[1], X.shape[2]),
        num_classes=y.shape[1]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=0
    )
    
    # Train
    print(f"\n🔄 Training final model...")
    start_time = time.time()
    
    history = model.fit(
        X, y,
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    print(f"\n✅ Final model trained in {training_time:.2f} seconds")
    print(f"   Final loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model, training_time

def save_results(fold_results, avg_precision, avg_recall, avg_f1, label_encoder, training_time):
    """
    Save results to files
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Text file
    with open('lstm_multiclass_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("LSTM MULTI-CLASS CLASSIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ATTACK TYPES:\n")
        f.write("-" * 80 + "\n")
        for i, class_name in enumerate(label_encoder.classes_):
            f.write(f"  {i}: {class_name}\n")
        
        f.write(f"\n5-FOLD CROSS-VALIDATION RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})\n")
        f.write(f"Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})\n")
        f.write(f"Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})\n")
        f.write(f"F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})\n")
        
        f.write(f"\nPER-CLASS PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        for i, class_name in enumerate(label_encoder.classes_):
            f.write(f"{class_name:<20} {avg_precision[i]:<12.4f} {avg_recall[i]:<12.4f} {avg_f1[i]:<12.4f}\n")
        
        f.write(f"\nFINAL MODEL TRAINING TIME: {training_time:.2f} seconds\n")
    
    # JSON file
    results_json = {
        'overall_metrics': {
            'accuracy_mean': float(np.mean(fold_results['accuracy'])),
            'accuracy_std': float(np.std(fold_results['accuracy'])),
            'precision_mean': float(np.mean(fold_results['precision'])),
            'precision_std': float(np.std(fold_results['precision'])),
            'recall_mean': float(np.mean(fold_results['recall'])),
            'recall_std': float(np.std(fold_results['recall'])),
            'f1_mean': float(np.mean(fold_results['f1'])),
            'f1_std': float(np.std(fold_results['f1']))
        },
        'per_class_metrics': {
            label_encoder.classes_[i]: {
                'precision': float(avg_precision[i]),
                'recall': float(avg_recall[i]),
                'f1_score': float(avg_f1[i])
            }
            for i in range(len(label_encoder.classes_))
        },
        'training_time': float(training_time),
        'classes': list(label_encoder.classes_)
    }
    
    with open('lstm_multiclass_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print("✅ Results saved to 'lstm_multiclass_results.txt'")
    print("✅ Results saved to 'lstm_multiclass_results.json'")

def main():
    """
    Main execution
    """
    print("\n" + "=" * 80)
    print("LSTM MULTI-CLASS ANOMALY DETECTION")
    print("Using same features and preprocessing as XGBoost")
    print("=" * 80 + "\n")
    
    file_path = "/Users/saniyalahoti/Downloads/MAS-LSTM-1/v1_dataset/NF-BoT-IoT.csv"
    
    # Load and engineer features
    df = load_and_engineer_features(file_path)
    
    # Balance dataset
    df_balanced = balance_multiclass_data(df, target_size=20000)
    
    # Prepare data for LSTM
    X, y, y_encoded, label_encoder, scaler = prepare_data_for_lstm(df_balanced)
    
    # Perform k-fold cross-validation
    fold_results, avg_precision, avg_recall, avg_f1 = perform_kfold_cross_validation(
        X, y, y_encoded, label_encoder, n_splits=5
    )
    
    # Train final model
    final_model, training_time = train_final_model(X, y, y_encoded)
    
    # Save model
    print("\n💾 Saving final model...")
    final_model.save('lstm_multiclass_model.h5')
    np.save('lstm_label_encoder.npy', label_encoder.classes_)
    np.save('lstm_scaler_mean.npy', scaler.mean_)
    np.save('lstm_scaler_scale.npy', scaler.scale_)
    print("✅ Model saved to 'lstm_multiclass_model.h5'")
    
    # Save results
    save_results(fold_results, avg_precision, avg_recall, avg_f1, label_encoder, training_time)
    
    print("\n" + "=" * 80)
    print("LSTM MULTI-CLASS CLASSIFICATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Overall Accuracy: {np.mean(fold_results['accuracy'])*100:.2f}%")
    print(f"✅ Overall F1-Score: {np.mean(fold_results['f1']):.4f}")
    print(f"✅ Model saved: lstm_multiclass_model.h5")
    print(f"✅ Results saved: lstm_multiclass_results.txt")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

