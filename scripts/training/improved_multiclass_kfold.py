import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            cohen_kappa_score, make_scorer)
import xgboost as xgb
import time
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data for improved multi-class classification
    """
    print("=" * 80)
    print("IMPROVED MULTI-CLASS ATTACK DETECTION WITH K-FOLD CV")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    print("\n📊 ORIGINAL ATTACK TYPE DISTRIBUTION")
    print("-" * 80)
    print(df['Attack'].value_counts())
    
    # Remove IP addresses and binary label
    features_to_drop = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label']
    df_processed = df.drop(columns=features_to_drop)
    
    return df_processed

def engineer_ddos_dos_features(df):
    """
    Create advanced features to distinguish DDoS from DoS attacks
    """
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING FOR DDOS/DOS DISTINCTION")
    print("=" * 80)
    
    df_enhanced = df.copy()
    
    # 1. Flow rates (critical for DDoS vs DoS)
    df_enhanced['PACKET_RATE'] = (df_enhanced['IN_PKTS'] + df_enhanced['OUT_PKTS']) / (df_enhanced['FLOW_DURATION_MILLISECONDS'] + 1)
    df_enhanced['BYTE_RATE'] = (df_enhanced['IN_BYTES'] + df_enhanced['OUT_BYTES']) / (df_enhanced['FLOW_DURATION_MILLISECONDS'] + 1)
    
    # 2. Average packet sizes (DDoS often uses smaller packets)
    df_enhanced['AVG_PACKET_SIZE'] = (df_enhanced['IN_BYTES'] + df_enhanced['OUT_BYTES']) / (df_enhanced['IN_PKTS'] + df_enhanced['OUT_PKTS'] + 1)
    df_enhanced['AVG_IN_PACKET_SIZE'] = df_enhanced['IN_BYTES'] / (df_enhanced['IN_PKTS'] + 1)
    df_enhanced['AVG_OUT_PACKET_SIZE'] = df_enhanced['OUT_BYTES'] / (df_enhanced['OUT_PKTS'] + 1)
    
    # 3. Traffic asymmetry (DDoS is often more asymmetric)
    df_enhanced['BYTE_ASYMMETRY'] = abs(df_enhanced['IN_BYTES'] - df_enhanced['OUT_BYTES']) / (df_enhanced['IN_BYTES'] + df_enhanced['OUT_BYTES'] + 1)
    df_enhanced['PACKET_ASYMMETRY'] = abs(df_enhanced['IN_PKTS'] - df_enhanced['OUT_PKTS']) / (df_enhanced['IN_PKTS'] + df_enhanced['OUT_PKTS'] + 1)
    
    # 4. Traffic ratios
    df_enhanced['IN_OUT_BYTE_RATIO'] = df_enhanced['IN_BYTES'] / (df_enhanced['OUT_BYTES'] + 1)
    df_enhanced['IN_OUT_PACKET_RATIO'] = df_enhanced['IN_PKTS'] / (df_enhanced['OUT_PKTS'] + 1)
    
    # 5. Duration normalized features
    df_enhanced['BYTES_PER_MS'] = (df_enhanced['IN_BYTES'] + df_enhanced['OUT_BYTES']) / (df_enhanced['FLOW_DURATION_MILLISECONDS'] + 1)
    df_enhanced['PACKETS_PER_MS'] = (df_enhanced['IN_PKTS'] + df_enhanced['OUT_PKTS']) / (df_enhanced['FLOW_DURATION_MILLISECONDS'] + 1)
    
    # 6. Protocol intensity (specific to attack types)
    df_enhanced['PROTOCOL_INTENSITY'] = df_enhanced['PROTOCOL'] * df_enhanced['PACKET_RATE']
    
    # Replace infinite values with 0
    df_enhanced = df_enhanced.replace([np.inf, -np.inf], 0)
    
    # Fill any NaN with 0
    df_enhanced = df_enhanced.fillna(0)
    
    print(f"✅ Created {len(df_enhanced.columns) - len(df.columns)} new features")
    print(f"✅ Total features now: {len(df_enhanced.columns) - 1}")  # -1 for Attack column
    
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    print(f"\nNew features: {new_features}")
    
    return df_enhanced

def balance_dataset(df, target_size=20000):
    """
    Balance dataset with larger sample size for better learning
    """
    print("\n" + "=" * 80)
    print("BALANCING DATASET")
    print("=" * 80)
    
    print(f"Target samples per class: {target_size}")
    
    balanced_dfs = []
    for attack_type in df['Attack'].unique():
        df_class = df[df['Attack'] == attack_type]
        
        if len(df_class) > target_size:
            df_resampled = resample(df_class, replace=False, n_samples=target_size, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=target_size, random_state=42)
        
        balanced_dfs.append(df_resampled)
    
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✅ Balanced dataset shape: {df_balanced.shape}")
    print("\nClass distribution:")
    print(df_balanced['Attack'].value_counts())
    
    return df_balanced

def prepare_data(df):
    """
    Prepare features and target
    """
    print("\n" + "=" * 80)
    print("PREPARING DATA")
    print("=" * 80)
    
    X = df.drop(columns=['Attack'])
    y = df['Attack']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"✅ Feature matrix: {X.shape}")
    print(f"✅ Target vector: {y_encoded.shape}")
    print(f"\n📊 Label Encoding:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"  {i}: {class_name}")
    
    return X, y_encoded, label_encoder

def create_improved_xgboost():
    """
    Create optimized XGBoost model for DDoS/DoS distinction
    """
    print("\n" + "=" * 80)
    print("CREATING IMPROVED XGBOOST MODEL")
    print("=" * 80)
    
    model = xgb.XGBClassifier(
        # Multi-class configuration
        objective='multi:softprob',     # Probability output for better calibration
        num_class=5,
        eval_metric='mlogloss',
        
        # Tree structure - optimized for complex patterns
        max_depth=10,                   # Deeper trees for subtle differences
        min_child_weight=3,             # Higher for stability
        gamma=0.1,                      # Minimum loss reduction
        
        # Sampling parameters
        subsample=0.85,                 # Slightly higher sampling
        colsample_bytree=0.85,
        colsample_bylevel=0.85,         # Column sampling per level
        
        # Regularization - prevent overfitting
        reg_alpha=0.05,                 # L1 regularization
        reg_lambda=2.0,                 # Stronger L2 regularization
        
        # Learning parameters
        learning_rate=0.05,             # Lower learning rate for better convergence
        n_estimators=300,               # More trees with lower learning rate
        
        # Performance
        tree_method='hist',             # Faster histogram-based algorithm
        random_state=42,
        n_jobs=-1,
        verbosity=0                     # Silent during CV
    )
    
    print("✅ Improved XGBoost configured")
    print(f"  - Objective: multi:softprob (probability-based)")
    print(f"  - Max depth: 10 (deeper for complex patterns)")
    print(f"  - Learning rate: 0.05 (slower, more accurate)")
    print(f"  - N estimators: 300 (more trees)")
    print(f"  - Regularization: L1=0.05, L2=2.0")
    
    return model

def perform_kfold_cv(model, X, y, label_encoder, k=5):
    """
    Perform k-fold cross-validation with comprehensive metrics
    """
    print("\n" + "=" * 80)
    print(f"PERFORMING {k}-FOLD CROSS-VALIDATION")
    print("=" * 80)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrices': [],
        'per_class_metrics': []
    }
    
    print(f"\nTraining with {k}-fold cross-validation...")
    print("-" * 80)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y), 1):
        print(f"\n📊 Fold {fold}/{k}")
        print("-" * 40)
        
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train, verbose=False)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1'].append(f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fold_results['confusion_matrices'].append(cm)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        fold_results['per_class_metrics'].append({
            'precision': precision_per_class,
            'recall': recall_per_class,
            'f1': f1_per_class
        })
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Time:      {train_time:.2f}s")
    
    # Calculate average metrics
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 OVERALL METRICS (Average across {k} folds)")
    print("-" * 80)
    print(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})")
    print(f"Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})")
    print(f"Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})")
    print(f"F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})")
    
    # Per-class performance
    print(f"\n📊 PER-CLASS PERFORMANCE (Average across {k} folds)")
    print("-" * 80)
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for i, class_name in enumerate(label_encoder.classes_):
        avg_precision = np.mean([fold['precision'][i] for fold in fold_results['per_class_metrics']])
        avg_recall = np.mean([fold['recall'][i] for fold in fold_results['per_class_metrics']])
        avg_f1 = np.mean([fold['f1'][i] for fold in fold_results['per_class_metrics']])
        
        print(f"{class_name:<20} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")
    
    # Average confusion matrix
    avg_cm = np.mean(fold_results['confusion_matrices'], axis=0).astype(int)
    print(f"\n📊 AVERAGE CONFUSION MATRIX")
    print("-" * 80)
    print("Rows: True Labels, Columns: Predicted Labels")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(avg_cm)
    
    return fold_results, scaler

def train_final_model(model, X, y, scaler):
    """
    Train final model on entire dataset
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL ON COMPLETE DATASET")
    print("=" * 80)
    
    X_scaled = scaler.transform(X)
    
    start_time = time.time()
    model.fit(X_scaled, y, verbose=True)
    training_time = time.time() - start_time
    
    print(f"\n✅ Final model trained")
    print(f"⏱️  Training time: {training_time:.2f} seconds")
    
    return model, training_time

def analyze_feature_importance(model, feature_names, label_encoder):
    """
    Analyze feature importance
    """
    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 most important features:")
    print("-" * 80)
    for i, (feature, importance) in enumerate(sorted_importance[:15], 1):
        print(f"{i:2d}. {feature:<35s}: {importance:.4f}")
    
    return sorted_importance

def save_results(fold_results, final_training_time, feature_importance, label_encoder, k):
    """
    Save comprehensive results
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Text file
    with open('improved_multiclass_kfold_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"IMPROVED MULTI-CLASS ATTACK DETECTION - {k}-FOLD CV RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ATTACK TYPES:\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(label_encoder.classes_):
            f.write(f"  {i}: {name}\n")
        
        f.write(f"\n{k}-FOLD CROSS-VALIDATION RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})\n")
        f.write(f"Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})\n")
        f.write(f"Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})\n")
        f.write(f"F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})\n\n")
        
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(label_encoder.classes_):
            avg_p = np.mean([fold['precision'][i] for fold in fold_results['per_class_metrics']])
            avg_r = np.mean([fold['recall'][i] for fold in fold_results['per_class_metrics']])
            avg_f1 = np.mean([fold['f1'][i] for fold in fold_results['per_class_metrics']])
            f.write(f"{name:<20} {avg_p:<12.4f} {avg_r:<12.4f} {avg_f1:<12.4f}\n")
        
        f.write(f"\nFINAL MODEL TRAINING TIME: {final_training_time:.2f} seconds\n\n")
        
        f.write("TOP 15 FEATURE IMPORTANCE:\n")
        f.write("-" * 80 + "\n")
        for i, (feature, importance) in enumerate(feature_importance[:15], 1):
            f.write(f"{i:2d}. {feature:<35s}: {importance:.4f}\n")
    
    # JSON file
    results_json = {
        'k_folds': k,
        'cv_results': {
            'accuracy': {
                'mean': float(np.mean(fold_results['accuracy'])),
                'std': float(np.std(fold_results['accuracy'])),
                'values': [float(v) for v in fold_results['accuracy']]
            },
            'precision': {
                'mean': float(np.mean(fold_results['precision'])),
                'std': float(np.std(fold_results['precision'])),
                'values': [float(v) for v in fold_results['precision']]
            },
            'recall': {
                'mean': float(np.mean(fold_results['recall'])),
                'std': float(np.std(fold_results['recall'])),
                'values': [float(v) for v in fold_results['recall']]
            },
            'f1_score': {
                'mean': float(np.mean(fold_results['f1'])),
                'std': float(np.std(fold_results['f1'])),
                'values': [float(v) for v in fold_results['f1']]
            }
        },
        'per_class_performance': {},
        'attack_types': {i: name for i, name in enumerate(label_encoder.classes_)},
        'final_training_time': float(final_training_time),
        'feature_importance': {k: float(v) for k, v in feature_importance[:15]}
    }
    
    for i, name in enumerate(label_encoder.classes_):
        results_json['per_class_performance'][name] = {
            'precision': float(np.mean([fold['precision'][i] for fold in fold_results['per_class_metrics']])),
            'recall': float(np.mean([fold['recall'][i] for fold in fold_results['per_class_metrics']])),
            'f1_score': float(np.mean([fold['f1'][i] for fold in fold_results['per_class_metrics']]))
        }
    
    with open('improved_multiclass_kfold_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print("✅ Results saved to 'improved_multiclass_kfold_results.txt'")
    print("✅ Results saved to 'improved_multiclass_kfold_results.json'")

def save_model(model, scaler, label_encoder):
    """
    Save the final model and preprocessing objects
    """
    model.save_model('improved_multiclass_model.json')
    np.save('improved_scaler.npy', scaler)
    np.save('improved_label_encoder.npy', label_encoder.classes_)
    
    print("✅ Model saved to 'improved_multiclass_model.json'")
    print("✅ Scaler saved to 'improved_scaler.npy'")
    print("✅ Label encoder saved to 'improved_label_encoder.npy'")

def main():
    """
    Main pipeline
    """
    print("\n" + "=" * 80)
    print("IMPROVED MULTI-CLASS ATTACK DETECTION WITH K-FOLD CV")
    print("Dataset: NF-BoT-IoT v1")
    print("=" * 80 + "\n")
    
    # Configuration
    file_path = "/Users/saniyalahoti/Downloads/MAS-LSTM-1/v1_dataset/NF-BoT-IoT.csv"
    K_FOLDS = 5  # Using 5-fold CV
    
    # Load and preprocess
    df = load_and_preprocess_data(file_path)
    
    # Feature engineering
    df_enhanced = engineer_ddos_dos_features(df)
    
    # Balance dataset
    df_balanced = balance_dataset(df_enhanced, target_size=20000)
    
    # Prepare data
    X, y, label_encoder = prepare_data(df_balanced)
    
    # Create model
    model = create_improved_xgboost()
    
    # Perform k-fold cross-validation
    fold_results, scaler = perform_kfold_cv(model, X, y, label_encoder, k=K_FOLDS)
    
    # Train final model
    final_model, training_time = train_final_model(model, X, y, scaler)
    
    # Feature importance
    feature_names = [col for col in df_balanced.columns if col != 'Attack']
    feature_importance = analyze_feature_importance(final_model, feature_names, label_encoder)
    
    # Save results
    save_results(fold_results, training_time, feature_importance, label_encoder, K_FOLDS)
    save_model(final_model, scaler, label_encoder)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ {K_FOLDS}-Fold CV Accuracy: {np.mean(fold_results['accuracy'])*100:.2f}%")
    print(f"✅ {K_FOLDS}-Fold CV F1-Score: {np.mean(fold_results['f1']):.4f}")
    print(f"✅ Model trained and saved successfully")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

