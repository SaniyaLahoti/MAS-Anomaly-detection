import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
import xgboost as xgb
import time
import json
import warnings
warnings.filterwarnings('ignore')

def load_and_engineer_features(file_path):
    """
    Load data and create enhanced features
    """
    print("=" * 80)
    print("HIERARCHICAL CLASSIFICATION: STAGE 1 & 2")
    print("=" * 80)
    
    df = pd.read_csv(file_path)
    print(f"✅ Dataset loaded: {df.shape}")
    
    print("\n📊 ORIGINAL ATTACK DISTRIBUTION")
    print("-" * 80)
    print(df['Attack'].value_counts())
    
    # Remove IP addresses
    df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'Label'])
    
    # Create enhanced features
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
    
    # Add cost-sensitive features for DDoS/DoS
    df['TCP_PACKET_INTERACTION'] = df['TCP_FLAGS'] * df['IN_PKTS']
    df['PROTOCOL_PORT_COMBO'] = df['PROTOCOL'] * df['L4_DST_PORT']
    df['FLOW_INTENSITY'] = (df['IN_PKTS'] + df['OUT_PKTS']) / (df['FLOW_DURATION_MILLISECONDS'] + 1) * df['AVG_PACKET_SIZE']
    
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"✅ Total features: {len(df.columns) - 1}")
    
    return df

def create_stage1_labels(df):
    """
    Stage 1: Merge DDoS and DoS into single 'DOS' class
    Classes: Benign, DOS, Reconnaissance, Theft
    """
    print("\n" + "=" * 80)
    print("STAGE 1: 4-CLASS CLASSIFICATION (DDoS+DoS merged)")
    print("=" * 80)
    
    df_stage1 = df.copy()
    
    # Merge DDoS and DoS
    df_stage1['Attack_Stage1'] = df_stage1['Attack'].replace({
        'DDoS': 'DOS',
        'DoS': 'DOS'
    })
    
    print("\nStage 1 Attack Distribution:")
    print(df_stage1['Attack_Stage1'].value_counts())
    
    return df_stage1

def balance_dataset_stage1(df, target_size=25000):
    """
    Balance dataset for Stage 1
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
    print("\nClass distribution:")
    print(df_balanced['Attack_Stage1'].value_counts())
    
    return df_balanced

def prepare_stage2_data(df_original):
    """
    Stage 2: Only DDoS vs DoS samples
    """
    print("\n" + "=" * 80)
    print("STAGE 2: BINARY CLASSIFICATION (DDoS vs DoS only)")
    print("=" * 80)
    
    # Extract only DDoS and DoS samples
    df_stage2 = df_original[df_original['Attack'].isin(['DDoS', 'DoS'])].copy()
    
    print(f"Stage 2 samples: {df_stage2.shape}")
    print("\nAttack distribution:")
    print(df_stage2['Attack'].value_counts())
    
    # Balance DDoS and DoS equally
    target_size = 30000
    balanced_dfs = []
    
    for attack_type in df_stage2['Attack'].unique():
        df_class = df_stage2[df_stage2['Attack'] == attack_type]
        
        if len(df_class) > target_size:
            df_resampled = resample(df_class, replace=False, n_samples=target_size, random_state=42)
        else:
            df_resampled = resample(df_class, replace=True, n_samples=target_size, random_state=42)
        
        balanced_dfs.append(df_resampled)
    
    df_balanced = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n✅ Balanced Stage 2: {df_balanced.shape}")
    print("Class distribution:")
    print(df_balanced['Attack'].value_counts())
    
    return df_balanced

def train_stage1_model(df_stage1, k_folds=5):
    """
    Train Stage 1: 4-class model
    """
    print("\n" + "=" * 80)
    print("TRAINING STAGE 1 MODEL (4-class)")
    print("=" * 80)
    
    # Prepare data
    X = df_stage1.drop(columns=['Attack', 'Attack_Stage1'])
    y = df_stage1['Attack_Stage1']
    
    # Encode labels
    le_stage1 = LabelEncoder()
    y_encoded = le_stage1.fit_transform(y)
    
    print(f"\nFeatures: {X.shape}")
    print(f"Classes: {list(le_stage1.classes_)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model
    model_stage1 = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        max_depth=10,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    print(f"\n🔄 {k_folds}-Fold Cross-Validation...")
    print("-" * 80)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_encoded), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        model_stage1.fit(X_train, y_train, verbose=False)
        y_pred = model_stage1.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1'].append(f1)
        
        print(f"Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
    
    print(f"\n📊 STAGE 1 CV RESULTS:")
    print(f"  Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})")
    print(f"  Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})")
    print(f"  Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})")
    print(f"  F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})")
    
    # Train final model on all data
    print("\n🔄 Training final Stage 1 model...")
    model_stage1.fit(X_scaled, y_encoded, verbose=False)
    
    return model_stage1, scaler, le_stage1, fold_results

def train_stage2_model(df_stage2, k_folds=5):
    """
    Train Stage 2: DDoS vs DoS specialized model
    """
    print("\n" + "=" * 80)
    print("TRAINING STAGE 2 MODEL (DDoS vs DoS)")
    print("=" * 80)
    
    # Prepare data
    X = df_stage2.drop(columns=['Attack'])
    y = df_stage2['Attack']
    
    # Encode labels
    le_stage2 = LabelEncoder()
    y_encoded = le_stage2.fit_transform(y)
    
    print(f"\nFeatures: {X.shape}")
    print(f"Classes: {list(le_stage2.classes_)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create specialized model with cost-sensitive learning
    model_stage2 = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=12,                    # Deeper for subtle patterns
        learning_rate=0.03,              # Even lower learning rate
        n_estimators=500,                # Many more trees
        subsample=0.9,
        colsample_bytree=0.9,
        colsample_bylevel=0.9,
        reg_alpha=0.01,                  # Lower regularization
        reg_lambda=3.0,                  # Higher L2
        gamma=0.1,
        min_child_weight=5,
        scale_pos_weight=1,              # Balanced classes
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    
    print(f"\n🔄 {k_folds}-Fold Cross-Validation...")
    print("-" * 80)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y_encoded), 1):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
        
        model_stage2.fit(X_train, y_train, verbose=False)
        y_pred = model_stage2.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1'].append(f1)
        
        print(f"Fold {fold}: Acc={acc:.4f}, F1={f1:.4f}")
    
    print(f"\n📊 STAGE 2 CV RESULTS:")
    print(f"  Accuracy:  {np.mean(fold_results['accuracy']):.4f} (±{np.std(fold_results['accuracy']):.4f})")
    print(f"  Precision: {np.mean(fold_results['precision']):.4f} (±{np.std(fold_results['precision']):.4f})")
    print(f"  Recall:    {np.mean(fold_results['recall']):.4f} (±{np.std(fold_results['recall']):.4f})")
    print(f"  F1-Score:  {np.mean(fold_results['f1']):.4f} (±{np.std(fold_results['f1']):.4f})")
    
    # Train final model on all data
    print("\n🔄 Training final Stage 2 model...")
    model_stage2.fit(X_scaled, y_encoded, verbose=False)
    
    return model_stage2, scaler, le_stage2, fold_results

def evaluate_hierarchical_model(df_test, model_stage1, scaler_stage1, le_stage1,
                                model_stage2, scaler_stage2, le_stage2):
    """
    Evaluate the complete hierarchical model
    """
    print("\n" + "=" * 80)
    print("EVALUATING HIERARCHICAL MODEL")
    print("=" * 80)
    
    # Prepare test data
    X_test = df_test.drop(columns=['Attack'])
    y_test_true = df_test['Attack']
    
    # Scale features
    X_test_scaled_s1 = scaler_stage1.transform(X_test)
    
    # Stage 1 prediction
    y_pred_stage1 = model_stage1.predict(X_test_scaled_s1)
    y_pred_stage1_labels = le_stage1.inverse_transform(y_pred_stage1)
    
    # Initialize final predictions
    y_pred_final = y_pred_stage1_labels.copy()
    
    # Stage 2 prediction for DOS samples
    dos_mask = (y_pred_stage1_labels == 'DOS')
    
    if dos_mask.any():
        X_dos = X_test[dos_mask]
        X_dos_scaled = scaler_stage2.transform(X_dos)
        y_pred_stage2 = model_stage2.predict(X_dos_scaled)
        y_pred_stage2_labels = le_stage2.inverse_transform(y_pred_stage2)
        
        # Replace DOS with specific predictions
        y_pred_final[dos_mask] = y_pred_stage2_labels
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_true, y_pred_final)
    precision = precision_score(y_test_true, y_pred_final, average='weighted', zero_division=0)
    recall = recall_score(y_test_true, y_pred_final, average='weighted', zero_division=0)
    f1 = f1_score(y_test_true, y_pred_final, average='weighted', zero_division=0)
    
    print(f"\n📊 HIERARCHICAL MODEL PERFORMANCE:")
    print("-" * 80)
    print(f"Overall Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall:    {recall:.4f}")
    print(f"Overall F1-Score:  {f1:.4f}")
    
    # Per-class performance
    print(f"\n📊 PER-CLASS PERFORMANCE:")
    print("-" * 80)
    print(classification_report(y_test_true, y_pred_final, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_true, y_pred_final, 
                         labels=['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft'])
    print(f"\n📊 CONFUSION MATRIX:")
    print("-" * 80)
    print("Classes: ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']")
    print(cm)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'predictions': y_pred_final
    }

def save_results(stage1_cv, stage2_cv, final_results):
    """
    Save all results
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    with open('hierarchical_model_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HIERARCHICAL CLASSIFICATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("STAGE 1 (4-Class: Benign, DOS, Reconnaissance, Theft):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {np.mean(stage1_cv['accuracy']):.4f} (±{np.std(stage1_cv['accuracy']):.4f})\n")
        f.write(f"F1-Score:  {np.mean(stage1_cv['f1']):.4f} (±{np.std(stage1_cv['f1']):.4f})\n\n")
        
        f.write("STAGE 2 (Binary: DDoS vs DoS):\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {np.mean(stage2_cv['accuracy']):.4f} (±{np.std(stage2_cv['accuracy']):.4f})\n")
        f.write(f"F1-Score:  {np.mean(stage2_cv['f1']):.4f} (±{np.std(stage2_cv['f1']):.4f})\n\n")
        
        f.write("FINAL HIERARCHICAL MODEL:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Overall Accuracy:  {final_results['accuracy']:.4f}\n")
        f.write(f"Overall Precision: {final_results['precision']:.4f}\n")
        f.write(f"Overall Recall:    {final_results['recall']:.4f}\n")
        f.write(f"Overall F1-Score:  {final_results['f1_score']:.4f}\n")
    
    # JSON
    final_results_clean = {k: v for k, v in final_results.items() if k != 'predictions'}
    results_json = {
        'stage1_cv': {k: [float(v) for v in vals] for k, vals in stage1_cv.items()},
        'stage2_cv': {k: [float(v) for v in vals] for k, vals in stage2_cv.items()},
        'final': final_results_clean
    }
    
    with open('hierarchical_model_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print("✅ Results saved to 'hierarchical_model_results.txt'")
    print("✅ Results saved to 'hierarchical_model_results.json'")

def main():
    """
    Main hierarchical classification pipeline
    """
    print("\n" + "=" * 80)
    print("HIERARCHICAL DDOS/DOS CLASSIFICATION")
    print("=" * 80 + "\n")
    
    file_path = "/Users/saniyalahoti/Downloads/MAS-LSTM-1/v1_dataset/NF-BoT-IoT.csv"
    
    # Load and engineer features
    df = load_and_engineer_features(file_path)
    
    # Stage 1: Prepare 4-class data
    df_stage1 = create_stage1_labels(df)
    df_stage1_balanced = balance_dataset_stage1(df_stage1)
    
    # Stage 2: Prepare DDoS vs DoS data
    df_stage2_balanced = prepare_stage2_data(df)
    
    # Train Stage 1
    model_s1, scaler_s1, le_s1, cv_s1 = train_stage1_model(df_stage1_balanced)
    
    # Train Stage 2
    model_s2, scaler_s2, le_s2, cv_s2 = train_stage2_model(df_stage2_balanced)
    
    # Create test set
    print("\n🧪 Creating test set for final evaluation...")
    test_samples = 3000
    df_test = df.groupby('Attack', group_keys=False).apply(
        lambda x: x.sample(min(len(x), test_samples), random_state=42)
    ).reset_index(drop=True)
    
    # Evaluate hierarchical model
    final_results = evaluate_hierarchical_model(
        df_test, model_s1, scaler_s1, le_s1,
        model_s2, scaler_s2, le_s2
    )
    
    # Save results
    save_results(cv_s1, cv_s2, final_results)
    
    # Save models
    model_s1.save_model('hierarchical_stage1_model.json')
    model_s2.save_model('hierarchical_stage2_model.json')
    np.save('hierarchical_stage1_scaler.npy', scaler_s1)
    np.save('hierarchical_stage2_scaler.npy', scaler_s2)
    np.save('hierarchical_stage1_encoder.npy', le_s1.classes_)
    np.save('hierarchical_stage2_encoder.npy', le_s2.classes_)
    
    print("\n✅ Models saved:")
    print("  - hierarchical_stage1_model.json")
    print("  - hierarchical_stage2_model.json")
    
    print("\n" + "=" * 80)
    print("HIERARCHICAL CLASSIFICATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Final Accuracy: {final_results['accuracy']*100:.2f}%")
    print(f"✅ Final F1-Score: {final_results['f1_score']:.4f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

