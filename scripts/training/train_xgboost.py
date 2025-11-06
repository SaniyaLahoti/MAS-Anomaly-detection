import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
import time
import json
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data():
    """
    Load the preprocessed data from numpy files
    """
    print("=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80)
    
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        print(f"✅ Training data loaded: {X_train.shape}")
        print(f"✅ Test data loaded: {X_test.shape}")
        print(f"✅ Training labels loaded: {y_train.shape}")
        print(f"✅ Test labels loaded: {y_test.shape}")
        
        # Check class distribution
        print(f"\nTraining class distribution:")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples ({count/len(y_train)*100:.1f}%)")
            
        print(f"\nTest class distribution:")
        unique, counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  Class {label}: {count} samples ({count/len(y_test)*100:.1f}%)")
        
        return X_train, X_test, y_train, y_test
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please run the preprocessing script first!")
        return None, None, None, None

def create_xgboost_model():
    """
    Create and configure XGBoost model for anomaly detection
    """
    print("\n" + "=" * 80)
    print("CREATING XGBOOST MODEL")
    print("=" * 80)
    
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        use_label_encoder=False
    )
    
    print("XGBoost model created with parameters:")
    print(f"  - Objective: {xgb_model.objective}")
    print(f"  - Eval metric: {xgb_model.eval_metric}")
    print(f"  - Max depth: {xgb_model.max_depth}")
    print(f"  - Learning rate: {xgb_model.learning_rate}")
    print(f"  - N estimators: {xgb_model.n_estimators}")
    print(f"  - Subsample: {xgb_model.subsample}")
    print(f"  - Colsample bytree: {xgb_model.colsample_bytree}")
    
    return xgb_model

def train_model(model, X_train, y_train, X_test, y_test):
    """
    Train the XGBoost model with evaluation
    """
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST MODEL")
    print("=" * 80)
    
    print("Training in progress...")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Start training timer
    start_time = time.time()
    
    # Train with evaluation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=True
    )
    
    # End training timer
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed!")
    print(f"⏱️  Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"⏱️  Training time per sample: {training_time/len(X_train):.6f} seconds")
    
    return model, training_time

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Evaluate the trained model on both train and test sets
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    
    # Predictions on training set
    print("\n📊 TRAINING SET EVALUATION")
    print("-" * 80)
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    train_pred_time = time.time() - start_time
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)
    
    print(f"Accuracy:  {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall:    {train_recall:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")
    print(f"ROC-AUC:   {train_auc:.4f}")
    print(f"Prediction time: {train_pred_time:.4f} seconds")
    
    print("\nConfusion Matrix (Training):")
    train_cm = confusion_matrix(y_train, y_train_pred)
    print(train_cm)
    print(f"  TN: {train_cm[0,0]}, FP: {train_cm[0,1]}")
    print(f"  FN: {train_cm[1,0]}, TP: {train_cm[1,1]}")
    
    # Predictions on test set
    print("\n📊 TEST SET EVALUATION")
    print("-" * 80)
    start_time = time.time()
    y_test_pred = model.predict(X_test)
    test_pred_time = time.time() - start_time
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)
    
    print(f"Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_auc:.4f}")
    print(f"Prediction time: {test_pred_time:.4f} seconds")
    
    print("\nConfusion Matrix (Test):")
    test_cm = confusion_matrix(y_test, y_test_pred)
    print(test_cm)
    print(f"  TN: {test_cm[0,0]}, FP: {test_cm[0,1]}")
    print(f"  FN: {test_cm[1,0]}, TP: {test_cm[1,1]}")
    
    print("\nDetailed Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malicious']))
    
    # Feature importance
    print("\n📈 FEATURE IMPORTANCE")
    print("-" * 80)
    feature_names = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO', 
                    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 
                    'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("Features ranked by importance:")
    for i, (feature, importance) in enumerate(sorted_importance, 1):
        print(f"  {i}. {feature:30s}: {importance:.4f}")
    
    return {
        'train': {
            'accuracy': train_accuracy,
            'precision': train_precision,
            'recall': train_recall,
            'f1_score': train_f1,
            'roc_auc': train_auc,
            'confusion_matrix': train_cm.tolist(),
            'prediction_time': train_pred_time
        },
        'test': {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': test_f1,
            'roc_auc': test_auc,
            'confusion_matrix': test_cm.tolist(),
            'prediction_time': test_pred_time
        },
        'feature_importance': dict(sorted_importance)
    }

def save_results(results, training_time):
    """
    Save training results to file
    """
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    # Save detailed results to text file
    with open('xgboost_training_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("XGBOOST ANOMALY DETECTION - TRAINING RESULTS\n")
        f.write("Dataset: NF-BoT-IoT v1\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("TRAINING SET RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {results['train']['accuracy']:.4f} ({results['train']['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['train']['precision']:.4f}\n")
        f.write(f"Recall:    {results['train']['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['train']['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:   {results['train']['roc_auc']:.4f}\n")
        f.write(f"Prediction Time: {results['train']['prediction_time']:.4f} seconds\n\n")
        
        f.write("TEST SET RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:  {results['test']['accuracy']:.4f} ({results['test']['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['test']['precision']:.4f}\n")
        f.write(f"Recall:    {results['test']['recall']:.4f}\n")
        f.write(f"F1-Score:  {results['test']['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:   {results['test']['roc_auc']:.4f}\n")
        f.write(f"Prediction Time: {results['test']['prediction_time']:.4f} seconds\n\n")
        
        f.write("TRAINING TIME\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Training Time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)\n\n")
        
        f.write("FEATURE IMPORTANCE\n")
        f.write("-" * 80 + "\n")
        for i, (feature, importance) in enumerate(results['feature_importance'].items(), 1):
            f.write(f"{i}. {feature:30s}: {importance:.4f}\n")
    
    # Save results as JSON for easy loading (convert numpy types to Python types)
    results_json = {
        'train': {k: float(v) if not isinstance(v, list) else v for k, v in results['train'].items()},
        'test': {k: float(v) if not isinstance(v, list) else v for k, v in results['test'].items()},
        'feature_importance': {k: float(v) for k, v in results['feature_importance'].items()},
        'training_time': float(training_time)
    }
    with open('xgboost_training_results.json', 'w') as f:
        json.dump(results_json, f, indent=4)
    
    print("✅ Results saved to 'xgboost_training_results.txt'")
    print("✅ Results saved to 'xgboost_training_results.json'")

def save_model(model):
    """
    Save the trained model
    """
    model_filename = 'xgboost_anomaly_model.json'
    model.save_model(model_filename)
    print(f"✅ Model saved to '{model_filename}'")

def main():
    """
    Main training pipeline
    """
    print("\n" + "=" * 80)
    print("XGBOOST ANOMALY DETECTION - TRAINING PIPELINE")
    print("Dataset: NF-BoT-IoT v1")
    print("=" * 80 + "\n")
    
    # Load data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    if X_train is None:
        return
    
    # Create model
    model = create_xgboost_model()
    
    # Train model
    trained_model, training_time = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    results = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
    
    # Save results and model
    save_results(results, training_time)
    save_model(trained_model)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✅ Model trained successfully")
    print(f"✅ Test Accuracy: {results['test']['accuracy']*100:.2f}%")
    print(f"✅ Test F1-Score: {results['test']['f1_score']:.4f}")
    print(f"✅ Test ROC-AUC: {results['test']['roc_auc']:.4f}")
    print(f"✅ Training Time: {training_time:.2f} seconds")
    print("=" * 80 + "\n")
    
    return trained_model, results

if __name__ == "__main__":
    model, results = main()

