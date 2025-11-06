import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve, precision_recall_curve,
                            matthews_corrcoef, cohen_kappa_score)
import warnings
warnings.filterwarnings('ignore')

def load_model_and_data():
    """
    Load the trained model and test data
    """
    print("=" * 80)
    print("LOADING MODEL AND TEST DATA")
    print("=" * 80)
    
    # Load trained model
    model = xgb.XGBClassifier()
    model.load_model('xgboost_anomaly_model.json')
    print("✅ Model loaded successfully")
    
    # Load test data
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    
    print(f"✅ Test data loaded: {X_test.shape}")
    print(f"✅ Test labels loaded: {y_test.shape}")
    
    return model, X_test, y_test

def comprehensive_testing(model, X_test, y_test):
    """
    Perform comprehensive testing with additional metrics
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE MODEL TESTING")
    print("=" * 80)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate all metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)  # Same as recall
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    
    print("\n📊 PRIMARY METRICS")
    print("-" * 80)
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision:          {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"ROC-AUC:            {auc:.4f}")
    
    print("\n📊 ADDITIONAL METRICS")
    print("-" * 80)
    print(f"Matthews Correlation Coefficient: {mcc:.4f}")
    print(f"Cohen's Kappa Score:              {kappa:.4f}")
    print(f"Specificity:                      {specificity:.4f}")
    print(f"False Positive Rate:              {fpr:.4f}")
    print(f"False Negative Rate:              {fnr:.4f}")
    
    print("\n📊 CONFUSION MATRIX BREAKDOWN")
    print("-" * 80)
    print(f"True Negatives (TN):   {tn:6d} (Correct Benign predictions)")
    print(f"False Positives (FP):  {fp:6d} (Benign misclassified as Malicious)")
    print(f"False Negatives (FN):  {fn:6d} (Malicious misclassified as Benign)")
    print(f"True Positives (TP):   {tp:6d} (Correct Malicious predictions)")
    print(f"\nTotal Test Samples:    {len(y_test):6d}")
    print(f"Correct Predictions:   {tn + tp:6d} ({(tn+tp)/len(y_test)*100:.2f}%)")
    print(f"Incorrect Predictions: {fp + fn:6d} ({(fp+fn)/len(y_test)*100:.2f}%)")
    
    print("\n📊 DETAILED CLASSIFICATION REPORT")
    print("-" * 80)
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'], digits=4))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc,
        'mcc': mcc,
        'kappa': kappa,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def threshold_analysis(y_test, y_pred_proba):
    """
    Analyze different probability thresholds
    """
    print("\n" + "=" * 80)
    print("THRESHOLD ANALYSIS")
    print("=" * 80)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\nPerformance at different probability thresholds:")
    print("-" * 80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred_thresh)
        prec = precision_score(y_test, y_pred_thresh)
        rec = recall_score(y_test, y_pred_thresh)
        f1 = f1_score(y_test, y_pred_thresh)
        print(f"{threshold:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

def cross_validation_simulation(model, X_test, y_test):
    """
    Simulate cross-validation by splitting test set into chunks
    """
    print("\n" + "=" * 80)
    print("STABILITY TEST (Test Set Chunking)")
    print("=" * 80)
    
    chunk_size = len(X_test) // 5  # Split into 5 chunks
    accuracies = []
    f1_scores = []
    
    print("\nTesting model stability across different test subsets:")
    print("-" * 80)
    
    for i in range(5):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < 4 else len(X_test)
        
        X_chunk = X_test[start_idx:end_idx]
        y_chunk = y_test[start_idx:end_idx]
        
        y_pred = model.predict(X_chunk)
        acc = accuracy_score(y_chunk, y_pred)
        f1 = f1_score(y_chunk, y_pred)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        
        print(f"Chunk {i+1}: Accuracy = {acc:.4f}, F1-Score = {f1:.4f}")
    
    print(f"\nMean Accuracy:  {np.mean(accuracies):.4f} (±{np.std(accuracies):.4f})")
    print(f"Mean F1-Score:  {np.mean(f1_scores):.4f} (±{np.std(f1_scores):.4f})")
    print(f"Min Accuracy:   {np.min(accuracies):.4f}")
    print(f"Max Accuracy:   {np.max(accuracies):.4f}")

def error_analysis(model, X_test, y_test, y_pred):
    """
    Analyze misclassified samples
    """
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)
    
    # Find misclassified samples
    misclassified = y_test != y_pred
    n_misclassified = np.sum(misclassified)
    
    # False positives and false negatives
    false_positives = (y_test == 0) & (y_pred == 1)
    false_negatives = (y_test == 1) & (y_pred == 0)
    
    n_fp = np.sum(false_positives)
    n_fn = np.sum(false_negatives)
    
    print(f"\nTotal Misclassifications: {n_misclassified} out of {len(y_test)} ({n_misclassified/len(y_test)*100:.2f}%)")
    print(f"  - False Positives: {n_fp} ({n_fp/len(y_test)*100:.2f}%)")
    print(f"  - False Negatives: {n_fn} ({n_fn/len(y_test)*100:.2f}%)")
    
    # Get prediction probabilities for misclassified samples
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n📊 MISCLASSIFICATION CONFIDENCE ANALYSIS")
    print("-" * 80)
    
    if n_fp > 0:
        fp_probs = y_pred_proba[false_positives]
        print(f"False Positive Probabilities:")
        print(f"  Mean:   {np.mean(fp_probs):.4f}")
        print(f"  Median: {np.median(fp_probs):.4f}")
        print(f"  Min:    {np.min(fp_probs):.4f}")
        print(f"  Max:    {np.max(fp_probs):.4f}")
    
    if n_fn > 0:
        fn_probs = y_pred_proba[false_negatives]
        print(f"\nFalse Negative Probabilities:")
        print(f"  Mean:   {np.mean(fn_probs):.4f}")
        print(f"  Median: {np.median(fn_probs):.4f}")
        print(f"  Min:    {np.min(fn_probs):.4f}")
        print(f"  Max:    {np.max(fn_probs):.4f}")

def save_testing_results(results):
    """
    Save testing results to file
    """
    print("\n" + "=" * 80)
    print("SAVING TESTING RESULTS")
    print("=" * 80)
    
    with open('binary_model_testing_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("XGBOOST BINARY MODEL - COMPREHENSIVE TESTING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PRIMARY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision:          {results['precision']:.4f}\n")
        f.write(f"Recall:             {results['recall']:.4f}\n")
        f.write(f"F1-Score:           {results['f1_score']:.4f}\n")
        f.write(f"ROC-AUC:            {results['roc_auc']:.4f}\n\n")
        
        f.write("ADDITIONAL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Matthews Correlation Coefficient: {results['mcc']:.4f}\n")
        f.write(f"Cohen's Kappa Score:              {results['kappa']:.4f}\n")
        f.write(f"Specificity:                      {results['specificity']:.4f}\n")
        f.write(f"False Positive Rate:              {results['fpr']:.4f}\n")
        f.write(f"False Negative Rate:              {results['fnr']:.4f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")
        cm = results['confusion_matrix']
        f.write(f"True Negatives:  {cm[0,0]}\n")
        f.write(f"False Positives: {cm[0,1]}\n")
        f.write(f"False Negatives: {cm[1,0]}\n")
        f.write(f"True Positives:  {cm[1,1]}\n")
    
    print("✅ Testing results saved to 'binary_model_testing_results.txt'")

def main():
    """
    Main testing pipeline
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BINARY MODEL TESTING")
    print("=" * 80 + "\n")
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Comprehensive testing
    results = comprehensive_testing(model, X_test, y_test)
    
    # Threshold analysis
    threshold_analysis(y_test, results['probabilities'])
    
    # Stability test
    cross_validation_simulation(model, X_test, y_test)
    
    # Error analysis
    error_analysis(model, X_test, y_test, results['predictions'])
    
    # Save results
    save_testing_results(results)
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETE!")
    print("=" * 80)
    print(f"✅ Model thoroughly tested")
    print(f"✅ Test Accuracy: {results['accuracy']*100:.2f}%")
    print(f"✅ Matthews Correlation: {results['mcc']:.4f}")
    print(f"✅ Cohen's Kappa: {results['kappa']:.4f}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

