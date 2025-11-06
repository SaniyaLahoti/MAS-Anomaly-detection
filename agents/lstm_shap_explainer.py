"""
LSTM SHAP Explainer using DeepExplainer

Provides interpretable explanations for LSTM hierarchical model predictions
using SHAP DeepExplainer optimized for deep learning models.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import warnings
warnings.filterwarnings('ignore')

class LSTMHierarchicalSHAPExplainer:
    """
    SHAP explainer for hierarchical LSTM model
    """
    
    def __init__(self):
        self.model_stage1 = None
        self.model_stage2 = None
        self.scaler_stage1 = None
        self.scaler_stage2 = None
        self.encoder_stage1_classes = None
        self.encoder_stage2_classes = None
        self.feature_names = None
        self.explainer_stage1 = None
        self.explainer_stage2 = None
        
    def load_models_and_data(self):
        """
        Load hierarchical LSTM models and preprocessing artifacts
        """
        print("=" * 80)
        print("LSTM SHAP EXPLAINABILITY SYSTEM - LOADING MODELS")
        print("=" * 80)
        
        # Load models
        self.model_stage1 = load_model('../models/lstm/lstm_hierarchical_stage1_model.h5')
        print("✅ Stage 1 LSTM model loaded")
        
        self.model_stage2 = load_model('../models/lstm/lstm_hierarchical_stage2_model.h5')
        print("✅ Stage 2 LSTM model loaded")
        
        # Load scalers
        s1_mean = np.load('../models/lstm/lstm_hierarchical_s1_scaler_mean.npy')
        s1_scale = np.load('../models/lstm/lstm_hierarchical_s1_scaler_scale.npy')
        self.scaler_stage1 = StandardScaler()
        self.scaler_stage1.mean_ = s1_mean
        self.scaler_stage1.scale_ = s1_scale
        
        s2_mean = np.load('../models/lstm/lstm_hierarchical_s2_scaler_mean.npy')
        s2_scale = np.load('../models/lstm/lstm_hierarchical_s2_scaler_scale.npy')
        self.scaler_stage2 = StandardScaler()
        self.scaler_stage2.mean_ = s2_mean
        self.scaler_stage2.scale_ = s2_scale
        
        # Load encoders
        self.encoder_stage1_classes = np.load('../models/lstm/lstm_hierarchical_s1_encoder.npy', allow_pickle=True)
        self.encoder_stage2_classes = np.load('../models/lstm/lstm_hierarchical_s2_encoder.npy', allow_pickle=True)
        print("✅ Scalers and encoders loaded")
        
        # Load sample data for feature names
        df = pd.read_csv('../datasets/v1_dataset/NF-BoT-IoT.csv')
        df = self._engineer_features(df)
        self.feature_names = [col for col in df.columns if col not in ['Attack', 'Label']]
        print(f"✅ Feature names loaded: {len(self.feature_names)} features")
        
        print(f"\n📊 Classes:")
        print(f"  Stage 1: {list(self.encoder_stage1_classes)}")
        print(f"  Stage 2: {list(self.encoder_stage2_classes)}")
        
        return df
    
    def _engineer_features(self, df):
        """
        Apply feature engineering
        """
        if 'IPV4_SRC_ADDR' in df.columns:
            df = df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'])
        if 'Label' in df.columns:
            df = df.drop(columns=['Label'])
        
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
    
    def create_shap_explainers(self, background_data):
        """
        Create SHAP DeepExplainers for both stages
        """
        print("\n" + "=" * 80)
        print("CREATING LSTM SHAP DEEP EXPLAINERS")
        print("=" * 80)
        
        # Prepare background data
        X_background = background_data.drop(columns=['Attack'])
        X_background_scaled_s1 = self.scaler_stage1.transform(X_background)
        X_background_lstm_s1 = X_background_scaled_s1.reshape((X_background_scaled_s1.shape[0], 1, X_background_scaled_s1.shape[1]))
        
        # Stage 1 - Use GradientExplainer (more stable than DeepExplainer)
        print("\n🔧 Creating Stage 1 GradientExplainer...")
        self.explainer_stage1 = shap.GradientExplainer(
            self.model_stage1,
            X_background_lstm_s1[:50]  # Smaller for speed
        )
        print("✅ Stage 1 GradientExplainer ready")
        
        # Stage 2 - Use GradientExplainer
        print("\n🔧 Creating Stage 2 GradientExplainer...")
        X_background_scaled_s2 = self.scaler_stage2.transform(X_background)
        X_background_lstm_s2 = X_background_scaled_s2.reshape((X_background_scaled_s2.shape[0], 1, X_background_scaled_s2.shape[1]))
        
        self.explainer_stage2 = shap.GradientExplainer(
            self.model_stage2,
            X_background_lstm_s2[:50]  # Smaller for speed
        )
        print("✅ Stage 2 GradientExplainer ready")
    
    def explain_prediction(self, sample_data, actual_label=None):
        """
        Provide detailed SHAP explanation for LSTM prediction
        """
        # Prepare features
        X = sample_data.drop(columns=['Attack']) if 'Attack' in sample_data.columns else sample_data
        
        # CRITICAL: Ensure column order matches what scaler expects
        # The scaler was fit with a specific column order and we MUST match it
        if isinstance(X, pd.DataFrame):
            # Reorder columns to match scaler's expected order
            X = X[self.feature_names]
            X_values = X.values
        else:
            X_values = X
            
        X_scaled_s1 = self.scaler_stage1.transform(X_values)
        X_lstm_s1 = X_scaled_s1.reshape((X_scaled_s1.shape[0], 1, X_scaled_s1.shape[1]))
        
        # Stage 1 prediction
        stage1_pred_proba = self.model_stage1.predict(X_lstm_s1, verbose=0)[0]
        stage1_pred_idx = np.argmax(stage1_pred_proba)
        stage1_pred_label = self.encoder_stage1_classes[stage1_pred_idx]
        stage1_confidence = stage1_pred_proba[stage1_pred_idx]
        
        # Stage 1 SHAP values
        print("🔍 Computing Stage 1 SHAP values...")
        shap_values_s1 = self.explainer_stage1.shap_values(X_lstm_s1)
        
        # Determine final prediction
        if stage1_pred_label in ['Benign', 'Reconnaissance', 'Theft']:
            final_prediction = stage1_pred_label
            final_confidence = stage1_confidence
            used_stage2 = False
            stage2_explanation = None
        else:
            # Stage 2 for DOS
            X_scaled_s2 = self.scaler_stage2.transform(X_values)
            X_lstm_s2 = X_scaled_s2.reshape((X_scaled_s2.shape[0], 1, X_scaled_s2.shape[1]))
            
            stage2_pred_proba = self.model_stage2.predict(X_lstm_s2, verbose=0)[0]
            stage2_pred_idx = np.argmax(stage2_pred_proba)
            final_prediction = self.encoder_stage2_classes[stage2_pred_idx]
            final_confidence = stage2_pred_proba[stage2_pred_idx]
            used_stage2 = True
            
            # Stage 2 SHAP values
            print("🔍 Computing Stage 2 SHAP values...")
            shap_values_s2 = self.explainer_stage2.shap_values(X_lstm_s2)
            
            # Get top features for Stage 2
            if isinstance(shap_values_s2, list):
                if stage2_pred_idx < len(shap_values_s2):
                    stage2_shap = shap_values_s2[stage2_pred_idx][0, 0, :]
                else:
                    stage2_shap = shap_values_s2[0][0, 0, :]
            elif len(shap_values_s2.shape) == 4:
                if shap_values_s2.shape[1] > stage2_pred_idx:
                    stage2_shap = shap_values_s2[0, stage2_pred_idx, 0, :]
                else:
                    stage2_shap = shap_values_s2[0, :, 0, :].mean(axis=0)
            else:
                stage2_shap = shap_values_s2[0, 0, :]
            
            stage2_top_features = self._get_top_contributing_features(
                stage2_shap,
                X.iloc[0],
                top_n=10
            )
            stage2_explanation = stage2_top_features
        
        # Get top features for Stage 1
        # GradientExplainer returns shape (samples, classes, timesteps, features)
        if isinstance(shap_values_s1, list):
            # List of arrays, one per class
            if stage1_pred_idx < len(shap_values_s1):
                stage1_shap = shap_values_s1[stage1_pred_idx][0, 0, :]
            else:
                # Fallback: use first one
                stage1_shap = shap_values_s1[0][0, 0, :]
        elif len(shap_values_s1.shape) == 4:
            # Shape: (samples, classes, timesteps, features)
            if shap_values_s1.shape[1] > stage1_pred_idx:
                stage1_shap = shap_values_s1[0, stage1_pred_idx, 0, :]
            else:
                # Fallback: average across all classes
                stage1_shap = shap_values_s1[0, :, 0, :].mean(axis=0)
        else:
            # Shape: (samples, timesteps, features) - no class dimension
            stage1_shap = shap_values_s1[0, 0, :]
        
        stage1_top_features = self._get_top_contributing_features(
            stage1_shap,
            X.iloc[0],
            top_n=10
        )
        
        # Build explanation
        explanation = self._build_explanation(
            final_prediction,
            final_confidence,
            stage1_pred_label,
            stage1_confidence,
            stage1_top_features,
            used_stage2,
            stage2_explanation,
            X.iloc[0],
            actual_label
        )
        
        return explanation
    
    def _get_top_contributing_features(self, shap_values, feature_values, top_n=10):
        """
        Get top features contributing to prediction
        """
        abs_shap = np.abs(shap_values)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]
        
        contributions = []
        for idx in top_indices:
            contributions.append({
                'feature': self.feature_names[idx],
                'value': float(feature_values.iloc[idx]),
                'shap_value': float(shap_values[idx]),
                'impact': 'PUSHES TOWARDS' if shap_values[idx] > 0 else 'PUSHES AWAY FROM',
                'importance': float(abs_shap[idx])
            })
        
        return contributions
    
    def _build_explanation(self, final_prediction, final_confidence, 
                          stage1_label, stage1_confidence, stage1_features,
                          used_stage2, stage2_features, feature_values, actual_label):
        """
        Build comprehensive explanation
        """
        explanation = {
            'model_type': 'LSTM',
            'prediction': final_prediction,
            'confidence': float(final_confidence),
            'actual_label': actual_label,
            'correct': final_prediction == actual_label if actual_label else None,
            'reasoning': self._generate_reasoning(
                final_prediction, final_confidence, stage1_label, 
                stage1_confidence, stage1_features, used_stage2, stage2_features, feature_values
            ),
            'stage1': {
                'predicted_class': stage1_label,
                'confidence': float(stage1_confidence),
                'top_features': stage1_features
            }
        }
        
        if used_stage2:
            explanation['stage2'] = {
                'predicted_class': final_prediction,
                'confidence': float(final_confidence),
                'top_features': stage2_features
            }
        
        return explanation
    
    def _generate_reasoning(self, final_prediction, final_confidence, stage1_label,
                           stage1_confidence, stage1_features, used_stage2, stage2_features, feature_values):
        """
        Generate human-readable reasoning
        """
        reasoning = []
        
        reasoning.append(f"🎯 LSTM CLASSIFICATION: {final_prediction} ({final_confidence*100:.1f}% confidence)")
        reasoning.append("")
        
        reasoning.append("📊 STAGE 1 ANALYSIS (LSTM):")
        reasoning.append(f"   Detected as: {stage1_label} ({stage1_confidence*100:.1f}% confidence)")
        reasoning.append("")
        reasoning.append("   🔍 Key LSTM-Learned Patterns:")
        
        for i, feat in enumerate(stage1_features[:5], 1):
            direction = "HIGHER" if feat['shap_value'] > 0 else "LOWER"
            reasoning.append(
                f"   {i}. {feat['feature']} = {feat['value']:.4f}"
            )
            reasoning.append(
                f"      → {direction} value {feat['impact']} {stage1_label}"
            )
            reasoning.append(
                f"      → LSTM Impact: {feat['importance']:.4f}"
            )
        
        if used_stage2:
            reasoning.append("")
            reasoning.append("📊 STAGE 2 ANALYSIS (LSTM - DDoS vs DoS):")
            reasoning.append(f"   Refined: {final_prediction} ({final_confidence*100:.1f}% confidence)")
            reasoning.append("")
            reasoning.append("   🔍 Key Distinguishing Patterns:")
            
            for i, feat in enumerate(stage2_features[:5], 1):
                direction = "HIGHER" if feat['shap_value'] > 0 else "LOWER"
                reasoning.append(
                    f"   {i}. {feat['feature']} = {feat['value']:.4f}"
                )
                reasoning.append(
                    f"      → {direction} value {feat['impact']} {final_prediction}"
                )
                reasoning.append(
                    f"      → LSTM Impact: {feat['importance']:.4f}"
                )
        
        reasoning.append("")
        reasoning.append("✅ LSTM MODEL EVIDENCE:")
        reasoning.append(self._generate_attack_proof(
            final_prediction, stage1_features, stage2_features, feature_values, used_stage2
        ))
        
        return "\n".join(reasoning)
    
    def _generate_attack_proof(self, attack_type, stage1_features, 
                               stage2_features, feature_values, used_stage2):
        """
        Generate attack-specific proof
        """
        proof = []
        
        protocol = feature_values.get('PROTOCOL', 0)
        tcp_flags = feature_values.get('TCP_FLAGS', 0)
        packet_rate = feature_values.get('PACKET_RATE', 0)
        byte_asymmetry = feature_values.get('BYTE_ASYMMETRY', 0)
        
        if attack_type == 'Benign':
            proof.append("   🟢 LSTM DETECTED BENIGN PATTERNS:")
            proof.append(f"      • Normal protocol sequences (Protocol: {protocol:.0f})")
            proof.append(f"      • Balanced traffic flow patterns")
            proof.append(f"      • LSTM recognized normal behavior sequences")
            
        elif attack_type == 'DDoS':
            proof.append("   🔴 LSTM DETECTED DDOS PATTERNS:")
            proof.append(f"      • High-volume attack pattern (Rate: {packet_rate:.2f} pkts/ms)")
            proof.append(f"      • LSTM learned distributed attack sequences")
            proof.append(f"      • Protocol: {protocol:.0f}, TCP Flags: {tcp_flags:.0f}")
            proof.append(f"      • Sequential pattern matches DDoS signature")
            
        elif attack_type == 'DoS':
            proof.append("   🔴 LSTM DETECTED DOS PATTERNS:")
            proof.append(f"      • High-volume attack pattern (Rate: {packet_rate:.2f} pkts/ms)")
            proof.append(f"      • LSTM learned single-source attack sequences")
            proof.append(f"      • Protocol: {protocol:.0f}, TCP Flags: {tcp_flags:.0f}")
            proof.append(f"      • Sequential pattern matches DoS signature")
            
        elif attack_type == 'Reconnaissance':
            proof.append("   🟡 LSTM DETECTED RECONNAISSANCE PATTERNS:")
            proof.append(f"      • Scanning behavior sequences detected")
            proof.append(f"      • LSTM recognized probing patterns")
            proof.append(f"      • Low-rate sustained activity (Rate: {packet_rate:.2f} pkts/ms)")
            
        elif attack_type == 'Theft':
            proof.append("   🔴 LSTM DETECTED DATA THEFT PATTERNS:")
            proof.append(f"      • Exfiltration sequences identified")
            proof.append(f"      • LSTM learned unauthorized access patterns")
            proof.append(f"      • Asymmetric data transfer: {byte_asymmetry:.2f}")
        
        return "\n".join(proof)
    
    def explain_multiple_samples(self, df_samples):
        """
        Explain multiple predictions
        """
        print("\n" + "=" * 80)
        print("GENERATING LSTM SHAP EXPLANATIONS")
        print("=" * 80)
        
        explanations = []
        
        for idx in range(len(df_samples)):
            sample = df_samples.iloc[[idx]]
            actual_label = sample['Attack'].values[0] if 'Attack' in sample.columns else None
            
            print(f"\n📝 Explaining sample {idx + 1}/{len(df_samples)} (Actual: {actual_label})...")
            
            explanation = self.explain_prediction(sample, actual_label)
            explanations.append(explanation)
            
            pred = explanation['prediction']
            conf = explanation['confidence'] * 100
            correct = "✅ CORRECT" if explanation['correct'] else "❌ INCORRECT"
            print(f"   LSTM Prediction: {pred} ({conf:.1f}%) - {correct}")
        
        return explanations
    
    def save_explanations(self, explanations, filename='lstm_shap_explanations.json'):
        """
        Save explanations
        """
        serializable = []
        for exp in explanations:
            serializable.append({
                'model_type': exp['model_type'],
                'prediction': exp['prediction'],
                'confidence': float(exp['confidence']),
                'actual_label': exp['actual_label'],
                'correct': exp['correct'],
                'reasoning': exp['reasoning'],
                'stage1': exp['stage1'],
                'stage2': exp.get('stage2', None)
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        print(f"\n✅ LSTM explanations saved to {filename}")

def main():
    """
    Main execution
    """
    print("\n" + "=" * 80)
    print("LSTM SHAP EXPLAINABILITY SYSTEM")
    print("=" * 80 + "\n")
    
    # Initialize
    explainer = LSTMHierarchicalSHAPExplainer()
    
    # Load
    df = explainer.load_models_and_data()
    
    # Background data
    print("\n📊 Preparing background data...")
    background_samples = df.groupby('Attack').apply(
        lambda x: x.sample(min(100, len(x)), random_state=42)
    ).reset_index(drop=True)
    print(f"✅ Background: {len(background_samples)} samples")
    
    # Create explainers
    explainer.create_shap_explainers(background_samples)
    
    # Test samples
    print("\n📊 Selecting test samples...")
    test_samples = df.groupby('Attack').apply(
        lambda x: x.sample(min(5, len(x)), random_state=123)
    ).reset_index(drop=True)
    print(f"✅ Selected {len(test_samples)} test samples")
    
    # Generate explanations
    explanations = explainer.explain_multiple_samples(test_samples)
    
    # Save
    explainer.save_explanations(explanations)
    
    # Summary
    correct = sum(1 for exp in explanations if exp['correct'])
    total = len(explanations)
    
    print("\n" + "=" * 80)
    print("LSTM SHAP EXPLANATION COMPLETE")
    print("=" * 80)
    print(f"✅ Total explained: {total}")
    print(f"✅ Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

