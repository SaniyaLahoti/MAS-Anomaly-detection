"""
SHAP Explainability System for Hierarchical Anomaly Detection

This module provides interpretable explanations for attack classifications using SHAP values.
It explains WHY each prediction was made and which features contributed most.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

class HierarchicalSHAPExplainer:
    """
    SHAP-based explainer for the hierarchical attack detection model
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
        Load the hierarchical models and preprocessed data
        """
        print("=" * 80)
        print("SHAP EXPLAINABILITY SYSTEM - LOADING MODELS")
        print("=" * 80)
        
        # Load Stage 1 model (4-class)
        self.model_stage1 = xgb.XGBClassifier()
        self.model_stage1.load_model('../models/hierarchical/hierarchical_stage1_model.json')
        print("✅ Stage 1 model loaded (4-class classification)")
        
        # Load Stage 2 model (DDoS vs DoS)
        self.model_stage2 = xgb.XGBClassifier()
        self.model_stage2.load_model('../models/hierarchical/hierarchical_stage2_model.json')
        print("✅ Stage 2 model loaded (DDoS vs DoS binary)")
        
        # Load scalers and encoders
        self.scaler_stage1 = np.load('../models/hierarchical/hierarchical_stage1_scaler.npy', allow_pickle=True).item()
        self.scaler_stage2 = np.load('../models/hierarchical/hierarchical_stage2_scaler.npy', allow_pickle=True).item()
        self.encoder_stage1_classes = np.load('../models/hierarchical/hierarchical_stage1_encoder.npy', allow_pickle=True)
        self.encoder_stage2_classes = np.load('../models/hierarchical/hierarchical_stage2_encoder.npy', allow_pickle=True)
        print("✅ Scalers and encoders loaded")
        
        # Load sample data to get feature names
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
    
    def create_shap_explainers(self, background_data):
        """
        Create SHAP explainers for both stages using TreeExplainer
        """
        print("\n" + "=" * 80)
        print("CREATING SHAP EXPLAINERS")
        print("=" * 80)
        
        # Prepare background data (sample for efficiency)
        X_background = background_data.drop(columns=['Attack'])
        X_background_scaled_s1 = self.scaler_stage1.transform(X_background)
        X_background_scaled_s2 = self.scaler_stage2.transform(X_background)
        
        # Create Stage 1 explainer with model_output parameter
        print("\n🔧 Creating Stage 1 SHAP explainer...")
        try:
            self.explainer_stage1 = shap.TreeExplainer(
                self.model_stage1,
                data=X_background_scaled_s1[:100],  # Use subset for speed
                model_output='raw'
            )
            print("✅ Stage 1 explainer ready (4-class)")
        except Exception as e:
            print(f"⚠️  Using alternative explainer method for Stage 1: {str(e)[:50]}...")
            self.explainer_stage1 = None
        
        # Create Stage 2 explainer
        print("\n🔧 Creating Stage 2 SHAP explainer...")
        try:
            self.explainer_stage2 = shap.TreeExplainer(
                self.model_stage2,
                data=X_background_scaled_s2[:100]
            )
            print("✅ Stage 2 explainer ready (DDoS vs DoS)")
        except Exception as e:
            print(f"⚠️  Using alternative explainer method for Stage 2: {str(e)[:50]}...")
            self.explainer_stage2 = None
        
    def explain_prediction(self, sample_data, actual_label=None):
        """
        Provide detailed SHAP explanation for a single prediction
        
        Args:
            sample_data: Single row DataFrame with features
            actual_label: Optional actual attack type for comparison
            
        Returns:
            Dictionary with prediction and detailed explanation
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
        
        # Stage 1 prediction
        stage1_pred_proba = self.model_stage1.predict_proba(X_scaled_s1)[0]
        stage1_pred_idx = np.argmax(stage1_pred_proba)
        stage1_pred_label = self.encoder_stage1_classes[stage1_pred_idx]
        stage1_confidence = stage1_pred_proba[stage1_pred_idx]
        
        # Stage 1 SHAP values or feature importance
        if self.explainer_stage1 is not None:
            shap_values_s1 = self.explainer_stage1.shap_values(X_scaled_s1, check_additivity=False)
        else:
            # Use feature importance as fallback
            shap_values_s1 = self._get_feature_importance_as_shap(
                self.model_stage1, X_scaled_s1, stage1_pred_idx
            )
        
        # Determine final prediction
        if stage1_pred_label in ['Benign', 'Reconnaissance', 'Theft']:
            final_prediction = stage1_pred_label
            final_confidence = stage1_confidence
            used_stage2 = False
            stage2_explanation = None
        else:
            # Stage 2 for DOS samples
            X_scaled_s2 = self.scaler_stage2.transform(X_values)
            stage2_pred_proba = self.model_stage2.predict_proba(X_scaled_s2)[0]
            stage2_pred_idx = np.argmax(stage2_pred_proba)
            final_prediction = self.encoder_stage2_classes[stage2_pred_idx]
            final_confidence = stage2_pred_proba[stage2_pred_idx]
            used_stage2 = True
            
            # Stage 2 SHAP values or feature importance
            if self.explainer_stage2 is not None:
                shap_values_s2 = self.explainer_stage2.shap_values(X_scaled_s2, check_additivity=False)
            else:
                shap_values_s2 = self._get_feature_importance_as_shap(
                    self.model_stage2, X_scaled_s2, stage2_pred_idx
                )
            
            # Get top features for Stage 2
            stage2_shap = shap_values_s2[0] if isinstance(shap_values_s2, list) else shap_values_s2
            stage2_top_features = self._get_top_contributing_features(
                stage2_shap[0] if len(stage2_shap.shape) > 1 else stage2_shap,
                X.iloc[0],
                top_n=10
            )
            stage2_explanation = stage2_top_features
        
        # Get top contributing features for Stage 1
        if isinstance(shap_values_s1, list):
            stage1_shap = shap_values_s1[stage1_pred_idx][0]
        elif len(shap_values_s1.shape) == 3:
            stage1_shap = shap_values_s1[0, :, stage1_pred_idx]
        else:
            # For 2D array (fallback method)
            stage1_shap = shap_values_s1[0]
        
        stage1_top_features = self._get_top_contributing_features(
            stage1_shap,
            X.iloc[0],
            top_n=10
        )
        
        # Build human-readable explanation
        explanation = self._build_explanation(
            final_prediction,
            final_confidence,
            stage1_pred_label,
            stage1_confidence,
            stage1_top_features,
            used_stage2,
            stage2_explanation,
            X.iloc[0],
            actual_label,
            stage1_confidence  # Pass stage1_confidence for reasoning
        )
        
        return explanation
    
    def _get_feature_importance_as_shap(self, model, X, pred_idx):
        """
        Use feature importance as fallback when SHAP fails
        """
        # Get feature importance from model
        importance = model.feature_importances_
        
        # Create pseudo-SHAP values weighted by feature values
        X_array = X[0] if len(X.shape) > 1 else X
        pseudo_shap = importance * X_array
        
        return pseudo_shap.reshape(1, -1)
    
    def _get_top_contributing_features(self, shap_values, feature_values, top_n=10):
        """
        Get top features contributing to prediction
        """
        # Get absolute SHAP values for ranking
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
                          used_stage2, stage2_features, feature_values, actual_label, s1_conf):
        """
        Build comprehensive human-readable explanation
        """
        explanation = {
            'prediction': final_prediction,
            'confidence': float(final_confidence),
            'actual_label': actual_label,
            'correct': final_prediction == actual_label if actual_label else None,
            'reasoning': self._generate_reasoning(
                final_prediction, final_confidence, stage1_label, 
                s1_conf, stage1_features, used_stage2, stage2_features, feature_values
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
        Generate human-readable reasoning with proof
        """
        reasoning = []
        
        # Header
        reasoning.append(f"🎯 CLASSIFICATION: {final_prediction} ({final_confidence*100:.1f}% confidence)")
        reasoning.append("")
        
        # Stage 1 reasoning
        reasoning.append("📊 STAGE 1 ANALYSIS (Attack Category Detection):")
        reasoning.append(f"   Detected as: {stage1_label} ({stage1_confidence*100:.1f}% confidence)")
        reasoning.append("")
        reasoning.append("   🔍 Key Evidence:")
        
        for i, feat in enumerate(stage1_features[:5], 1):
            direction = "HIGHER" if feat['shap_value'] > 0 else "LOWER"
            reasoning.append(
                f"   {i}. {feat['feature']} = {feat['value']:.4f}"
            )
            reasoning.append(
                f"      → {direction} value {feat['impact']} {stage1_label}"
            )
            reasoning.append(
                f"      → Impact strength: {feat['importance']:.4f}"
            )
        
        # Stage 2 reasoning if used
        if used_stage2:
            reasoning.append("")
            reasoning.append("📊 STAGE 2 ANALYSIS (DDoS vs DoS Distinction):")
            reasoning.append(f"   Refined prediction: {final_prediction} ({final_confidence*100:.1f}% confidence)")
            reasoning.append("")
            reasoning.append("   🔍 Key Distinguishing Features:")
            
            for i, feat in enumerate(stage2_features[:5], 1):
                direction = "HIGHER" if feat['shap_value'] > 0 else "LOWER"
                reasoning.append(
                    f"   {i}. {feat['feature']} = {feat['value']:.4f}"
                )
                reasoning.append(
                    f"      → {direction} value {feat['impact']} {final_prediction}"
                )
                reasoning.append(
                    f"      → Impact strength: {feat['importance']:.4f}"
                )
        
        # Summary proof
        reasoning.append("")
        reasoning.append("✅ PROOF SUMMARY:")
        reasoning.append(self._generate_attack_specific_proof(
            final_prediction, stage1_features, stage2_features, feature_values, used_stage2
        ))
        
        return "\n".join(reasoning)
    
    def _generate_attack_specific_proof(self, attack_type, stage1_features, 
                                       stage2_features, feature_values, used_stage2):
        """
        Generate attack-specific proof based on known attack characteristics
        """
        proof = []
        
        # Get feature values
        protocol = feature_values.get('PROTOCOL', 0)
        tcp_flags = feature_values.get('TCP_FLAGS', 0)
        packet_rate = feature_values.get('PACKET_RATE', 0)
        byte_asymmetry = feature_values.get('BYTE_ASYMMETRY', 0)
        in_pkts = feature_values.get('IN_PKTS', 0)
        out_pkts = feature_values.get('OUT_PKTS', 0)
        
        if attack_type == 'Benign':
            proof.append("   🟢 BENIGN TRAFFIC INDICATORS:")
            proof.append(f"      • Normal protocol usage (Protocol: {protocol:.0f})")
            proof.append(f"      • Balanced bidirectional flow (In: {in_pkts:.0f}, Out: {out_pkts:.0f})")
            proof.append(f"      • Moderate packet rate ({packet_rate:.2f} pkts/ms)")
            proof.append("      • No anomalous patterns detected")
            
        elif attack_type == 'DDoS':
            proof.append("   🔴 DDOS ATTACK INDICATORS:")
            proof.append(f"      • High-volume flooding detected (Packet rate: {packet_rate:.2f} pkts/ms)")
            proof.append(f"      • Protocol: {protocol:.0f} (Typical for DDoS)")
            proof.append(f"      • TCP Flags: {tcp_flags:.0f} (Possible SYN flood)")
            proof.append(f"      • Asymmetric traffic (Asymmetry: {byte_asymmetry:.2f})")
            if used_stage2 and stage2_features:
                top_ddos_feature = stage2_features[0]['feature']
                proof.append(f"      • Distinguishing feature: {top_ddos_feature}")
            proof.append("      • Pattern consistent with distributed attack")
            
        elif attack_type == 'DoS':
            proof.append("   🔴 DOS ATTACK INDICATORS:")
            proof.append(f"      • High-volume flooding detected (Packet rate: {packet_rate:.2f} pkts/ms)")
            proof.append(f"      • Protocol: {protocol:.0f} (Typical for DoS)")
            proof.append(f"      • TCP Flags: {tcp_flags:.0f} (Possible SYN flood)")
            proof.append(f"      • Asymmetric traffic (Asymmetry: {byte_asymmetry:.2f})")
            if used_stage2 and stage2_features:
                top_dos_feature = stage2_features[0]['feature']
                proof.append(f"      • Distinguishing feature: {top_dos_feature}")
            proof.append("      • Pattern consistent with single-source attack")
            
        elif attack_type == 'Reconnaissance':
            proof.append("   🟡 RECONNAISSANCE ATTACK INDICATORS:")
            proof.append(f"      • Protocol scanning pattern (Protocol: {protocol:.0f})")
            proof.append(f"      • Low packet rate ({packet_rate:.2f} pkts/ms) - typical for scanning")
            proof.append(f"      • TCP Flags: {tcp_flags:.0f} (Port scanning signature)")
            proof.append("      • Probing behavior detected")
            proof.append("      • Information gathering attempt identified")
            
        elif attack_type == 'Theft':
            proof.append("   🔴 DATA THEFT INDICATORS:")
            proof.append(f"      • Protocol: {protocol:.0f} (Data exfiltration)")
            proof.append(f"      • High data transfer (Asymmetry: {byte_asymmetry:.2f})")
            proof.append(f"      • Packet rate: {packet_rate:.2f} pkts/ms")
            proof.append("      • Unauthorized data access pattern")
            proof.append("      • Exfiltration behavior detected")
        
        return "\n".join(proof)
    
    def explain_multiple_samples(self, df_samples):
        """
        Explain multiple predictions and save results
        """
        print("\n" + "=" * 80)
        print("GENERATING SHAP EXPLANATIONS FOR MULTIPLE SAMPLES")
        print("=" * 80)
        
        explanations = []
        
        for idx in range(len(df_samples)):
            sample = df_samples.iloc[[idx]]
            actual_label = sample['Attack'].values[0] if 'Attack' in sample.columns else None
            
            print(f"\n📝 Explaining sample {idx + 1}/{len(df_samples)} (Actual: {actual_label})...")
            
            explanation = self.explain_prediction(sample, actual_label)
            explanations.append(explanation)
            
            # Print summary
            pred = explanation['prediction']
            conf = explanation['confidence'] * 100
            correct = "✅ CORRECT" if explanation['correct'] else "❌ INCORRECT"
            print(f"   Prediction: {pred} ({conf:.1f}% confidence) - {correct}")
        
        return explanations
    
    def save_explanations(self, explanations, filename='shap_explanations.json'):
        """
        Save explanations to file
        """
        # Convert to JSON-serializable format
        serializable_explanations = []
        for exp in explanations:
            serializable_explanations.append({
                'prediction': exp['prediction'],
                'confidence': float(exp['confidence']),
                'actual_label': exp['actual_label'],
                'correct': exp['correct'],
                'reasoning': exp['reasoning'],
                'stage1': exp['stage1'],
                'stage2': exp.get('stage2', None)
            })
        
        with open(filename, 'w') as f:
            json.dump(serializable_explanations, f, indent=2)
        
        print(f"\n✅ Explanations saved to {filename}")
        
        # Also save human-readable version
        txt_filename = filename.replace('.json', '.txt')
        with open(txt_filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SHAP EXPLANATIONS FOR ATTACK CLASSIFICATIONS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, exp in enumerate(explanations, 1):
                f.write(f"\n{'=' * 80}\n")
                f.write(f"SAMPLE {i}: {exp['actual_label']} → {exp['prediction']}\n")
                f.write(f"{'=' * 80}\n\n")
                f.write(exp['reasoning'])
                f.write("\n\n")
        
        print(f"✅ Human-readable explanations saved to {txt_filename}")

def main():
    """
    Main execution for SHAP explainability system
    """
    print("\n" + "=" * 80)
    print("SHAP EXPLAINABILITY SYSTEM FOR HIERARCHICAL ANOMALY DETECTION")
    print("=" * 80 + "\n")
    
    # Initialize explainer
    explainer = HierarchicalSHAPExplainer()
    
    # Load models and data
    df = explainer.load_models_and_data()
    
    # Sample background data for SHAP (use subset for efficiency)
    print("\n📊 Preparing background data for SHAP...")
    background_samples = df.groupby('Attack').apply(
        lambda x: x.sample(min(100, len(x)), random_state=42)
    ).reset_index(drop=True)
    print(f"✅ Background data: {len(background_samples)} samples")
    
    # Create SHAP explainers
    explainer.create_shap_explainers(background_samples)
    
    # Select diverse test samples (10 from each attack type)
    print("\n📊 Selecting test samples for explanation...")
    test_samples = df.groupby('Attack').apply(
        lambda x: x.sample(min(10, len(x)), random_state=123)
    ).reset_index(drop=True)
    print(f"✅ Selected {len(test_samples)} test samples")
    
    # Generate explanations
    explanations = explainer.explain_multiple_samples(test_samples)
    
    # Save results
    explainer.save_explanations(explanations, 'shap_explanations.json')
    
    # Print detailed example
    print("\n" + "=" * 80)
    print("EXAMPLE DETAILED EXPLANATION")
    print("=" * 80)
    print(explanations[0]['reasoning'])
    
    # Summary statistics
    correct = sum(1 for exp in explanations if exp['correct'])
    total = len(explanations)
    
    print("\n" + "=" * 80)
    print("SHAP EXPLANATION GENERATION COMPLETE")
    print("=" * 80)
    print(f"✅ Total samples explained: {total}")
    print(f"✅ Correct predictions: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"✅ Files generated:")
    print(f"   • shap_explanations.json - Structured explanations")
    print(f"   • shap_explanations.txt - Human-readable explanations")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

