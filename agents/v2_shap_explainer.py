"""
SHAP Explainability System for V2 Hierarchical XGBoost Models

Provides interpretable explanations for V2 dataset predictions.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

class V2HierarchicalSHAPExplainer:
    """
    SHAP-based explainer for V2 hierarchical XGBoost models
    """
    
    def __init__(self, model_dir='../models/v2_hierarchical/'):
        self.model_dir = model_dir
        self.model_stage1 = None
        self.model_stage2 = None
        self.scaler_stage1 = None
        self.scaler_stage2 = None
        self.encoder_stage1 = None
        self.encoder_stage2 = None
        self.feature_names = None
        self.explainer_stage1 = None
        self.explainer_stage2 = None
        
    def load_models(self):
        """
        Load V2 hierarchical models and artifacts
        """
        print(f"📂 Loading V2 XGBoost models from {self.model_dir}")
        
        # Load Stage 1 model
        self.model_stage1 = joblib.load(f'{self.model_dir}/xgboost_stage1.pkl')
        self.scaler_stage1 = joblib.load(f'{self.model_dir}/scaler_stage1.pkl')
        self.encoder_stage1 = joblib.load(f'{self.model_dir}/label_encoder_stage1.pkl')
        
        # Load Stage 2 model
        self.model_stage2 = joblib.load(f'{self.model_dir}/xgboost_stage2.pkl')
        self.scaler_stage2 = joblib.load(f'{self.model_dir}/scaler_stage2.pkl')
        self.encoder_stage2 = joblib.load(f'{self.model_dir}/label_encoder_stage2.pkl')
        
        print("✅ V2 XGBoost models loaded")
        print(f"   Stage 1 classes: {list(self.encoder_stage1.classes_)}")
        print(f"   Stage 2 classes: {list(self.encoder_stage2.classes_)}")
        
    def _engineer_features(self, df):
        """
        Apply feature engineering (same as training)
        """
        # Remove columns not used in features
        feature_df = df.drop(columns=['Attack'], errors='ignore')
        
        # Engineer features
        feature_df['PACKET_RATE'] = (feature_df['IN_PKTS'] + feature_df['OUT_PKTS']) / (feature_df['FLOW_DURATION_MILLISECONDS'] + 1)
        feature_df['BYTE_RATE'] = (feature_df['IN_BYTES'] + feature_df['OUT_BYTES']) / (feature_df['FLOW_DURATION_MILLISECONDS'] + 1)
        feature_df['AVG_PACKET_SIZE'] = (feature_df['IN_BYTES'] + feature_df['OUT_BYTES']) / (feature_df['IN_PKTS'] + feature_df['OUT_PKTS'] + 1)
        feature_df['AVG_IN_PACKET_SIZE'] = feature_df['IN_BYTES'] / (feature_df['IN_PKTS'] + 1)
        feature_df['AVG_OUT_PACKET_SIZE'] = feature_df['OUT_BYTES'] / (feature_df['OUT_PKTS'] + 1)
        feature_df['BYTE_ASYMMETRY'] = abs(feature_df['IN_BYTES'] - feature_df['OUT_BYTES']) / (feature_df['IN_BYTES'] + feature_df['OUT_BYTES'] + 1)
        feature_df['PACKET_ASYMMETRY'] = abs(feature_df['IN_PKTS'] - feature_df['OUT_PKTS']) / (feature_df['IN_PKTS'] + feature_df['OUT_PKTS'] + 1)
        feature_df['IN_OUT_BYTE_RATIO'] = feature_df['IN_BYTES'] / (feature_df['OUT_BYTES'] + 1)
        feature_df['IN_OUT_PACKET_RATIO'] = feature_df['IN_PKTS'] / (feature_df['OUT_PKTS'] + 1)
        feature_df['PROTOCOL_INTENSITY'] = feature_df['PROTOCOL'] * feature_df['PACKET_RATE']
        feature_df['TCP_PACKET_INTERACTION'] = feature_df['TCP_FLAGS'] * feature_df['IN_PKTS']
        feature_df['PROTOCOL_PORT_COMBO'] = feature_df['PROTOCOL'] * feature_df['L4_DST_PORT']
        feature_df['FLOW_INTENSITY'] = (feature_df['IN_PKTS'] + feature_df['OUT_PKTS']) / (feature_df['FLOW_DURATION_MILLISECONDS'] + 1) * feature_df['AVG_PACKET_SIZE']
        
        feature_df = feature_df.replace([np.inf, -np.inf], 0).fillna(0)
        
        if self.feature_names is None:
            self.feature_names = list(feature_df.columns)
        
        return feature_df
    
    def create_shap_explainers(self, background_data):
        """
        Create SHAP explainers using background data
        """
        print("\n🔍 Creating V2 XGBoost SHAP explainers...")
        
        # Prepare background data
        X_bg = self._engineer_features(background_data)
        X_bg_scaled = self.scaler_stage1.transform(X_bg)
        
        # Create Stage 1 explainer
        self.explainer_stage1 = shap.TreeExplainer(self.model_stage1, X_bg_scaled)
        print("✅ Stage 1 SHAP explainer created")
        
        # Create Stage 2 explainer
        X_bg_scaled_s2 = self.scaler_stage2.transform(X_bg)
        self.explainer_stage2 = shap.TreeExplainer(self.model_stage2, X_bg_scaled_s2)
        print("✅ Stage 2 SHAP explainer created")
        
    def explain_prediction(self, input_df):
        """
        Generate SHAP explanation for a single prediction
        """
        # Engineer features
        X = self._engineer_features(input_df)
        X_scaled_s1 = self.scaler_stage1.transform(X)
        
        # Stage 1 prediction
        pred_s1 = self.model_stage1.predict(X_scaled_s1)[0]
        pred_s1_proba = self.model_stage1.predict_proba(X_scaled_s1)[0]
        pred_s1_label = self.encoder_stage1.classes_[pred_s1]
        
        # Stage 1 SHAP values
        shap_values_s1 = self.explainer_stage1.shap_values(X_scaled_s1)
        
        # Handle multi-class SHAP values
        if isinstance(shap_values_s1, list):
            shap_values_for_pred = shap_values_s1[pred_s1][0]
        else:
            shap_values_for_pred = shap_values_s1[0]
        
        # Get top features for Stage 1
        feature_importance = []
        for i, (feat, shap_val) in enumerate(zip(self.feature_names, shap_values_for_pred)):
            feature_importance.append({
                'feature': feat,
                'shap_impact': float(shap_val),
                'value': float(X.iloc[0][feat])
            })
        
        feature_importance = sorted(feature_importance, key=lambda x: abs(x['shap_impact']), reverse=True)
        
        result = {
            'prediction': pred_s1_label,
            'confidence': float(pred_s1_proba[pred_s1]),
            'all_probabilities': {cls: float(prob) for cls, prob in zip(self.encoder_stage1.classes_, pred_s1_proba)},
            'top_5_features': feature_importance[:5],
            'all_features': feature_importance
        }
        
        # Stage 2 if DOS category
        if pred_s1_label == 'DOS':
            X_scaled_s2 = self.scaler_stage2.transform(X)
            pred_s2 = self.model_stage2.predict(X_scaled_s2)[0]
            pred_s2_proba = self.model_stage2.predict_proba(X_scaled_s2)[0]
            pred_s2_label = self.encoder_stage2.classes_[pred_s2]
            
            # Stage 2 SHAP values
            shap_values_s2 = self.explainer_stage2.shap_values(X_scaled_s2)
            
            if isinstance(shap_values_s2, list):
                shap_values_for_pred_s2 = shap_values_s2[pred_s2][0]
            else:
                shap_values_for_pred_s2 = shap_values_s2[0]
            
            feature_importance_s2 = []
            for i, (feat, shap_val) in enumerate(zip(self.feature_names, shap_values_for_pred_s2)):
                feature_importance_s2.append({
                    'feature': feat,
                    'shap_impact': float(shap_val),
                    'value': float(X.iloc[0][feat])
                })
            
            feature_importance_s2 = sorted(feature_importance_s2, key=lambda x: abs(x['shap_impact']), reverse=True)
            
            result['stage2_prediction'] = pred_s2_label
            result['stage2_confidence'] = float(pred_s2_proba[pred_s2])
            result['stage2_probabilities'] = {cls: float(prob) for cls, prob in zip(self.encoder_stage2.classes_, pred_s2_proba)}
            result['stage2_top_features'] = feature_importance_s2[:5]
            
            # Final prediction is from Stage 2
            result['prediction'] = pred_s2_label
            result['confidence'] = float(pred_s2_proba[pred_s2])
        
        return result

def main():
    """
    Main execution
    """
    print("\n" + "=" * 80)
    print("V2 XGBOOST SHAP EXPLAINER")
    print("=" * 80)
    print("\n⚠️  This module is used by the interpreter agent.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

