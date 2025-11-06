"""
Interpreter Agent - Ensemble Combiner for XGBoost and LSTM Predictions

This agent combines SHAP explanations from both XGBoost and LSTM models
using ensemble voting logic and prepares comprehensive evidence for LLM analysis.
"""

import pandas as pd
import numpy as np
import json
from shap_explainer import HierarchicalSHAPExplainer as XGBoostExplainer
from lstm_shap_explainer import LSTMHierarchicalSHAPExplainer as LSTMExplainer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class InterpreterAgent:
    """
    Ensemble agent combining XGBoost and LSTM predictions with SHAP explanations
    """
    
    def __init__(self):
        self.xgboost_explainer = None
        self.lstm_explainer = None
        
    def initialize_agents(self):
        """
        Initialize both XGBoost and LSTM SHAP explainers
        """
        print("=" * 80)
        print("INTERPRETER AGENT - INITIALIZING ENSEMBLE SYSTEM")
        print("=" * 80)
        
        # Initialize XGBoost Agent
        print("\n🤖 AGENT 1: XGBoost + SHAP")
        print("-" * 80)
        self.xgboost_explainer = XGBoostExplainer()
        df_xgb = self.xgboost_explainer.load_models_and_data()
        
        # Prepare background for XGBoost
        background_xgb = df_xgb.groupby('Attack').apply(
            lambda x: x.sample(min(100, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        self.xgboost_explainer.create_shap_explainers(background_xgb)
        print("✅ XGBoost Agent ready")
        
        # Initialize LSTM Agent
        print("\n🤖 AGENT 2: LSTM + SHAP")
        print("-" * 80)
        self.lstm_explainer = LSTMExplainer()
        df_lstm = self.lstm_explainer.load_models_and_data()
        
        # Prepare background for LSTM
        background_lstm = df_lstm.groupby('Attack').apply(
            lambda x: x.sample(min(100, len(x)), random_state=42)
        ).reset_index(drop=True)
        
        self.lstm_explainer.create_shap_explainers(background_lstm)
        print("✅ LSTM Agent ready")
        
        print("\n" + "=" * 80)
        print("✅ ENSEMBLE SYSTEM INITIALIZED")
        print("=" * 80)
        
        return df_xgb  # Return for test samples
    
    def ensemble_predict(self, sample_data, actual_label=None, timestamp=None, 
                        source_ip="Unknown", affected_assets="System"):
        """
        Get predictions from both agents and create ensemble decision
        """
        print("\n" + "=" * 80)
        print("ENSEMBLE PREDICTION & INTERPRETATION")
        print("=" * 80)
        
        # Get timestamp
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get XGBoost prediction
        print("\n🤖 Agent 1 (XGBoost) analyzing...")
        xgb_explanation = self.xgboost_explainer.explain_prediction(sample_data, actual_label)
        print(f"   XGBoost: {xgb_explanation['prediction']} ({xgb_explanation['confidence']*100:.1f}%)")
        
        # Get LSTM prediction
        print("\n🤖 Agent 2 (LSTM) analyzing...")
        lstm_explanation = self.lstm_explainer.explain_prediction(sample_data, actual_label)
        print(f"   LSTM: {lstm_explanation['prediction']} ({lstm_explanation['confidence']*100:.1f}%)")
        
        # Ensemble decision
        ensemble_result = self._combine_predictions(
            xgb_explanation, 
            lstm_explanation,
            sample_data,
            actual_label,
            timestamp,
            source_ip,
            affected_assets
        )
        
        return ensemble_result
    
    def _combine_predictions(self, xgb_exp, lstm_exp, sample_data, 
                            actual_label, timestamp, source_ip, affected_assets):
        """
        Combine predictions using ensemble voting logic
        """
        print("\n🔮 ENSEMBLE VOTING:")
        print("-" * 80)
        
        xgb_pred = xgb_exp['prediction']
        xgb_conf = xgb_exp['confidence']
        lstm_pred = lstm_exp['prediction']
        lstm_conf = lstm_exp['confidence']
        
        # Check agreement
        if xgb_pred == lstm_pred:
            # Both agree
            final_prediction = xgb_pred
            final_confidence = (xgb_conf + lstm_conf) / 2
            agreement = "FULL_AGREEMENT"
            confidence_level = "HIGH" if final_confidence > 0.8 else "MEDIUM"
            
            print(f"✅ FULL AGREEMENT: {final_prediction}")
            print(f"   Combined Confidence: {final_confidence*100:.1f}%")
            
        else:
            # Disagreement - use confidence-weighted voting
            print(f"⚠️  DISAGREEMENT DETECTED:")
            print(f"   XGBoost: {xgb_pred} ({xgb_conf*100:.1f}%)")
            print(f"   LSTM:    {lstm_pred} ({lstm_conf*100:.1f}%)")
            
            if xgb_conf > lstm_conf:
                final_prediction = xgb_pred
                final_confidence = xgb_conf
                winning_model = "XGBoost"
            else:
                final_prediction = lstm_pred
                final_confidence = lstm_conf
                winning_model = "LSTM"
            
            agreement = "DISAGREEMENT"
            confidence_level = "LOW"
            
            print(f"   Winner: {winning_model} ({final_confidence*100:.1f}%)")
        
        # Extract feature values
        feature_values = sample_data.drop(columns=['Attack']).iloc[0] if 'Attack' in sample_data.columns else sample_data.iloc[0]
        
        # Build comprehensive result
        result = {
            'timestamp': timestamp,
            'ensemble_decision': {
                'final_prediction': final_prediction,
                'confidence': float(final_confidence),
                'agreement': agreement,
                'confidence_level': confidence_level
            },
            'actual_label': actual_label,
            'correct': final_prediction == actual_label if actual_label else None,
            'agent_predictions': {
                'xgboost': {
                    'prediction': xgb_pred,
                    'confidence': float(xgb_conf),
                    'model_type': 'XGBoost'
                },
                'lstm': {
                    'prediction': lstm_pred,
                    'confidence': float(lstm_conf),
                    'model_type': 'LSTM'
                }
            },
            'metadata': {
                'source_ip': source_ip,
                'affected_assets': affected_assets,
                'protocol': float(feature_values.get('PROTOCOL', 0)),
                'dest_port': float(feature_values.get('L4_DST_PORT', 0)),
                'packet_rate': float(feature_values.get('PACKET_RATE', 0)),
                'byte_rate': float(feature_values.get('BYTE_RATE', 0)),
                'flow_duration': float(feature_values.get('FLOW_DURATION_MILLISECONDS', 0)),
                'tcp_flags': float(feature_values.get('TCP_FLAGS', 0))
            },
            'evidence': self._build_evidence(
                xgb_exp, lstm_exp, feature_values, agreement
            ),
            'shap_explanations': {
                'xgboost': xgb_exp,
                'lstm': lstm_exp
            }
        }
        
        return result
    
    def _build_evidence(self, xgb_exp, lstm_exp, feature_values, agreement):
        """
        Build evidence for LLM consumption
        """
        evidence = {
            'agreement_status': agreement,
            'key_indicators': self._extract_key_indicators(feature_values),
            'xgboost_evidence': self._extract_model_evidence(xgb_exp, 'XGBoost'),
            'lstm_evidence': self._extract_model_evidence(lstm_exp, 'LSTM'),
            'combined_top_features': self._merge_top_features(xgb_exp, lstm_exp)
        }
        
        return evidence
    
    def _extract_key_indicators(self, feature_values):
        """
        Extract human-readable key indicators
        """
        packet_rate = feature_values.get('PACKET_RATE', 0)
        byte_rate = feature_values.get('BYTE_RATE', 0)
        flow_duration = feature_values.get('FLOW_DURATION_MILLISECONDS', 0)
        protocol = feature_values.get('PROTOCOL', 0)
        tcp_flags = feature_values.get('TCP_FLAGS', 0)
        dest_port = feature_values.get('L4_DST_PORT', 0)
        
        # Interpret protocol
        protocol_name = {6: 'TCP', 17: 'UDP', 1: 'ICMP'}.get(int(protocol), f'Protocol {int(protocol)}')
        
        # Traffic intensity
        if packet_rate > 10:
            traffic_intensity = "HIGH (Possible attack)"
        elif packet_rate > 1:
            traffic_intensity = "MODERATE"
        else:
            traffic_intensity = "LOW (Normal)"
        
        return {
            'traffic_volume': f"{packet_rate:.2f} packets/ms",
            'data_rate': f"{byte_rate:.2f} bytes/ms",
            'duration': f"{flow_duration:.0f} ms",
            'protocol': protocol_name,
            'tcp_flags': int(tcp_flags),
            'destination_port': int(dest_port),
            'traffic_intensity': traffic_intensity
        }
    
    def _extract_model_evidence(self, explanation, model_name):
        """
        Extract model-specific evidence
        """
        top_features = explanation['stage1']['top_features'][:5]
        
        features_summary = []
        for feat in top_features:
            features_summary.append({
                'feature': feat['feature'],
                'value': float(feat['value']),
                'shap_impact': float(feat['shap_value']),
                'direction': feat['impact']
            })
        
        return {
            'model': model_name,
            'prediction': explanation['prediction'],
            'confidence': float(explanation['confidence']),
            'top_5_features': features_summary,
            'used_two_stage': 'stage2' in explanation
        }
    
    def _merge_top_features(self, xgb_exp, lstm_exp):
        """
        Merge top features from both V1 models (kept for backwards compatibility)
        """
        xgb_features = {f['feature']: f for f in xgb_exp['stage1']['top_features'][:10]}
        lstm_features = {f['feature']: f for f in lstm_exp['stage1']['top_features'][:10]}
        
        # Find common important features
        common_features = set(xgb_features.keys()) & set(lstm_features.keys())
        
        merged = []
        for feat_name in common_features:
            xgb_feat = xgb_features[feat_name]
            lstm_feat = lstm_features[feat_name]
            
            merged.append({
                'feature': feat_name,
                'value': float(xgb_feat['value']),
                'xgboost_shap': float(xgb_feat['shap_value']),
                'lstm_shap': float(lstm_feat['shap_value']),
                'both_agree': (xgb_feat['shap_value'] > 0) == (lstm_feat['shap_value'] > 0),
                'average_importance': float((xgb_feat['importance'] + lstm_feat['importance']) / 2)
            })
        
        # Sort by average importance
        merged.sort(key=lambda x: x['average_importance'], reverse=True)
        
        return merged[:10]
    
    def _extract_v2_model_evidence(self, explanation, model_name):
        """
        Extract V2 model-specific evidence
        """
        top_features = explanation['top_5_features']
        
        return {
            'model': model_name,
            'prediction': explanation['prediction'],
            'confidence': float(explanation['confidence']),
            'top_5_features': top_features
        }
    
    def _merge_all_features(self, v1_xgb_exp, v1_lstm_exp, v2_xgb_exp, v2_lstm_exp):
        """
        Merge top features from all 4 models
        """
        # Get top features from all models
        v1_xgb_features = {f['feature']: f for f in v1_xgb_exp['stage1']['top_features'][:5]}
        v1_lstm_features = {f['feature']: f for f in v1_lstm_exp['stage1']['top_features'][:5]}
        v2_xgb_features = {f['feature']: f for f in v2_xgb_exp['top_5_features']}
        v2_lstm_features = {f['feature']: f for f in v2_lstm_exp['top_5_features']}
        
        # Find common important features
        all_features = set(v1_xgb_features.keys()) | set(v1_lstm_features.keys()) | \
                       set(v2_xgb_features.keys()) | set(v2_lstm_features.keys())
        
        merged = []
        for feat_name in all_features:
            # Collect SHAP values from all models (if available)
            shap_values = []
            
            if feat_name in v1_xgb_features:
                shap_values.append(v1_xgb_features[feat_name]['shap_value'])
            if feat_name in v1_lstm_features:
                shap_values.append(v1_lstm_features[feat_name]['shap_value'])
            if feat_name in v2_xgb_features:
                shap_values.append(v2_xgb_features[feat_name]['shap_impact'])
            if feat_name in v2_lstm_features:
                shap_values.append(v2_lstm_features[feat_name]['shap_impact'])
            
            if shap_values:
                avg_shap = sum(abs(s) for s in shap_values) / len(shap_values)
                value = v1_xgb_features.get(feat_name, v2_xgb_features.get(feat_name, {'value': 0})).get('value', 0)
                
                merged.append({
                    'feature': feat_name,
                    'value': float(value),
                    'average_shap_importance': float(avg_shap),
                    'num_models_agree': len(shap_values)
                })
        
        # Sort by average importance
        merged.sort(key=lambda x: (x['num_models_agree'], x['average_shap_importance']), reverse=True)
        
        return merged[:10]
    
    def process_multiple_samples(self, df_samples):
        """
        Process multiple samples through ensemble
        """
        print("\n" + "=" * 80)
        print("PROCESSING MULTIPLE SAMPLES THROUGH ENSEMBLE")
        print("=" * 80)
        
        results = []
        
        for idx in range(len(df_samples)):
            sample = df_samples.iloc[[idx]]
            actual_label = sample['Attack'].values[0] if 'Attack' in sample.columns else None
            
            print(f"\n{'=' * 80}")
            print(f"SAMPLE {idx + 1}/{len(df_samples)} - Actual: {actual_label}")
            print(f"{'=' * 80}")
            
            result = self.ensemble_predict(
                sample, 
                actual_label=actual_label,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                source_ip=f"192.168.1.{idx+10}",
                affected_assets=f"Server-{idx+1}"
            )
            
            results.append(result)
            
            # Summary
            pred = result['ensemble_decision']['final_prediction']
            conf = result['ensemble_decision']['confidence'] * 100
            agreement = result['ensemble_decision']['agreement']
            correct = "✅ CORRECT" if result['correct'] else "❌ INCORRECT"
            
            print(f"\n🎯 ENSEMBLE DECISION: {pred} ({conf:.1f}%) - {agreement}")
            print(f"   {correct}")
        
        return results
    
    def save_ensemble_results(self, results, filename='ensemble_results.json'):
        """
        Save ensemble results
        """
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Ensemble results saved to {filename}")
        
        # Statistics
        total = len(results)
        correct = sum(1 for r in results if r['correct'])
        agreements = sum(1 for r in results if r['ensemble_decision']['agreement'] == 'FULL_AGREEMENT')
        
        print("\n" + "=" * 80)
        print("ENSEMBLE STATISTICS")
        print("=" * 80)
        print(f"Total samples:     {total}")
        print(f"Correct:           {correct}/{total} ({correct/total*100:.1f}%)")
        print(f"Full agreements:   {agreements}/{total} ({agreements/total*100:.1f}%)")
        print(f"Disagreements:     {total-agreements}/{total} ({(total-agreements)/total*100:.1f}%)")
        print("=" * 80)

def main():
    """
    Main execution
    """
    print("\n" + "=" * 80)
    print("INTERPRETER AGENT - MULTI-AGENT ENSEMBLE SYSTEM")
    print("=" * 80 + "\n")
    
    # Initialize
    agent = InterpreterAgent()
    df = agent.initialize_agents()
    
    # Test samples
    print("\n📊 Selecting test samples...")
    test_samples = df.groupby('Attack').apply(
        lambda x: x.sample(min(3, len(x)), random_state=456)
    ).reset_index(drop=True)
    print(f"✅ Selected {len(test_samples)} test samples")
    
    # Process
    results = agent.process_multiple_samples(test_samples)
    
    # Save
    agent.save_ensemble_results(results)
    
    print("\n" + "=" * 80)
    print("✅ INTERPRETER AGENT COMPLETE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

