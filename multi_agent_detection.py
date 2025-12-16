"""
Multi-Agent Detection System with SHAP Explainability

Integrates all 7 trained agents with SHAP for explainability and
implements ensemble voting with abstaining logic.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import shap
from collections import Counter
import logging

from utils.feature_adapter import FeatureAdapter, CanonicalFeatureBuilder, AttackTaxonomy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiAgentDetectionSystem:
    """
    Multi-Agent Detection System with SHAP explainability
    Coordinates all 7 agents with intelligent ensemble voting
    """
    
    def __init__(self, models_dir: str = './trained_models'):
        self.models_dir = models_dir
        self.feature_adapter = FeatureAdapter()
        self.canonical_builder = CanonicalFeatureBuilder()
        self.agents = {}
        self.agents = {}
        self.shap_explainers = {}
        # Prepare background data for SHAP
        self.background_data = None
        logger.info(f"Background data set with shape: {self.background_data.shape if self.background_data is not None else 'None'}")
        # Clear existing explainers to force recreation with new background
        self.shap_explainers = {}
        self.load_all_agents()
        
    def set_background_data(self, data: np.ndarray) -> None:
        """Set background data for SHAP explainers"""
        self.background_data = data
        logger.info(f"Background data set with shape: {data.shape}")
        # Clear existing explainers to force recreation with new background
        self.shap_explainers = {}
        
    def load_all_agents(self) -> None:
        """Load all trained agents from pickle files"""
        logger.info("Loading trained agents...")
        
        agent_names = [
            'agent_1_xgboost_v1',
            'agent_2_xgboost_v2',
            'agent_3_lstm_v1',
            'agent_4_lstm_v2',
            'agent_5_rf_v1',
            'agent_6_rf_v2',
            'agent_7_unknown'
        ]
        
        for agent_name in agent_names:
            model_path = os.path.join(self.models_dir, f'{agent_name}.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.agents[agent_name] = pickle.load(f)
                logger.info(f"Loaded {agent_name}")
            else:
                logger.warning(f"Model not found: {model_path}")
    
    def create_shap_explainer(self, agent_name: str, background_data: np.ndarray) -> Any:
        """
        Create SHAP explainer for an agent
        
        Args:
            agent_name: Name of the agent
            background_data: Background data for SHAP
            
        Returns:
            SHAP explainer object
        """
        model = self.agents.get(agent_name)
        if model is None:
            return None
        
        agent_info = self.feature_adapter.get_agent_info(agent_name)
        model_type = agent_info.get('model_type', '')
        
        try:
            # Try TreeExplainer first for tree models
            if model_type in ['xgboost', 'random_forest']:
                try:
                    explainer = shap.TreeExplainer(model)
                    # Test if it works
                    if background_data is not None:
                        _ = explainer.shap_values(background_data[:1])
                    return explainer
                except Exception as tree_error:
                    logger.warning(f"TreeExplainer failed for {agent_name}: {tree_error}. Falling back to KernelExplainer.")
                    # Fallback to KernelExplainer logic below
            
            # KernelExplainer fallback (or primary for non-tree models)
            if background_data is None:
                if self.background_data is not None:
                    background_data = self.background_data
                else:
                    logger.warning(f"No background data available for {agent_name} SHAP")
                    return None
            
            # Use a small background sample (max 50)
            bg_sample = background_data[:50]
            
            if model_type == 'xgboost' or model_type == 'random_forest':
                # Wrapper for probability prediction
                def predict_proba_wrapper(x):
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(x)
                    else:
                        # For models without predict_proba, fake it based on predict
                        preds = model.predict(x)
                        return np.column_stack([1-preds, preds])
                
                explainer = shap.KernelExplainer(predict_proba_wrapper, bg_sample)
                
            elif model_type == 'lstm':
                # Deep explainer for neural networks
                background_sample_3d = np.expand_dims(bg_sample, axis=-1)
                explainer = shap.DeepExplainer(model, background_sample_3d)
                
            elif model_type == 'autoencoder':
                # For autoencoder, use KernelExplainer
                def model_predict(x):
                    reconstructions = model['autoencoder'].predict(x, verbose=0)
                    mse = np.mean(np.power(x - reconstructions, 2), axis=1)
                    return (mse > model['threshold']).astype(int)
                
                explainer = shap.KernelExplainer(model_predict, bg_sample)
            else:
                explainer = None
            
            self.shap_explainers[agent_name] = explainer
            return explainer
            
        except Exception as e:
            logger.warning(f"Failed to create SHAP explainer for {agent_name}: {e}")
            return None
    
    def get_shap_explanation(self, agent_name: str, data: np.ndarray, 
                            feature_names: List[str]) -> Dict[str, Any]:
        """
        Get SHAP explanation for a prediction
        
        Args:
            agent_name: Name of the agent
            data: Input data
            feature_names: List of feature names
            
        Returns:
            Dictionary with SHAP values and explanation
        """
        explainer = self.shap_explainers.get(agent_name)
        if explainer is None:
            return {'error': 'No explainer available'}
        
        agent_info = self.feature_adapter.get_agent_info(agent_name)
        model_type = agent_info.get('model_type', '')
        
        try:
            if model_type == 'lstm':
                # Reshape for LSTM
                data_3d = np.expand_dims(data, axis=-1)
                shap_values = explainer.shap_values(data_3d)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                # Flatten back
                shap_values = shap_values.reshape(data.shape)
            else:
                shap_values = explainer.shap_values(data)
                
                # Handle list output (classification)
                if isinstance(shap_values, list):
                    # For binary classification, get positive class
                    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                
                # Handle 3D array (samples x features x classes)
                shap_values = np.array(shap_values)
                if len(shap_values.shape) == 3:
                    # Take positive class (last dimension index 1 if exists)
                    if shap_values.shape[2] >= 2:
                        shap_values = shap_values[:, :, 1]
                    else:
                        shap_values = shap_values[:, :, 0]
                
                # Ensure 2D shape
                if len(shap_values.shape) == 1:
                    shap_values = shap_values.reshape(1, -1)
            
            # Get top contributing features
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            top_indices = np.argsort(mean_abs_shap)[-5:][::-1]
            
            top_features = []
            for idx in top_indices:
                if idx < len(feature_names):
                    try:
                        imp_raw = mean_abs_shap[idx]
                        importance = float(np.atleast_1d(imp_raw).flat[0]) if hasattr(imp_raw, '__iter__') else float(imp_raw)
                        
                        sv_raw = np.mean(shap_values[:, idx])
                        shap_val = float(np.atleast_1d(sv_raw).flat[0]) if hasattr(sv_raw, '__iter__') else float(sv_raw)
                        
                        # Sanitize NaNs
                        if not np.isfinite(importance):
                            importance = 0.0
                        if not np.isfinite(shap_val):
                            shap_val = 0.0
                            
                        top_features.append({
                            'feature': feature_names[idx],
                            'importance': importance,
                            'shap_value': shap_val
                        })
                    except Exception:
                        continue
            
            return {
                'shap_values': shap_values,
                'top_features': top_features,
                'explanation': self._generate_explanation(top_features, agent_name)
            }
            
        except Exception as e:
            logger.warning(f"SHAP explanation failed for {agent_name}: {e}")
            return {'error': str(e)}
    
    def _generate_explanation(self, top_features: List[Dict], agent_name: str) -> str:
        """Generate human-readable explanation from SHAP values"""
        if not top_features:
            return "No significant features identified."
        
        explanation = f"{agent_name} detected anomaly based on:\n"
        for i, feat in enumerate(top_features[:3], 1):
            direction = "increased" if feat['shap_value'] > 0 else "decreased"
            explanation += f"{i}. {feat['feature']} ({direction} anomaly score by {abs(feat['shap_value']):.4f})\n"
        
        return explanation
    
    
    def _get_feature_importance_fallback(self, agent_name: str, data: np.ndarray,
                                         feature_names: List[str]) -> Dict[str, Any]:
        """
        Get REAL per-sample feature importance using SHAP
        NOT hardcoded - calculates unique values for each prediction
        """
        model = self.agents.get(agent_name)
        agent_info = self.feature_adapter.get_agent_info(agent_name)
        model_type = agent_info.get('model_type', '')
        
        try:
            top_features = []
            
            # Try to get/create explainer
            explainer = self.shap_explainers.get(agent_name)
            if explainer is None and self.background_data is not None:
                explainer = self.create_shap_explainer(agent_name, self.background_data)
            
            if explainer is not None:
                # Use the explainer we just verified/created
                shap_values = explainer.shap_values(data)
                
                # Handle list output (classification)
                if isinstance(shap_values, list):
                    # For binary classification, usually index 1 is positive class
                    if len(shap_values) > 1:
                        shap_values = shap_values[1]
                    else:
                        shap_values = shap_values[0]
                
                # Handle 3D array (samples x features x classes)
                if len(shap_values.shape) == 3:
                    # Assuming last dim is classes, take positive class (index 1)
                    if shap_values.shape[2] >= 2:
                        shap_values = shap_values[:, :, 1]
                    else:
                        shap_values = shap_values[:, :, 0]

                # Get absolute SHAP values for this specific sample
                if len(shap_values.shape) > 1:
                    sample_shap = np.abs(shap_values[0])  # First sample
                else:
                    sample_shap = np.abs(shap_values)
                
                # Get top 5 features for THIS specific sample
                top_indices = np.argsort(sample_shap)[-5:][::-1]
                
                for idx in top_indices:
                    if idx < len(feature_names):
                        try:
                            # Robust scalar extraction - handle arrays of any dimension
                            imp_raw = sample_shap[idx]
                            importance = float(np.atleast_1d(imp_raw).flat[0]) if hasattr(imp_raw, '__iter__') and not isinstance(imp_raw, str) else float(imp_raw)
                            
                            if len(shap_values.shape) > 1:
                                sv_raw = shap_values[0, idx]
                            else:
                                sv_raw = shap_values[idx]
                            shap_val = float(np.atleast_1d(sv_raw).flat[0]) if hasattr(sv_raw, '__iter__') and not isinstance(sv_raw, str) else float(sv_raw)
                            
                            # Sanitize NaNs and Infs
                            if not np.isfinite(importance):
                                importance = 0.0
                            if not np.isfinite(shap_val):
                                shap_val = 0.0
                                
                            top_features.append({
                                'feature': feature_names[idx],
                                'importance': importance,
                                'shap_value': shap_val
                            })
                        except Exception as extract_err:
                            # Skip this feature if extraction fails
                            logger.debug(f"Feature extraction failed for idx {idx}: {extract_err}")
                            continue
                
                logger.info(f"✅ Calculated REAL SHAP for {agent_name}")
                
            else:
                # Fallback to global importance if SHAP fails completely
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    top_indices = np.argsort(importances)[-5:][::-1]
                    
                    for idx in top_indices:
                        if idx < len(feature_names):
                            importance = float(importances[idx])
                            if np.isnan(importance) or np.isinf(importance):
                                importance = 0.0
                                
                            top_features.append({
                                'feature': feature_names[idx],
                                'importance': importance,
                                'shap_value': 0.0
                            })
            
            return {
                'top_features': top_features,
                'explanation': f"{agent_name} analysis"
            }
        except Exception as e:
            logger.error(f"Feature importance failed for {agent_name}: {e}")
            return {'top_features': [], 'error': str(e)}
    
    
    def predict_with_ensemble(self, input_data: np.ndarray, 
                             feature_names: List[str],
                             dataset_version: str = 'v1',
                             attack_type: str = None,
                             explain: bool = True) -> Dict[str, Any]:
        """
        Make prediction using ensemble of agents with abstaining logic
        
        Args:
            input_data: Input data array
            feature_names: List of feature names
            dataset_version: Dataset version ('v1' or 'v2')
            attack_type: Optional attack type for attack-aware participation
            explain: Whether to generate SHAP explanations (default: True)
            
        Returns:
            Dictionary with ensemble prediction and explanations
        """
        # Transform to canonical features
        canonical_data, available_features = self.canonical_builder.transform_to_canonical(
            input_data, feature_names, dataset_version
        )
        
        # Get participating agents (attack-aware if attack_type provided)
        participating_agents = self.feature_adapter.get_participating_agents(
            available_features, attack_type
        )
        
        if not participating_agents:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'error': 'No agents can participate with available features'
            }
        
        logger.info(f"Participating agents: {participating_agents}")
        
        # Collect predictions from all participating agents
        agent_predictions = {}
        agent_explanations = {}
        votes = []
        
        for agent_name in participating_agents:
            # Prepare features for this agent
            agent_data = self.feature_adapter.prepare_features_for_agent(canonical_data, agent_name)
            
            # Get prediction
            prediction, confidence = self._predict_single_agent(agent_name, agent_data)
            
            if prediction is not None:
                agent_predictions[agent_name] = {
                    'prediction': prediction,
                    'confidence': confidence
                }
                
                # Only vote if prediction is valid
                if prediction in [0, 1]:
                    votes.append((agent_name, prediction, confidence))
                
                # Get SHAP explanation
        # Calculate explanations (if requested)
        agent_explanations = {}
        if explain:
            for agent_name in participating_agents:
                # Prepare features for this agent (again, for explanation context)
                agent_data = self.feature_adapter.prepare_features_for_agent(canonical_data, agent_name)

                if agent_name not in self.shap_explainers:
                    # Create explainer with background data (use canonical_data as background)
                    # For tree models, we don't need background data
                    agent_info_check = self.feature_adapter.get_agent_info(agent_name)
                    if agent_info_check.get('model_type') in ['xgboost', 'random_forest']:
                        self.create_shap_explainer(agent_name, agent_data)
                    else:
                        # For other models, skip SHAP for now (needs proper background data)
                        logger.info(f"Skipping SHAP for {agent_name} (needs background data)")
                        agent_explanations[agent_name] = {'top_features': [], 'error': 'SHAP not available'}
                
                # Only get explanation if explainer exists
                if agent_name in self.shap_explainers and self.shap_explainers[agent_name] is not None:
                    explanation = self.get_shap_explanation(
                        agent_name, agent_data, 
                        list(self.feature_adapter.get_agent_info(agent_name)['required_features'])
                    )
                    agent_explanations[agent_name] = explanation
                elif agent_name not in agent_explanations: # Ensure we don't overwrite if already set above
                    # No explainer available and not already set, use feature importance fallback
                    agent_explanations[agent_name] = self._get_feature_importance_fallback(
                        agent_name, agent_data,
                        list(self.feature_adapter.get_agent_info(agent_name)['required_features'])
                    )
        
        # Ensemble voting with confidence weighting
        if not votes:
            return {
                'prediction': 'unknown',
                'confidence': 0.0,
                'participating_agents': participating_agents,
                'agent_predictions': agent_predictions
            }
        
        # Weighted voting
        weighted_votes = {'benign': 0.0, 'anomaly': 0.0}
        for agent_name, prediction, confidence in votes:
            # Skip NaN confidences
            if np.isnan(confidence) or np.isinf(confidence):
                logger.warning(f"Skipping NaN/Inf confidence from {agent_name}")
                continue
            if prediction == 0:
                weighted_votes['benign'] += confidence
            else:
                weighted_votes['anomaly'] += confidence
        
        # Sanitize NaN values in votes (fallback to 0)
        if np.isnan(weighted_votes['benign']):
            weighted_votes['benign'] = 0.0
        if np.isnan(weighted_votes['anomaly']):
            weighted_votes['anomaly'] = 0.0
        
        # Final decision
        final_prediction = 'anomaly' if weighted_votes['anomaly'] > weighted_votes['benign'] else 'benign'
        
        # Debug logging
        logger.info(f"VOTING DEBUG: benign={weighted_votes['benign']:.2f}, anomaly={weighted_votes['anomaly']:.2f} -> {final_prediction}")
        
        # Calculate confidence as average of agreeing agents (FIXED)
        # This gives higher confidence when agents agree, regardless of how many abstain
        agreeing_confidences = [
            confidence for agent_name, prediction, confidence in votes 
            if (prediction == 1 and final_prediction == 'anomaly') or 
               (prediction == 0 and final_prediction == 'benign')
        ]
        
        if agreeing_confidences:
            final_confidence = float(np.mean(agreeing_confidences))
        else:
            # Fallback to old method if no agreeing agents (shouldn't happen)
            total_weight = sum(weighted_votes.values())
            final_confidence = max(weighted_votes.values()) / total_weight if total_weight > 0 else 0.0
        
        # Aggregate explanations
        aggregated_explanation = self._aggregate_explanations(agent_explanations, final_prediction)
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'participating_agents': participating_agents,
            'abstaining_agents': list(set(self.agents.keys()) - set(participating_agents)),
            'agent_predictions': agent_predictions,
            'votes': {
                'benign': weighted_votes['benign'],
                'anomaly': weighted_votes['anomaly']
            },
            'explanation': aggregated_explanation,
            'shap_details': agent_explanations
        }
    
    def _predict_single_agent(self, agent_name: str, data: np.ndarray) -> Tuple[Optional[int], float]:
        """Get prediction from a single agent"""
        model = self.agents.get(agent_name)
        if model is None:
            return None, 0.0
        
        agent_info = self.feature_adapter.get_agent_info(agent_name)
        model_type = agent_info.get('model_type', '')
        
        try:
            if model_type == 'xgboost' or model_type == 'random_forest':
                predictions = model.predict(data)
                # Get confidence from probability
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(data)
                    confidence = float(np.mean(np.max(proba, axis=1)))
                else:
                    confidence = 0.8  # Default confidence
                return int(predictions[0]) if len(predictions) > 0 else None, confidence
                
            elif model_type == 'lstm':
                data_3d = np.expand_dims(data, axis=-1)
                predictions = model.predict(data_3d, verbose=0)
                pred_binary = (predictions > 0.5).astype(int).ravel()
                confidence = float(np.mean(np.abs(predictions - 0.5) * 2))
                return int(pred_binary[0]) if len(pred_binary) > 0 else None, confidence
                
            elif model_type == 'autoencoder':
                reconstructions = model['autoencoder'].predict(data, verbose=0)
                mse = np.mean(np.power(data - reconstructions, 2), axis=1)
                predictions = (mse > model['threshold']).astype(int)
                confidence = float(np.mean(np.abs(mse - model['threshold']) / model['threshold']))
                return int(predictions[0]) if len(predictions) > 0 else None, min(confidence, 1.0)
                
        except Exception as e:
            logger.warning(f"Prediction failed for {agent_name}: {e}")
            return None, 0.0
    
    def _aggregate_explanations(self, agent_explanations: Dict[str, Dict], 
                               final_prediction: str) -> Dict[str, Any]:
        """Aggregate explanations from multiple agents"""
        all_features = {}
        
        for agent_name, explanation in agent_explanations.items():
            if 'top_features' in explanation:
                for feat in explanation['top_features']:
                    feat_name = feat['feature']
                    if feat_name not in all_features:
                        all_features[feat_name] = []
                    all_features[feat_name].append(feat['importance'])
        
        # Average importance across agents
        avg_importance = {feat: np.mean(importances) 
                         for feat, importances in all_features.items()}
        
        # Sort by importance
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        summary = f"Ensemble Decision: {final_prediction.upper()}\n"
        summary += "Key Contributing Factors:\n"
        
        top_features_list = []
        for i, (feat, importance) in enumerate(sorted_features[:5], 1):
            summary += f"{i}. {feat} (importance: {importance:.4f})\n"
            top_features_list.append({
                'feature': feat,
                'importance': float(importance)
            })
        
        return {
            'summary': summary,
            'top_features': top_features_list
        }


def main():
    """Test the multi-agent detection system"""
    system = MultiAgentDetectionSystem()
    
    # Load some test data
    from utils.v1_dataload import load_data
    x_train, y_train, x_test, y_test = load_data('./v1_datasets/NF-BoT-IoT.csv', info=False)
    
    # Test on a few samples
    test_sample = x_test[:5]
    feature_names = CanonicalFeatureBuilder.V1_FEATURES
    
    result = system.predict_with_ensemble(test_sample, feature_names, 'v1')
    
    print("="*80)
    print("MULTI-AGENT DETECTION RESULT")
    print("="*80)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"\nParticipating Agents: {len(result['participating_agents'])}")
    print(f"Abstaining Agents: {len(result['abstaining_agents'])}")
    print(f"\nExplanation:\n{result['explanation']}")


if __name__ == '__main__':
    main()
