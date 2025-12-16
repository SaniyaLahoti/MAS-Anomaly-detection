"""
Feature Adapter Layer for Multi-Agent IIOT Anomaly Detection System

This module handles:
1. Canonical feature building - standardizes features across v1 and v2 datasets
2. Feature taxonomy - maps features to agents
3. Agent knowledge base - defines which agents can handle which features/classes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanonicalFeatureBuilder:
    """
    Builds canonical feature representation from raw network data
    Handles differences between v1 and v2 datasets
    """
    
    # Common features present in both v1 and v2
    V1_FEATURES = [
        'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 
        'IN_PKTS', 'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS'
    ]
    
    # Additional features in v2
    V2_ADDITIONAL_FEATURES = [
        'CLIENT_TCP_FLAGS', 'SERVER_TCP_FLAGS', 'DURATION_IN', 'DURATION_OUT',
        'MIN_TTL', 'MAX_TTL', 'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
        'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 'SRC_TO_DST_SECOND_BYTES',
        'DST_TO_SRC_SECOND_BYTES', 'RETRANSMITTED_IN_BYTES', 'RETRANSMITTED_IN_PKTS',
        'RETRANSMITTED_OUT_BYTES', 'RETRANSMITTED_OUT_PKTS', 'SRC_TO_DST_AVG_THROUGHPUT',
        'DST_TO_SRC_AVG_THROUGHPUT', 'NUM_PKTS_UP_TO_128_BYTES', 'NUM_PKTS_128_TO_256_BYTES',
        'NUM_PKTS_256_TO_512_BYTES', 'NUM_PKTS_512_TO_1024_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES',
        'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'ICMP_TYPE', 'ICMP_IPV4_TYPE',
        'DNS_QUERY_ID', 'DNS_QUERY_TYPE', 'DNS_TTL_ANSWER', 'FTP_COMMAND_RET_CODE'
    ]
    
    # All canonical features (union of v1 and v2)
    ALL_CANONICAL_FEATURES = V1_FEATURES + V2_ADDITIONAL_FEATURES
    
    def __init__(self):
        self.feature_mapping = self._build_feature_mapping()
        
    def _build_feature_mapping(self) -> Dict[str, int]:
        """Create mapping from feature name to canonical index"""
        return {feat: idx for idx, feat in enumerate(self.ALL_CANONICAL_FEATURES)}
    
    def transform_to_canonical(self, data: np.ndarray, feature_names: List[str], 
                              dataset_version: str = 'v1') -> Tuple[np.ndarray, List[str]]:
        """
        Transform dataset to canonical feature representation
        
        Args:
            data: Input data array
            feature_names: List of feature names in the data
            dataset_version: 'v1' or 'v2'
            
        Returns:
            Tuple of (canonical_data, available_features)
        """
        n_samples = data.shape[0]
        n_canonical_features = len(self.ALL_CANONICAL_FEATURES)
        
        # Initialize canonical data with zeros (missing features will be 0)
        canonical_data = np.zeros((n_samples, n_canonical_features))
        available_features = []
        
        # Map existing features to canonical positions
        for i, feat_name in enumerate(feature_names):
            if feat_name in self.feature_mapping:
                canonical_idx = self.feature_mapping[feat_name]
                canonical_data[:, canonical_idx] = data[:, i]
                available_features.append(feat_name)
        
        logger.info(f"Transformed {dataset_version} data: {len(available_features)}/{n_canonical_features} features available")
        
        return canonical_data, available_features
    
    def get_feature_mask(self, feature_names: List[str]) -> np.ndarray:
        """
        Get binary mask indicating which canonical features are present
        
        Args:
            feature_names: List of available feature names
            
        Returns:
            Binary mask array
        """
        mask = np.zeros(len(self.ALL_CANONICAL_FEATURES), dtype=bool)
        for feat_name in feature_names:
            if feat_name in self.feature_mapping:
                mask[self.feature_mapping[feat_name]] = True
        return mask


class AttackTaxonomy:
    """
    Unified taxonomy of attack types across all datasets
    Maps attack labels to canonical attack classes
    """
    
    # Known attack types in v1 datasets
    V1_ATTACKS = {
        'Benign': 0,
        'DDoS': 1,
        'DoS': 2,
        'Reconnaissance': 3,
        'Theft': 4
    }
    
    # Known attack types in v2 datasets (extended)
    V2_ATTACKS = {
        'Benign': 0,
        'DDoS': 1,
        'DoS': 2,
        'Reconnaissance': 3,
        'Theft': 4,
        'Backdoor': 5,
        'Injection': 6,
        'Password': 7,
        'Ransomware': 8,
        'XSS': 9,
        'MITM': 10
    }
    
    # Map dataset labels to canonical attack names
    ATTACK_NAME_MAP = {
        'benign': 'Benign',
        'ddos': 'DDoS',
        'dos': 'DoS',
        'reconnaissance': 'Reconnaissance',
        'scanning': 'Reconnaissance',  # scanning is a type of reconnaissance
        'port_scan': 'Reconnaissance',  # port scanning
        'theft': 'Theft',
        'data_theft': 'Theft',  # data theft
        'backdoor': 'Backdoor',
        'injection': 'Injection',
        'password': 'Password',
        'ransomware': 'Ransomware',
        'xss': 'XSS',
        'mitm': 'MITM',
        'unknown': 'Unknown',
    }
    
    # All canonical attack classes
    ALL_ATTACK_CLASSES = set(V1_ATTACKS.keys()) | set(V2_ATTACKS.keys())
    
    @classmethod
    def normalize_attack_type(cls, attack_type: str) -> str:
        """Normalize attack type to canonical form"""
        if not attack_type:
            return 'Unknown'
        lower = attack_type.lower()
        return cls.ATTACK_NAME_MAP.get(lower, attack_type)
    
    @classmethod
    def get_attack_id(cls, attack_name: str, dataset_version: str = 'v1') -> int:
        """Get canonical attack ID for an attack name"""
        normalized = cls.normalize_attack_type(attack_name)
        taxonomy = cls.V2_ATTACKS if dataset_version == 'v2' else cls.V1_ATTACKS
        return taxonomy.get(normalized, -1)  # -1 for unknown
    
    @classmethod
    def is_known_attack(cls, attack_name: str, dataset_version: str) -> bool:
        """Check if attack is known in the given dataset version"""
        normalized = cls.normalize_attack_type(attack_name)
        taxonomy = cls.V2_ATTACKS if dataset_version == 'v2' else cls.V1_ATTACKS
        return normalized in taxonomy


class FeatureAdapter:
    """
    Feature Adapter Layer - manages agent capabilities and feature routing
    Determines which agents can participate in prediction based on available features
    """
    
    def __init__(self):
        self.canonical_builder = CanonicalFeatureBuilder()
        self.agent_capabilities = self._define_agent_capabilities()
        
    def _define_agent_capabilities(self) -> Dict[str, Dict]:
        """
        Define capabilities of each agent:
        - dataset_version: which dataset they were trained on
        - required_features: minimum features needed
        - known_attacks: attacks they can detect
        """
        return {
            'agent_1_xgboost_v1': {
                'dataset_version': 'v1',
                'model_type': 'xgboost',
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),
                'known_attacks': set(AttackTaxonomy.V1_ATTACKS.keys()),
                'feature_count': len(CanonicalFeatureBuilder.V1_FEATURES)
            },
            'agent_2_xgboost_v2': {
                'dataset_version': 'v2',
                'model_type': 'xgboost',
                # V2 agents can participate with V1 features (padded to 39 with zeros)
                # This allows them to vote on all attack types
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),
                'known_attacks': set(AttackTaxonomy.V2_ATTACKS.keys()),
                'feature_count': len(CanonicalFeatureBuilder.ALL_CANONICAL_FEATURES)
            },
            'agent_3_lstm_v1': {
                'dataset_version': 'v1',
                'model_type': 'lstm',
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),
                'known_attacks': set(AttackTaxonomy.V1_ATTACKS.keys()),
                'feature_count': len(CanonicalFeatureBuilder.V1_FEATURES)
            },
            'agent_4_lstm_v2': {
                'dataset_version': 'v2',
                'model_type': 'lstm',
                # V2 agents can participate with V1 features (padded to 39 with zeros)
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),
                'known_attacks': set(AttackTaxonomy.V2_ATTACKS.keys()),
                'feature_count': len(CanonicalFeatureBuilder.ALL_CANONICAL_FEATURES)
            },
            'agent_5_rf_v1': {
                'dataset_version': 'v1',
                'model_type': 'random_forest',
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),
                'known_attacks': set(AttackTaxonomy.V1_ATTACKS.keys()),
                'feature_count': len(CanonicalFeatureBuilder.V1_FEATURES)
            },
            'agent_6_rf_v2': {
                'dataset_version': 'v2',
                'model_type': 'random_forest',
                # V2 agents can participate with V1 features (padded to 39 with zeros)
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),
                'known_attacks': set(AttackTaxonomy.V2_ATTACKS.keys()),
                'feature_count': len(CanonicalFeatureBuilder.ALL_CANONICAL_FEATURES)
            },
            'agent_7_unknown': {
                'dataset_version': 'both',
                'model_type': 'autoencoder',
                'required_features': set(CanonicalFeatureBuilder.V1_FEATURES),  # Uses common features
                'known_attacks': set(list(AttackTaxonomy.V1_ATTACKS.keys()) + list(AttackTaxonomy.V2_ATTACKS.keys()) + ['Unknown']),  # Can detect all attacks via anomaly detection
                'feature_count': len(CanonicalFeatureBuilder.V1_FEATURES)
            }
        }
    
    def can_agent_participate(self, agent_name: str, available_features: Set[str]) -> bool:
        """
        Determine if an agent can participate based on available features
        
        Args:
            agent_name: Name of the agent
            available_features: Set of available feature names
            
        Returns:
            True if agent has sufficient features to participate
        """
        if agent_name not in self.agent_capabilities:
            return False
        
        required = self.agent_capabilities[agent_name]['required_features']
        
        # Agent can participate if at least 80% of required features are available
        overlap = len(required & available_features)
        threshold = 0.8 * len(required)
        
        can_participate = overlap >= threshold
        
        if not can_participate:
            logger.debug(f"{agent_name} cannot participate: {overlap}/{len(required)} features available")
        
        return can_participate
    
    def should_agent_participate_for_attack(self, agent_name: str, 
                                            attack_type: str,
                                            available_features: Set[str]) -> bool:
        """
        Determine if agent should participate based on attack type and features
        
        STRICT ATTACK-AWARE SELECTION:
        - Only agents trained on this specific attack type can participate
        - Ensures specialized detection based on training data
        
        Args:
            agent_name: Name of the agent
            attack_type: Type of attack being detected
            available_features: Set of available feature names
            
        Returns:
            True if agent should participate
        """
        if agent_name not in self.agent_capabilities:
            return False
        
        # First check: Does agent have sufficient features?
        capabilities = self.agent_capabilities[agent_name]
        required = capabilities['required_features']
        overlap = len(required & available_features)
        threshold = 0.8 * len(required)
        
        if overlap < threshold:
            logger.debug(f"{agent_name} lacks features for participation")
            return False
        
        # Second check: Was agent trained on this attack type?
        if attack_type:
            normalized_attack = AttackTaxonomy.normalize_attack_type(attack_type)
            known_attacks = capabilities['known_attacks']
            
            # Strict rule: Only participate if trained on this attack
            if normalized_attack not in known_attacks:
                logger.info(f"⛔ {agent_name} abstaining - not trained on '{normalized_attack}'")
                return False
            else:
                logger.info(f"✅ {agent_name} participating - trained on '{normalized_attack}'")
                return True
        
        # If no attack_type specified, allow based on features only
        return True
    
    def get_participating_agents(self, available_features: List[str], 
                                attack_type: str = None) -> List[str]:
        """
        Get list of agents that can participate given available features and attack type
        
        Args:
            available_features: List of available feature names
            attack_type: Optional attack type for attack-aware participation
            
        Returns:
            List of agent names that can participate
        """
        available_set = set(available_features)
        participating = []
        
        if attack_type:
            # Attack-aware participation
            for agent_name in self.agent_capabilities.keys():
                if self.should_agent_participate_for_attack(agent_name, attack_type, available_set):
                    participating.append(agent_name)
        else:
            # Feature-only participation (backward compatibility)
            for agent_name in self.agent_capabilities.keys():
                if self.can_agent_participate(agent_name, available_set):
                    participating.append(agent_name)
        
        logger.info(f"Participating agents for {attack_type or 'unknown'}: {len(participating)}/7")
        return participating
    
    def prepare_features_for_agent(self, canonical_data: np.ndarray, 
                                   agent_name: str) -> np.ndarray:
        """
        Extract relevant features for a specific agent from canonical data
        
        Args:
            canonical_data: Full canonical feature array
            agent_name: Name of the agent
            
        Returns:
            Feature array for the specific agent
        """
        if agent_name not in self.agent_capabilities:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        capabilities = self.agent_capabilities[agent_name]
        
        # Use feature_count to determine how many features the agent needs
        # V1 agents need 8 features, V2 agents need 39 features
        feature_count = capabilities['feature_count']
        
        if feature_count == len(CanonicalFeatureBuilder.V1_FEATURES):
            # V1 agent - extract only V1 features
            required_features = CanonicalFeatureBuilder.V1_FEATURES
        else:
            # V2 agent - needs all canonical features
            required_features = CanonicalFeatureBuilder.ALL_CANONICAL_FEATURES
        
        # Get indices of required features in canonical representation
        indices = []
        for feat_name in required_features:
            if feat_name in self.canonical_builder.feature_mapping:
                indices.append(self.canonical_builder.feature_mapping[feat_name])
        
        indices = sorted(indices)
        
        # Extract relevant features
        agent_data = canonical_data[:, indices]
        
        return agent_data
    
    def can_agent_classify(self, agent_name: str, predicted_class: str) -> bool:
        """
        Check if an agent knows about a predicted attack class
        
        Args:
            agent_name: Name of the agent
            predicted_class: Predicted attack class name
            
        Returns:
            True if agent knows this class, False if should abstain
        """
        if agent_name not in self.agent_capabilities:
            return False
        
        known_attacks = self.agent_capabilities[agent_name]['known_attacks']
        return predicted_class in known_attacks
    
    def get_agent_info(self, agent_name: str) -> Dict:
        """Get capability information for an agent"""
        return self.agent_capabilities.get(agent_name, {})
