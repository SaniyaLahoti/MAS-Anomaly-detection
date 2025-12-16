"""
Centralized State Management for Multi-Agent IIoT Anomaly Detection System

This module defines the shared state that all agents in the LangGraph workflow
can access and modify. It ensures data consistency and enables seamless
communication between agents.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime


@dataclass
class AnomalyState:
    """
    Central state for multi-agent anomaly detection system.
    
    This state object is shared across all agents in the LangGraph workflow,
    allowing them to pass data and results between each other.
    """
    
    # ==================== INPUT DATA ====================
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    
    # ==================== RAW DATA ====================
    raw_data: Optional[Dict[str, Any]] = None
    data_info: Optional[Dict[str, Any]] = None
    
    # ==================== PREPROCESSED DATA ====================
    processed_data: Optional[Dict[str, Any]] = None
    features: Optional[List[str]] = None
    feature_names: Optional[List[str]] = None
    
    # ==================== MODEL PREDICTIONS ====================
    lstm_predictions: Optional[np.ndarray] = None
    transformer_predictions: Optional[np.ndarray] = None
    gnn_predictions: Optional[np.ndarray] = None
    autoencoder_predictions: Optional[np.ndarray] = None
    
    # ==================== ENSEMBLE RESULTS ====================
    ensemble_prediction: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    model_weights: Optional[Dict[str, float]] = None
    
    # ==================== PERFORMANCE METRICS ====================
    individual_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ensemble_metrics: Optional[Dict[str, float]] = None
    
    # Accuracy metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # ==================== EXPLAINABILITY ====================
    explanations: Optional[List[str]] = None
    feature_importance: Optional[Dict[str, float]] = None
    anomaly_samples: Optional[List[Dict[str, Any]]] = None
    
    # ==================== SYSTEM STATUS ====================
    current_step: str = "initialized"
    workflow_status: str = "pending"
    processing_time: Optional[Dict[str, float]] = None
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # ==================== CONFIGURATION ====================
    config: Optional[Dict[str, Any]] = None
    
    # ==================== TIMESTAMPS ====================
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize default values after object creation"""
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def add_error(self, error_message: str) -> None:
        """Add an error message to the state"""
        self.errors.append(f"[{datetime.now().strftime('%H:%M:%S')}] {error_message}")
        self.workflow_status = "error"
    
    def add_warning(self, warning_message: str) -> None:
        """Add a warning message to the state"""
        self.warnings.append(f"[{datetime.now().strftime('%H:%M:%S')}] {warning_message}")
    
    def update_step(self, step_name: str) -> None:
        """Update the current processing step"""
        self.current_step = step_name
        print(f"🔄 Step: {step_name}")
    
    def set_processing_time(self, agent_name: str, duration: float) -> None:
        """Record processing time for an agent"""
        if self.processing_time is None:
            self.processing_time = {}
        self.processing_time[agent_name] = duration
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state"""
        return {
            "dataset": self.dataset_name,
            "current_step": self.current_step,
            "workflow_status": self.workflow_status,
            "has_errors": len(self.errors) > 0,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "processing_time": self.processing_time,
            "metrics": {
                "accuracy": self.accuracy,
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score
            } if self.ensemble_metrics else None
        }
    
    def is_ready_for_next_step(self) -> bool:
        """Check if the state is ready for the next processing step"""
        return len(self.errors) == 0 and self.workflow_status != "error"
    
    def finalize(self) -> None:
        """Finalize the processing and record end time"""
        self.end_time = datetime.now()
        self.workflow_status = "completed"
        
        if self.start_time:
            total_time = (self.end_time - self.start_time).total_seconds()
            print(f"✅ Total processing time: {total_time:.2f} seconds")


# ==================== HELPER FUNCTIONS ====================

def create_initial_state(dataset_name: str, config: Optional[Dict[str, Any]] = None) -> AnomalyState:
    """
    Create an initial state for the anomaly detection workflow
    
    Args:
        dataset_name: Name of the dataset to process
        config: Optional configuration dictionary
    
    Returns:
        AnomalyState: Initialized state object
    """
    state = AnomalyState(
        dataset_name=dataset_name,
        config=config or {},
        current_step="initialized",
        workflow_status="pending"
    )
    
    print(f"🚀 Initialized state for dataset: {dataset_name}")
    return state


def validate_state(state: AnomalyState) -> bool:
    """
    Validate the current state for required fields
    
    Args:
        state: AnomalyState to validate
    
    Returns:
        bool: True if state is valid, False otherwise
    """
    if not state.dataset_name:
        state.add_error("Dataset name is required")
        return False
    
    if state.workflow_status == "error":
        return False
    
    return True


def log_state_progress(state: AnomalyState) -> None:
    """Log the current progress of the state"""
    summary = state.get_summary()
    print(f"\n📊 State Progress:")
    print(f"   Dataset: {summary['dataset']}")
    print(f"   Step: {summary['current_step']}")
    print(f"   Status: {summary['workflow_status']}")
    
    if summary['has_errors']:
        print(f"   ⚠️ Errors: {summary['error_count']}")
        for error in state.errors[-3:]:  # Show last 3 errors
            print(f"      - {error}")
    
    if summary['warning_count'] > 0:
        print(f"   ⚠️ Warnings: {summary['warning_count']}")
    
    if summary['processing_time']:
        print(f"   ⏱️ Processing Times:")
        for agent, time in summary['processing_time'].items():
            print(f"      - {agent}: {time:.2f}s")
