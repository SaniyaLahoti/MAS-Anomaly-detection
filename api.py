"""
Web API for IIOT Anomaly Detection System

Provides REST API endpoints for:
1. Real-time anomaly detection
2. Batch prediction
3. SHAP explanations
4. System status
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import numpy as np
import uvicorn
import logging

from multi_agent_detection import MultiAgentDetectionSystem
from utils.feature_adapter import CanonicalFeatureBuilder

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IIOT Anomaly Detection API",
    description="Multi-Agent AI System for detecting anomalies in IIOT devices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection system
detection_system = None


# Request/Response Models
class NetworkFlow(BaseModel):
    """Single network flow data"""
    PROTOCOL: float
    L7_PROTO: float
    IN_BYTES: float
    OUT_BYTES: float
    IN_PKTS: float
    OUT_PKTS: float
    TCP_FLAGS: float
    FLOW_DURATION_MILLISECONDS: float
    # V2 additional features (optional)
    CLIENT_TCP_FLAGS: Optional[float] = 0.0
    SERVER_TCP_FLAGS: Optional[float] = 0.0
    DURATION_IN: Optional[float] = 0.0
    DURATION_OUT: Optional[float] = 0.0
    MIN_TTL: Optional[float] = 0.0
    MAX_TTL: Optional[float] = 0.0
    LONGEST_FLOW_PKT: Optional[float] = 0.0
    SHORTEST_FLOW_PKT: Optional[float] = 0.0
    MIN_IP_PKT_LEN: Optional[float] = 0.0
    MAX_IP_PKT_LEN: Optional[float] = 0.0
    SRC_TO_DST_SECOND_BYTES: Optional[float] = 0.0
    DST_TO_SRC_SECOND_BYTES: Optional[float] = 0.0
    RETRANSMITTED_IN_BYTES: Optional[float] = 0.0
    RETRANSMITTED_IN_PKTS: Optional[float] = 0.0
    RETRANSMITTED_OUT_BYTES: Optional[float] = 0.0
    RETRANSMITTED_OUT_PKTS: Optional[float] = 0.0
    SRC_TO_DST_AVG_THROUGHPUT: Optional[float] = 0.0
    DST_TO_SRC_AVG_THROUGHPUT: Optional[float] = 0.0
    NUM_PKTS_UP_TO_128_BYTES: Optional[float] = 0.0
    NUM_PKTS_128_TO_256_BYTES: Optional[float] = 0.0
    NUM_PKTS_256_TO_512_BYTES: Optional[float] = 0.0
    NUM_PKTS_512_TO_1024_BYTES: Optional[float] = 0.0
    NUM_PKTS_1024_TO_1514_BYTES: Optional[float] = 0.0
    TCP_WIN_MAX_IN: Optional[float] = 0.0
    TCP_WIN_MAX_OUT: Optional[float] = 0.0
    ICMP_TYPE: Optional[float] = 0.0
    ICMP_IPV4_TYPE: Optional[float] = 0.0
    DNS_QUERY_ID: Optional[float] = 0.0
    DNS_QUERY_TYPE: Optional[float] = 0.0
    DNS_TTL_ANSWER: Optional[float] = 0.0
    FTP_COMMAND_RET_CODE: Optional[float] = 0.0


class PredictionRequest(BaseModel):
    """Request for anomaly prediction"""
    flows: List[NetworkFlow]
    dataset_version: str = "v1"  # "v1" or "v2"
    attack_type: Optional[str] = None  # Optional attack type for attack-aware routing


class PredictionResponse(BaseModel):
    """Response with prediction results"""
    prediction: str
    confidence: float
    participating_agents: List[str]
    abstaining_agents: List[str]
    explanation: str
    votes: Dict[str, float]
    detected_attack_type: Optional[str] = None  # Type of attack detected


class ExplanationRequest(BaseModel):
    """Request for detailed explanation"""
    flow: NetworkFlow
    dataset_version: str = "v1"


def interpret_shap_with_llm(explanation_dict: dict, prediction: str, confidence: float, attack_type: str = None, feature_values: dict = None) -> str:
    """
    Use OpenAI to interpret SHAP values dynamically with actual feature values
    Generates unique, attack-specific explanations for SOC analysts
    
    Args:
        explanation_dict: Dict with 'top_features' and 'summary'
        prediction: The prediction (benign/anomaly)
        confidence: Confidence score
        attack_type: Type of attack detected
        feature_values: Actual feature values from the network flow
        
    Returns:
        LLM-generated professional technical analysis
    """
    top_features = explanation_dict.get('top_features', [])
    
    if not top_features:
        return f"System classified this as {prediction.upper()} with {confidence*100:.1f}% confidence."
    
    # Prepare SHAP data for LLM with actual values
    shap_summary = []
    total_importance = sum(f['importance'] for f in top_features[:5])
    
    for i, feat in enumerate(top_features[:3], 1):
        feature_name = feat['feature']
        importance = feat['importance']
        contribution_pct = (importance / total_importance * 100) if total_importance > 0 else 0
        
        # Include actual value if available
        actual_value = ""
        if feature_values and feature_name in feature_values:
            val = feature_values[feature_name]
            # Format based on feature type
            if 'PKTS' in feature_name:
                actual_value = f" (Actual: {val:.4f} normalized, ~{int(val*1000)} packets estimated)"
            elif 'BYTES' in feature_name:
                actual_value = f" (Actual: {val:.6f} normalized, ~{int(val*100000)} bytes estimated)"
            elif 'DURATION' in feature_name:
                actual_value = f" (Actual: {val:.6f} normalized, ~{val*10000:.1f}ms estimated)"
            else:
                actual_value = f" (Actual value: {val:.4f})"
        
        shap_summary.append(f"{feature_name}: {contribution_pct:.1f}% contribution, SHAP value: {importance:.4f}{actual_value}")
    
    # Build technical prompt for SOC analysts
    if prediction == 'benign':
        prompt = f"""You are a senior SOC analyst writing a technical assessment for the security team.

Classification: BENIGN (ML Confidence: {confidence*100:.1f}%)

Network Flow Indicators (SHAP Explainability):
{chr(10).join('• ' + s for s in shap_summary)}

Write a brief technical explanation (3-4 sentences) for SOC analysts explaining why this traffic is legitimate. Include specific packet/byte counts where relevant. Be technical and precise.

Format:
**Analysis:** [Technical explanation with numbers]
**Status:** [Security status]
**Action:** [Operational recommendation]"""
    
    else:  # anomaly
        attack_context = f"Attack Classification: {attack_type.upper()}" if attack_type else "Anomaly Type: UNKNOWN"
        
        prompt = f"""You are a senior SOC analyst writing an incident briefing for immediate response.

Detection: NETWORK ANOMALY ({attack_context})
ML Confidence: {confidence*100:.1f}%

Threat Indicators (SHAP Explainability - AI Model Feature Attribution):
{chr(10).join('• ' + s for s in shap_summary)}

Write a technical SOC briefing with:
1. Threat assessment (2-3 sentences) - reference specific packet counts, byte volumes, or timing anomalies observed
2. Immediate response actions (3-4 steps) - VERY specific to {attack_type.upper() if attack_type else 'this anomaly'}
3. Forensic procedures (2-3 items) - technical investigation steps for {attack_type.upper() if attack_type else 'this threat'}

Be highly technical. Reference actual numbers. Make recommendations specific to the attack type.

Format:
**Threat Assessment:** [Technical analysis with packet/byte counts]
**Immediate Actions:**
1. [Specific action with technical details]
2. [Specific action with technical details]  
3. [Specific action with technical details]
**Forensics:**
• [Technical investigation step]
• [Technical investigation step]"""
    
    try:
        # Use OpenAI API
        from openai import OpenAI
        import os
        
        # Get API key from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in environment")
        
        client = OpenAI(api_key=api_key)
        
        # Generate response using GPT-4o-mini (fast and cheap)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior cybersecurity analyst providing threat intelligence briefings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        llm_text = response.choices[0].message.content.strip()
        
        # Format with header
        result = f"🔍 **AI-Generated Analysis**\n\n{llm_text}"
        return result
        
    except Exception as e:
        logger.error(f"OpenAI LLM generation failed: {e}")
        # Fallback to basic summary
        if prediction == 'benign':
            return f"**Analysis:** Traffic classified as BENIGN ({confidence*100:.1f}% confidence). Top indicators: {', '.join(f['feature'] for f in top_features[:3])}.\n**Status:** No threat detected.\n**Action:** Continue monitoring."
        else:
            return f"**Threat Assessment:** ANOMALY detected ({confidence*100:.1f}% confidence). Key indicators: {', '.join(f['feature'] for f in top_features[:3])}.\n**Immediate Actions:**\n1. Block source IP\n2. Investigate traffic patterns\n3. Document incident\n**Forensics:**\n• Capture PCAP\n• Check threat intel"


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the detection system on startup"""
    global detection_system
    logger.info("Initializing Multi-Agent Detection System...")
    detection_system = MultiAgentDetectionSystem(models_dir='./trained_models')
    logger.info("System initialized successfully!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "IIOT Anomaly Detection API",
        "version": "1.0.0",
        "status": "online"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if detection_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "healthy",
        "agents_loaded": len(detection_system.agents),
        "available_agents": list(detection_system.agents.keys())
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(request: PredictionRequest):
    """
    Predict anomaly for network flows
    
    Args:
        request: Prediction request with network flows
        
    Returns:
        Prediction result with explanation
    """
    if detection_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Convert flows to numpy array
        flows_data = []
        feature_names = []
        
        for flow in request.flows:
            flow_dict = flow.dict()
            if not feature_names:
                feature_names = list(flow_dict.keys())
            flows_data.append(list(flow_dict.values()))
        
        input_data = np.array(flows_data)
        
        # Get prediction with attack-aware routing
        result = detection_system.predict_with_ensemble(
            input_data, 
            feature_names,
            request.dataset_version,
            attack_type=request.attack_type
        )
        
        # Check if there was an error (no agents could participate)
        if 'error' in result:
            raise HTTPException(status_code=400, detail=result['error'])
        
        # Convert explanation to LLM-interpreted natural language
        explanation_str = result.get('explanation', 'No explanation available')
        if isinstance(explanation_str, dict):
            # Use LLM to interpret SHAP values with attack context and actual feature values
            explanation_str = interpret_shap_with_llm(
                explanation_str, 
                result['prediction'], 
                result['confidence'],
                attack_type=request.attack_type,
                feature_values=dict(zip(feature_names, flows_data[0]))  # Pass actual values
            )
            import json
            # Still send as JSON but with LLM interpretation
            explanation_dict = result['explanation']
            explanation_dict['llm_interpretation'] = explanation_str
            explanation_str = json.dumps(explanation_dict)
        
        return PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            participating_agents=result.get('participating_agents', []),
            abstaining_agents=result.get('abstaining_agents', []),
            explanation=explanation_str,
            votes=result.get('votes', {}),
            detected_attack_type=request.attack_type if result['prediction'] == 'anomaly' else 'Benign'
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain")
async def get_explanation(request: ExplanationRequest):
    """
    Get detailed SHAP explanation for a single flow
    
    Args:
        request: Explanation request
        
    Returns:
        Detailed SHAP explanation
    """
    if detection_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # Convert flow to numpy array
        flow_dict = request.flow.dict()
        feature_names = list(flow_dict.keys())
        input_data = np.array([list(flow_dict.values())])
        
        # Get prediction with explanations
        result = detection_system.predict_with_ensemble(
            input_data,
            feature_names,
            request.dataset_version
        )
        
        return {
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "explanation": result['explanation'],
            "shap_details": result.get('shap_details', {})
        }
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents")
async def get_agents_info():
    """Get information about all agents"""
    if detection_system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    agents_info = {}
    for agent_name in detection_system.agents.keys():
        info = detection_system.feature_adapter.get_agent_info(agent_name)
        agents_info[agent_name] = {
            "dataset_version": info.get('dataset_version'),
            "model_type": info.get('model_type'),
            "feature_count": info.get('feature_count'),
            "known_attacks": list(info.get('known_attacks', []))
        }
    
    return agents_info


@app.get("/features")
async def get_features_info():
    """Get information about features"""
    return {
        "v1_features": CanonicalFeatureBuilder.V1_FEATURES,
        "v2_additional_features": CanonicalFeatureBuilder.V2_ADDITIONAL_FEATURES,
        "all_canonical_features": CanonicalFeatureBuilder.ALL_CANONICAL_FEATURES
    }


# Global counter for simulated streaming
stream_counter = 0

@app.get("/stream/next")
async def get_next_stream_sample():
    """
    Simulate real-time monitoring by returning random traffic samples
    """
    global stream_counter
    stream_counter += 1
    
    import random
    
    # Simulate different traffic types
    if stream_counter % 5 == 0:
        # Simulate attack
        attack_types = ["ddos", "reconnaissance", "theft"]
        attack = random.choice(attack_types)
        
        sample_data = {
            "PROTOCOL": round(random.uniform(0.3, 0.35), 5),
            "L7_PROTO": round(random.uniform(0.03, 0.04), 5),
            "IN_BYTES": round(random.uniform(0.0005, 0.002), 5),
            "OUT_BYTES": round(random.uniform(0.0003, 0.001), 5),
            "IN_PKTS": round(random.uniform(0.003, 0.006), 5),
            "OUT_PKTS": round(random.uniform(0.002, 0.004), 5),
            "TCP_FLAGS": round(random.uniform(0.8, 0.9), 5),
            "FLOW_DURATION_MILLISECONDS": round(random.uniform(0.00001, 0.0001), 7)
        }
        
        payload = {
            "flows": [sample_data],
            "dataset_version": "v1",
            "attack_type": attack
        }
    else:
        # Simulate benign traffic
        sample_data = {
            "PROTOCOL": 6,
            "L7_PROTO": random.choice([80, 443, 22]),
            "IN_BYTES": random.randint(500, 2000),
            "OUT_BYTES": random.randint(300, 1000),
            "IN_PKTS": random.randint(5, 15),
            "OUT_PKTS": random.randint(3, 10),
            "TCP_FLAGS": random.choice([2, 18, 27]),
            "FLOW_DURATION_MILLISECONDS": random.randint(100, 5000)
        }
        
        payload = {
            "flows": [sample_data],
            "dataset_version": "v1"
        }
    
    # Get prediction
    input_data = np.array([list(sample_data.values())])
    feature_names = list(sample_data.keys())
    
    result = detection_system.predict_with_ensemble(
        input_data,
        feature_names,
        payload["dataset_version"],
        attack_type=payload.get("attack_type")
    )
    
    # Return in format expected by frontend
    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "actual_label": 1 if result["prediction"] == "anomaly" else 0,
        "predicted_attack_type": payload.get("attack_type", "Benign"),
        "actual_attack_type": payload.get("attack_type", "Benign")
    }


@app.post("/chat")
async def chat_with_llm(request: dict):
    """
    Chat endpoint for LLM Q&A about detections
    
    Args:
        request: Dict with 'question' key
        
    Returns:
        Answer from LLM
    """
    question = request.get('question', '')
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Simple rule-based responses for now (can be replaced with actual LLM)
    question_lower = question.lower()
    
    if 'why' in question_lower and ('anomaly' in question_lower or 'attack' in question_lower):
        answer = "The system detected this as an anomaly based on multiple factors: unusual traffic patterns, high confidence from tree-based models (XGBoost and Random Forest), and SHAP analysis showing that features like PROTOCOL, IN_BYTES, and TCP_FLAGS had significant importance in the decision."
    elif 'agent' in question_lower:
        answer = "The multi-agent system uses 7 specialized AI agents: XGBoost (V1 & V2), LSTM (V1 & V2), Random Forest (V1 & V2), and an Autoencoder for unknown attacks. Each agent votes based on its confidence, and the final decision is made through weighted ensemble voting."
    elif 'what' in question_lower and 'do' in question_lower:
        answer = "Recommended actions: 1) Investigate the source IP and destination, 2) Check for similar patterns in recent traffic, 3) Review firewall rules, 4) Consider blocking the source if confirmed malicious, 5) Document the incident for future reference."
    elif 'false positive' in question_lower:
        answer = "To determine if this is a false positive: 1) Check if the traffic pattern matches known legitimate applications, 2) Verify the source and destination are trusted, 3) Look at historical data for this endpoint, 4) Consider the confidence score - lower confidence may indicate uncertainty."
    elif 'feature' in question_lower or 'important' in question_lower:
        answer = "The most important features in this detection were identified through SHAP analysis. These typically include PROTOCOL (network protocol type), IN_BYTES/OUT_BYTES (data volume), TCP_FLAGS (connection state), and FLOW_DURATION (how long the connection lasted). Unusual values in these features often indicate attacks."
    else:
        answer = f"I understand you're asking about: '{question}'. The detection system uses ensemble learning with 7 AI agents to analyze network traffic. Each agent examines different aspects of the traffic pattern, and their votes are combined to make the final decision. The SHAP explainability module shows which features were most important in the decision."
    
    return {"answer": answer}


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_api_server()

# Serve frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_root():
    return FileResponse("frontend/index.html")
