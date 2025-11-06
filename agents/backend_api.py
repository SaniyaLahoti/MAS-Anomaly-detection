"""
FastAPI Backend for Multi-Agent Anomaly Detection System
Serves the React frontend and handles predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from interpreter_agent import InterpreterAgent
from llm_agent import LLMAgent
from datetime import datetime
import uvicorn
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI
app = FastAPI(title="Multi-Agent Anomaly Detection API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agents (initialized once)
agent = None
llm_agent = None

class NetworkFlow(BaseModel):
    """Network flow input model"""
    protocol: int
    l7_proto: float
    l4_src_port: int
    l4_dst_port: int
    in_bytes: int
    in_pkts: int
    out_bytes: int
    out_pkts: int
    tcp_flags: int
    flow_duration_ms: int
    actual_label: str = None

class PredictionResponse(BaseModel):
    """Prediction response model"""
    success: bool
    prediction: str
    confidence: float
    agreement: str
    xgboost_prediction: str
    xgboost_confidence: float
    lstm_prediction: str
    lstm_confidence: float
    key_indicators: dict
    threat_assessment: dict
    timestamp: str
    error: str = None

def engineer_features_from_flow(flow: NetworkFlow):
    """Convert network flow to DataFrame with engineered features"""
    # Define the exact feature order as used during training
    # This is critical - order must match training exactly!
    FEATURE_ORDER = [
        'PROTOCOL', 'L7_PROTO', 'L4_SRC_PORT', 'L4_DST_PORT', 
        'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 'OUT_PKTS', 
        'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
        'PACKET_RATE', 'BYTE_RATE', 'AVG_PACKET_SIZE',
        'AVG_IN_PACKET_SIZE', 'AVG_OUT_PACKET_SIZE',
        'BYTE_ASYMMETRY', 'PACKET_ASYMMETRY',
        'IN_OUT_BYTE_RATIO', 'IN_OUT_PACKET_RATIO',
        'PROTOCOL_INTENSITY', 'TCP_PACKET_INTERACTION',
        'PROTOCOL_PORT_COMBO', 'FLOW_INTENSITY'
    ]
    
    # Create base DataFrame
    df = pd.DataFrame([{
        'PROTOCOL': flow.protocol,
        'L7_PROTO': flow.l7_proto,
        'L4_SRC_PORT': flow.l4_src_port,
        'L4_DST_PORT': flow.l4_dst_port,
        'IN_BYTES': flow.in_bytes,
        'IN_PKTS': flow.in_pkts,
        'OUT_BYTES': flow.out_bytes,
        'OUT_PKTS': flow.out_pkts,
        'TCP_FLAGS': flow.tcp_flags,
        'FLOW_DURATION_MILLISECONDS': flow.flow_duration_ms
    }])
    
    # Engineer features (same order as training)
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
    
    # Ensure column order matches training
    df = df[FEATURE_ORDER]
    
    return df

def get_threat_assessment(prediction: str):
    """Get threat assessment for prediction"""
    assessments = {
        'Benign': {
            'level': 'LOW',
            'color': 'green',
            'description': 'Normal network traffic detected',
            'action': 'Continue monitoring',
            'icon': '✅'
        },
        'DDoS': {
            'level': 'CRITICAL',
            'color': 'red',
            'description': 'Distributed Denial of Service attack detected',
            'action': 'Activate DDoS mitigation, enable rate limiting',
            'icon': '🔴'
        },
        'DoS': {
            'level': 'HIGH',
            'color': 'red',
            'description': 'Denial of Service attack detected',
            'action': 'Block source IP, implement rate limiting',
            'icon': '🔴'
        },
        'Reconnaissance': {
            'level': 'MEDIUM',
            'color': 'orange',
            'description': 'Network scanning/probing detected',
            'action': 'Monitor source, check for escalation attempts',
            'icon': '🟡'
        },
        'Theft': {
            'level': 'CRITICAL',
            'color': 'red',
            'description': 'Data exfiltration detected',
            'action': 'Block immediately, investigate data breach',
            'icon': '🔴'
        }
    }
    return assessments.get(prediction, assessments['Benign'])

@app.on_event("startup")
async def startup_event():
    """Initialize the multi-agent system on startup"""
    global agent, llm_agent
    print("🚀 Initializing Multi-Agent System...")
    try:
        agent = InterpreterAgent()
        agent.initialize_agents()
        print("✅ Multi-Agent System ready!")
        
        # Initialize LLM agent (optional - will work without API key for predictions)
        try:
            llm_agent = LLMAgent()  # Loads from .env
            print("✅ LLM Agent ready for chat!")
        except ValueError as e:
            print(f"⚠️  LLM Agent not available: {str(e)}")
            print("   Predictions will work, but chat requires OpenAI API key")
            llm_agent = None
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        agent = None
        llm_agent = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Multi-Agent Anomaly Detection API",
        "status": "online" if agent else "initializing",
        "agents": ["XGBoost", "LSTM", "Interpreter", "LLM"] if agent else []
    }

@app.get("/presets")
async def get_presets():
    """Get predefined network flow presets"""
    presets = {
        "benign": {
            "name": "Benign Traffic",
            "description": "Normal network communication",
            "protocol": 17,
            "l7_proto": 5.0,
            "l4_src_port": 53,
            "l4_dst_port": 51554,
            "in_bytes": 28,
            "in_pkts": 1,
            "out_bytes": 0,
            "out_pkts": 0,
            "tcp_flags": 0,
            "flow_duration_ms": 0,
            "actual_label": "Benign"
        },
        "ddos": {
            "name": "DDoS Attack",
            "description": "Distributed Denial of Service",
            "protocol": 6,
            "l7_proto": 7.0,
            "l4_src_port": 80,
            "l4_dst_port": 443,
            "in_bytes": 5000,
            "in_pkts": 150,
            "out_bytes": 100,
            "out_pkts": 5,
            "tcp_flags": 2,
            "flow_duration_ms": 1000,
            "actual_label": "DDoS"
        },
        "dos": {
            "name": "DoS Attack",
            "description": "Denial of Service",
            "protocol": 6,
            "l7_proto": 7.0,
            "l4_src_port": 8080,
            "l4_dst_port": 80,
            "in_bytes": 8000,
            "in_pkts": 200,
            "out_bytes": 50,
            "out_pkts": 2,
            "tcp_flags": 2,
            "flow_duration_ms": 1500,
            "actual_label": "DoS"
        },
        "reconnaissance": {
            "name": "Port Scan",
            "description": "Network reconnaissance",
            "protocol": 6,
            "l7_proto": 7.0,
            "l4_src_port": 45000,
            "l4_dst_port": 22,
            "in_bytes": 100,
            "in_pkts": 5,
            "out_bytes": 80,
            "out_pkts": 4,
            "tcp_flags": 2,
            "flow_duration_ms": 50,
            "actual_label": "Reconnaissance"
        },
        "theft": {
            "name": "Data Theft",
            "description": "Data exfiltration",
            "protocol": 6,
            "l7_proto": 7.0,
            "l4_src_port": 443,
            "l4_dst_port": 8443,
            "in_bytes": 50000,
            "in_pkts": 1000,
            "out_bytes": 500,
            "out_pkts": 10,
            "tcp_flags": 24,
            "flow_duration_ms": 30000,
            "actual_label": "Theft"
        }
    }
    return presets

@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(flow: NetworkFlow):
    """Predict anomaly for network flow"""
    if not agent:
        raise HTTPException(status_code=503, detail="Multi-Agent System not initialized")
    
    try:
        # Convert to DataFrame with proper feature engineering
        df_input = engineer_features_from_flow(flow)
        
        # Add a placeholder Attack column (required by SHAP explainer)
        # It will be dropped before prediction anyway
        df_input['Attack'] = 'Unknown'
        
        # Get prediction from ensemble
        result = agent.ensemble_predict(
            df_input,
            actual_label=flow.actual_label,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            source_ip="Web-Interface",
            affected_assets="Demo-System"
        )
        
        # Extract results
        ens = result['ensemble_decision']
        agents = result['agent_predictions']
        evidence = result['evidence']
        
        # Get threat assessment
        threat = get_threat_assessment(ens['final_prediction'])
        
        # Set prediction context for LLM chat (if available)
        if llm_agent is not None:
            llm_agent.set_prediction_context(result)
        
        return PredictionResponse(
            success=True,
            prediction=ens['final_prediction'],
            confidence=ens['confidence'],
            agreement=ens['agreement'],
            xgboost_prediction=agents['xgboost']['prediction'],
            xgboost_confidence=agents['xgboost']['confidence'],
            lstm_prediction=agents['lstm']['prediction'],
            lstm_confidence=agents['lstm']['confidence'],
            key_indicators=evidence['key_indicators'],
            threat_assessment=threat,
            timestamp=result['timestamp']
        )
        
    except Exception as e:
        return PredictionResponse(
            success=False,
            prediction="Error",
            confidence=0.0,
            agreement="ERROR",
            xgboost_prediction="Error",
            xgboost_confidence=0.0,
            lstm_prediction="Error",
            lstm_confidence=0.0,
            key_indicators={},
            threat_assessment=get_threat_assessment('Benign'),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=str(e)
        )

class ChatMessage(BaseModel):
    """Chat message model"""
    message: str

class ChatResponse(BaseModel):
    """Chat response model"""
    success: bool
    response: str = ""
    timestamp: str
    error: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """
    Chat endpoint for asking questions about the latest prediction
    """
    global llm_agent
    
    if llm_agent is None:
        return ChatResponse(
            success=False,
            response="",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error="LLM Agent not available. Please configure OpenAI API key in .env file."
        )
    
    try:
        result = llm_agent.chat(chat_message.message)
        
        if result['error']:
            return ChatResponse(
                success=False,
                response="",
                timestamp=result.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                error=result['message']
            )
        
        return ChatResponse(
            success=True,
            response=result['message'],
            timestamp=result['timestamp'],
            error=None
        )
        
    except Exception as e:
        return ChatResponse(
            success=False,
            response="",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=f"Chat error: {str(e)}"
        )

if __name__ == "__main__":
    print("🚀 Starting Multi-Agent Anomaly Detection API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
