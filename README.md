# 🤖 Multi-Agent Network Anomaly Detection System

A sophisticated multi-agent system for real-time network anomaly detection using XGBoost and LSTM models with SHAP explainability.

## 🎯 **System Overview**

This system combines multiple AI agents to detect network attacks with high accuracy and provide explainable results:

- **Agent 1**: XGBoost + SHAP (Hierarchical Classification)
- **Agent 2**: LSTM + SHAP (Deep Learning Classification)  
- **Agent 3**: Interpreter (Ensemble Voting & Evidence Combination)
- **Agent 4**: LLM (Natural Language Security Reports)

## 📁 **Project Structure**

```
MAS-LSTM-1/
├── 🤖 agents/                    # Core AI agents
│   ├── backend_api.py            # FastAPI server
│   ├── interpreter_agent.py      # Ensemble combiner
│   ├── llm_agent.py             # LLM report generator
│   ├── shap_explainer.py        # XGBoost + SHAP
│   └── lstm_shap_explainer.py   # LSTM + SHAP
│
├── 🧠 models/                    # Trained models & artifacts
│   ├── hierarchical/            # Hierarchical XGBoost models
│   ├── lstm/                    # LSTM models & scalers
│   └── xgboost/                 # XGBoost models & data
│
├── 📊 datasets/                  # Training datasets
│   ├── v1_dataset/              # NF-BoT-IoT dataset (primary)
│   └── v2_dataset/              # Extended datasets
│
├── 🌐 web_interface/             # Web frontend
│   └── web_frontend.html        # Interactive demo interface
│
├── 📜 scripts/                   # Training & utility scripts
│   ├── training/                # Model training scripts
│   ├── testing/                 # Test & validation scripts
│   └── utilities/               # Helper utilities
│
├── 📈 results/                   # Training results & reports
│   ├── training/                # Training metrics & logs
│   ├── testing/                 # Test results
│   └── reports/                 # Analysis reports
│
├── 📋 logs/                      # System logs
├── 📚 documentation/             # Project documentation
└── requirements.txt             # Python dependencies
```

## 🚀 **Quick Start**

### 1. **Start the System**
```bash
# Navigate to agents directory
cd agents/

# Start the backend API
python backend_api.py
```

### 2. **Access Web Interface**
```bash
# Start frontend server (in new terminal)
cd web_interface/
python -m http.server 8080

# Open browser to: http://127.0.0.1:8080/web_frontend.html
```

### 3. **Test Attack Detection**
- Click preset buttons for verified attack samples
- Or enter custom network flow parameters
- View real-time multi-agent analysis with SHAP explanations

## 🎯 **Supported Attack Types**

| Attack Type | Description | Key Indicators |
|-------------|-------------|----------------|
| **Benign** | Normal traffic | Low bytes, balanced flow |
| **DDoS** | Distributed DoS | High packet rate, port 80 |
| **DoS** | Denial of Service | Similar to DDoS, single source |
| **Reconnaissance** | Port scanning | Low bytes, multiple ports |
| **Data Theft** | Exfiltration | Very high bytes (millions) |

## 📊 **Performance Metrics**

| Model | Theft Detection | Benign Detection | DDoS Detection |
|-------|----------------|------------------|----------------|
| **XGBoost** | 99.84% | 99.94% | 50.97% |
| **LSTM** | 99.999% | 99.998% | 52.21% |
| **Ensemble** | 99.92% | 99.97% | 51.59% |

## 🔧 **Network Flow Parameters**

| Parameter | Range | Description |
|-----------|-------|-------------|
| `protocol` | 1, 6, 17 | 1=ICMP, 6=TCP, 17=UDP |
| `l7_proto` | 0.0-222.0 | Application protocol |
| `l4_src_port` | 0-65535 | Source port |
| `l4_dst_port` | 0-65535 | Destination port |
| `in_bytes` | 0-2B | Incoming bytes |
| `in_pkts` | 0-50K | Incoming packets |
| `out_bytes` | 0-2B | Outgoing bytes |
| `out_pkts` | 0-50K | Outgoing packets |
| `tcp_flags` | 0-63 | TCP flag bits |
| `flow_duration_ms` | 0-5M | Flow duration |

## 🧠 **Model Architecture**

### **Hierarchical Classification**
1. **Stage 1**: 4-class classification (Benign, DOS, Reconnaissance, Theft)
2. **Stage 2**: Binary classification (DDoS vs DoS) for DOS samples

### **Feature Engineering**
- 10 raw features → 23 engineered features
- Packet rates, byte ratios, asymmetry metrics
- Protocol interactions, flow intensity

### **SHAP Explainability**
- **XGBoost**: TreeExplainer for feature importance
- **LSTM**: GradientExplainer for deep learning insights
- Real-time explanation generation (not hardcoded)

## 🌐 **API Endpoints**

### **GET /** - Health Check
```json
{
  "message": "Multi-Agent Anomaly Detection API",
  "status": "online",
  "agents": ["XGBoost", "LSTM", "Interpreter", "LLM"]
}
```

### **POST /predict** - Anomaly Detection
```json
{
  "protocol": 6,
  "l7_proto": 0.0,
  "l4_src_port": 49160,
  "l4_dst_port": 4444,
  "in_bytes": 217753000,
  "in_pkts": 4521,
  "out_bytes": 199100,
  "out_pkts": 4049,
  "tcp_flags": 24,
  "flow_duration_ms": 4176249
}
```

**Response:**
```json
{
  "success": true,
  "prediction": "Theft",
  "confidence": 0.9992,
  "agreement": "FULL_AGREEMENT",
  "xgboost_prediction": "Theft",
  "lstm_prediction": "Theft",
  "key_indicators": {...},
  "threat_assessment": {
    "level": "CRITICAL",
    "action": "Block immediately, investigate data breach"
  }
}
```

## 🛠 **Development**

### **Training New Models**
```bash
cd scripts/training/

# Train hierarchical XGBoost
python hierarchical_classification.py

# Train hierarchical LSTM
python lstm_hierarchical_classification.py
```

### **Testing & Validation**
```bash
cd scripts/testing/

# Test with real attack samples
python test_real_attacks.py

# Binary model testing
python test_binary_model.py
```

## 📈 **Key Features**

✅ **Multi-Agent Architecture** - Independent models working together  
✅ **Real-Time Detection** - < 500ms response time  
✅ **SHAP Explainability** - Understand why attacks were detected  
✅ **Hierarchical Classification** - Improved attack type distinction  
✅ **Ensemble Voting** - Intelligent prediction combination  
✅ **Web Interface** - Beautiful, interactive demo  
✅ **Verified Results** - Tested with real dataset samples  

## 📚 **Documentation**

- `documentation/SYSTEM_WORKING_AND_TESTED.md` - Complete system verification
- `documentation/DEMO_READY.md` - Demo preparation guide
- `documentation/LSTM_vs_XGBOOST_COMPARISON.md` - Model comparison
- `documentation/SHAP_EXPLAINABILITY_SUMMARY.md` - SHAP implementation details

## 🔬 **Research Context**

This system was developed for network security research, demonstrating:
- Multi-agent AI collaboration
- Explainable AI in cybersecurity
- Hierarchical classification for similar attack types
- Real-time anomaly detection with feature engineering

## 📄 **License**

See `documentation/LICENSE` for license information.

---

**🎓 Ready for academic demonstration and real-world deployment!**
