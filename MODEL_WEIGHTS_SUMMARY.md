# Model Weights Summary - Quick Reference

## ✅ All Models Successfully Saved and Verified

### Model Files Overview

| Model Type | Dataset | Stage 1 (.pkl) | Stage 2 (.pkl) | Size | Status |
|------------|---------|----------------|----------------|------|--------|
| **XGBoost V1** | V1 (600K) | `hierarchical/hierarchical_stage1_model.pkl` | `hierarchical/hierarchical_stage2_model.pkl` | 8.2MB + 7.5MB | ✅ Verified |
| **XGBoost V2** | V2 (125K) | `v2_hierarchical/xgboost_stage1.pkl` | `v2_hierarchical/xgboost_stage2.pkl` | 4.3MB + 1.4MB | ✅ Verified |
| **LSTM V1** | V1 (600K) | `lstm/lstm_hierarchical_stage1_model.pkl` | `lstm/lstm_hierarchical_stage2_model.pkl` | 567KB + 615KB | ✅ Verified |
| **LSTM V2** | V2 (125K) | `v2_lstm/lstm_stage1_model.pkl` | `v2_lstm/lstm_stage2_model.pkl` | 567KB + 615KB | ✅ Verified |

---

## Performance on Test Data

| Model | Dataset | Test Samples | Accuracy | Verification Status |
|-------|---------|--------------|----------|---------------------|
| XGBoost V1 | V1 | 120,020 | **90.85%** | ✅ Predictions 100% match |
| XGBoost V2 | V2 | 25,000 | **99.36%** | ✅ Verified |
| LSTM V1 | V1 | 120,020 | **87.65%** | ✅ Predictions 100% match (0.00e+00 diff) |
| LSTM V2 | V2 | 25,000 | **99.01%** | ✅ Predictions 100% match (0.00e+00 diff) |

---

## Loading Models - Quick Guide

### XGBoost Models

```python
import joblib

# XGBoost V1 - Stage 1 (4-class)
model_xgb_v1_s1 = joblib.load('models/hierarchical/hierarchical_stage1_model.pkl')
scaler_xgb_v1_s1 = np.load('models/hierarchical/hierarchical_stage1_scaler.npy', allow_pickle=True).item()
encoder_xgb_v1_s1 = np.load('models/hierarchical/hierarchical_stage1_encoder.npy', allow_pickle=True)

# XGBoost V2 - Stage 1 (4-class)
model_xgb_v2_s1 = joblib.load('models/v2_hierarchical/xgboost_stage1.pkl')
scaler_xgb_v2_s1 = joblib.load('models/v2_hierarchical/scaler_stage1.pkl')
encoder_xgb_v2_s1 = joblib.load('models/v2_hierarchical/label_encoder_stage1.pkl')

# Make predictions
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
pred_labels = encoder.inverse_transform(predictions)
```

### LSTM Models

```python
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# LSTM V1 - Stage 1 (4-class)
model_lstm_v1_s1 = joblib.load('models/lstm/lstm_hierarchical_stage1_model.pkl')

# Load and reconstruct scaler
scaler_mean = np.load('models/lstm/lstm_hierarchical_s1_scaler_mean.npy')
scaler_scale = np.load('models/lstm/lstm_hierarchical_s1_scaler_scale.npy')
scaler = StandardScaler()
scaler.mean_ = scaler_mean
scaler.scale_ = scaler_scale

# Load encoder
encoder = np.load('models/lstm/lstm_hierarchical_s1_encoder.npy', allow_pickle=True)

# Make predictions
X_scaled = scaler.transform(X)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
predictions = model_lstm_v1_s1.predict(X_lstm, verbose=0)
pred_classes = np.argmax(predictions, axis=1)
pred_labels = [encoder[i] for i in pred_classes]
```

### Alternative: Load from Original Formats

```python
# XGBoost from .json
import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model('models/hierarchical/hierarchical_stage1_model.json')

# LSTM from .h5
from tensorflow.keras.models import load_model
model = load_model('models/lstm/lstm_hierarchical_stage1_model.h5')
```

---

## Verification Scripts

### Save All Models
```bash
cd scripts/
python save_all_model_weights.py
```

**Output:** Saves all models to .pkl format
- ✅ XGBoost V1 (Stage 1 & 2)
- ✅ XGBoost V2 (Stage 1 & 2)
- ✅ LSTM V1 (Stage 1 & 2)
- ✅ LSTM V2 (Stage 1 & 2)

### Verify All Models
```bash
cd scripts/
python verify_all_model_weights.py
```

**Output:** Comprehensive verification report
- Compares original vs saved models
- Tests on same test dataset
- Verifies predictions match 100%
- Reports test accuracy

---

## Model Architectures

### XGBoost (Hierarchical)

**Stage 1:** 4-class classification (Benign, DOS, Reconnaissance, Theft)
- V1: Default XGBoost parameters, trained on full dataset
- V2: 300 estimators, max_depth=8, trained on balanced dataset

**Stage 2:** Binary classification (DDoS vs DoS)
- V1: Default XGBoost parameters
- V2: 500 estimators, max_depth=8

### LSTM (Hierarchical)

**Stage 1:** 4-class classification
```
Input: (batch, 1, 23)
LSTM(128) → Dropout(0.3) → BatchNormalization
LSTM(64) → Dropout(0.3) → BatchNormalization
Dense(32, relu) → Dropout(0.3)
Dense(4, softmax)
Parameters: 132,676
```

**Stage 2:** Binary classification (DDoS vs DoS)
```
Input: (batch, 1, 23)
LSTM(128) → Dropout(0.3) → BatchNormalization
LSTM(64) → Dropout(0.3) → BatchNormalization
Dense(32, relu) → Dropout(0.3)
Dense(2, softmax)
Parameters: 142,786
```

---

## File Locations

### XGBoost V1
```
models/hierarchical/
├── hierarchical_stage1_model.pkl      # 8.2 MB
├── hierarchical_stage1_model.json     # Original
├── hierarchical_stage1_scaler.npy
├── hierarchical_stage1_encoder.npy
├── hierarchical_stage2_model.pkl      # 7.5 MB
├── hierarchical_stage2_model.json     # Original
├── hierarchical_stage2_scaler.npy
└── hierarchical_stage2_encoder.npy
```

### XGBoost V2
```
models/v2_hierarchical/
├── xgboost_stage1.pkl                 # 4.3 MB
├── scaler_stage1.pkl
├── label_encoder_stage1.pkl
├── xgboost_stage2.pkl                 # 1.4 MB
├── scaler_stage2.pkl
└── label_encoder_stage2.pkl
```

### LSTM V1
```
models/lstm/
├── lstm_hierarchical_stage1_model.pkl # 567 KB
├── lstm_hierarchical_stage1_model.h5  # Original
├── lstm_hierarchical_s1_scaler_mean.npy
├── lstm_hierarchical_s1_scaler_scale.npy
├── lstm_hierarchical_s1_encoder.npy
├── lstm_hierarchical_stage2_model.pkl # 615 KB
├── lstm_hierarchical_stage2_model.h5  # Original
├── lstm_hierarchical_s2_scaler_mean.npy
├── lstm_hierarchical_s2_scaler_scale.npy
└── lstm_hierarchical_s2_encoder.npy
```

### LSTM V2
```
models/v2_lstm/
├── lstm_stage1_model.pkl              # 567 KB
├── lstm_stage1_model.h5               # Original
├── scaler_stage1_mean.npy
├── scaler_stage1_scale.npy
├── label_encoder_stage1.npy
├── lstm_stage2_model.pkl              # 615 KB
├── lstm_stage2_model.h5               # Original
├── scaler_stage2_mean.npy
├── scaler_stage2_scale.npy
└── label_encoder_stage2.npy
```

---

## Key Verification Results

### ✅ XGBoost Models
- **Predictions:** 100% identical between .json and .pkl
- **Probabilities:** Exact match (rtol=1e-10)
- **Feature Importances:** Identical
- **Test Performance:** Verified on consistent test sets

### ✅ LSTM Models
- **Predictions:** 100% identical between .h5 and .pkl
- **Difference:** 0.00e+00 (exact match)
- **Test Performance:** Verified on consistent test sets
- **Architecture:** Preserved with all 132K-143K parameters

---

## Recommendations

1. **For Production:** Use V2 models (99%+ accuracy)
2. **For Development:** Both V1 and V2 models available
3. **Format Preference:** 
   - `.pkl` files are compatible with both joblib and pickle
   - Original formats (.json, .h5) maintained for reference
4. **Backup:** All model files should be backed up regularly

---

## Total Storage

| Model Type | Total Size |
|------------|------------|
| XGBoost V1 | 15.7 MB |
| XGBoost V2 | 5.7 MB |
| LSTM V1 | 1.2 MB |
| LSTM V2 | 1.2 MB |
| **Total** | **23.8 MB** |

---

**Last Updated:** November 6, 2025  
**Verification Status:** ✅ ALL MODELS VERIFIED  
**See Full Report:** MODEL_WEIGHTS_VERIFICATION_REPORT.md

