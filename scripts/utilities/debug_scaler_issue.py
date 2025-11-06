"""
Debug the scaler/preprocessing issue that's causing all predictions to be Benign
"""

import numpy as np
import pandas as pd
import pickle

print("=" * 80)
print("DEBUGGING SCALER ISSUE")
print("=" * 80)

# Load the scaler
print("\nLoading Stage 1 scaler...")
scaler = np.load('hierarchical_stage1_scaler.npy', allow_pickle=True).item()

print(f"Scaler type: {type(scaler)}")
print(f"Scaler class: {scaler.__class__.__name__}")

# Check if it has feature names
if hasattr(scaler, 'feature_names_in_'):
    print(f"\n✅ Scaler HAS feature_names_in_:")
    print(f"   Number of features: {len(scaler.feature_names_in_)}")
    print(f"   Feature names:")
    for i, name in enumerate(scaler.feature_names_in_, 1):
        print(f"      {i:2d}. {name}")
    
    print(f"\n⚠️  THIS IS THE PROBLEM!")
    print(f"   The scaler expects these EXACT feature names in this EXACT order.")
    print(f"   When we pass a numpy array, it fails feature name validation.")
else:
    print(f"\n❌ Scaler does NOT have feature_names_in_")

# Check scaler mean and scale
if hasattr(scaler, 'mean_'):
    print(f"\nScaler statistics (first 5 features):")
    print(f"  Mean: {scaler.mean_[:5]}")
    print(f"  Scale: {scaler.scale_[:5]}")

# Test with a sample
print(f"\n{'=' * 80}")
print("TESTING SCALER WITH SAMPLE DATA")
print(f"{'=' * 80}")

# Create a test sample (Theft attack from line 3)
test_data = {
    'PROTOCOL': 6,
    'L7_PROTO': 0.0,
    'L4_SRC_PORT': 49160,
    'L4_DST_PORT': 4444,
    'IN_BYTES': 217753000,
    'IN_PKTS': 4521,
    'OUT_BYTES': 199100,
    'OUT_PKTS': 4049,
    'TCP_FLAGS': 24,
    'FLOW_DURATION_MILLISECONDS': 4176249
}

# Engineer features
test_data['PACKET_RATE'] = (test_data['IN_PKTS'] + test_data['OUT_PKTS']) / (test_data['FLOW_DURATION_MILLISECONDS'] + 1)
test_data['BYTE_RATE'] = (test_data['IN_BYTES'] + test_data['OUT_BYTES']) / (test_data['FLOW_DURATION_MILLISECONDS'] + 1)
test_data['AVG_PACKET_SIZE'] = (test_data['IN_BYTES'] + test_data['OUT_BYTES']) / (test_data['IN_PKTS'] + test_data['OUT_PKTS'] + 1)
test_data['AVG_IN_PACKET_SIZE'] = test_data['IN_BYTES'] / (test_data['IN_PKTS'] + 1)
test_data['AVG_OUT_PACKET_SIZE'] = test_data['OUT_BYTES'] / (test_data['OUT_PKTS'] + 1)
test_data['BYTE_ASYMMETRY'] = abs(test_data['IN_BYTES'] - test_data['OUT_BYTES']) / (test_data['IN_BYTES'] + test_data['OUT_BYTES'] + 1)
test_data['PACKET_ASYMMETRY'] = abs(test_data['IN_PKTS'] - test_data['OUT_PKTS']) / (test_data['IN_PKTS'] + test_data['OUT_PKTS'] + 1)
test_data['IN_OUT_BYTE_RATIO'] = test_data['IN_BYTES'] / (test_data['OUT_BYTES'] + 1)
test_data['IN_OUT_PACKET_RATIO'] = test_data['IN_PKTS'] / (test_data['OUT_PKTS'] + 1)
test_data['PROTOCOL_INTENSITY'] = test_data['PROTOCOL'] * test_data['PACKET_RATE']
test_data['TCP_PACKET_INTERACTION'] = test_data['TCP_FLAGS'] * test_data['IN_PKTS']
test_data['PROTOCOL_PORT_COMBO'] = test_data['PROTOCOL'] * test_data['L4_DST_PORT']
test_data['FLOW_INTENSITY'] = (test_data['IN_PKTS'] + test_data['OUT_PKTS']) / (test_data['FLOW_DURATION_MILLISECONDS'] + 1) * test_data['AVG_PACKET_SIZE']

print("\nOriginal features:")
for k, v in list(test_data.items())[:10]:
    print(f"  {k}: {v}")

# Convert to DataFrame
df_test = pd.DataFrame([test_data])

print(f"\nDataFrame shape: {df_test.shape}")
print(f"DataFrame columns ({len(df_test.columns)}):")
for i, col in enumerate(df_test.columns, 1):
    print(f"  {i:2d}. {col}")

# Try to transform with DataFrame
print(f"\n{'=' * 80}")
print("ATTEMPTING TO TRANSFORM WITH DATAFRAME:")
print(f"{'=' * 80}")
try:
    scaled_df = scaler.transform(df_test)
    print(f"✅ SUCCESS! Scaled shape: {scaled_df.shape}")
    print(f"   First 5 scaled values: {scaled_df[0,:5]}")
except Exception as e:
    print(f"❌ FAILED: {str(e)}")

# Try to transform with numpy array
print(f"\n{'=' * 80}")
print("ATTEMPTING TO TRANSFORM WITH NUMPY ARRAY:")
print(f"{'=' * 80}")
try:
    scaled_np = scaler.transform(df_test.values)
    print(f"✅ SUCCESS! Scaled shape: {scaled_np.shape}")
    print(f"   First 5 scaled values: {scaled_np[0,:5]}")
except Exception as e:
    print(f"❌ FAILED: {str(e)}")

print("\n" + "=" * 80)

