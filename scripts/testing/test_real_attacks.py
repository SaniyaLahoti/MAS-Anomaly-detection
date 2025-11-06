"""
Test the multi-agent system with REAL attack samples from the dataset
to verify it can actually detect attacks (not just Benign)
"""

import pandas as pd
import requests
import json

print("=" * 80)
print("TESTING MULTI-AGENT SYSTEM WITH REAL ATTACK SAMPLES")
print("=" * 80)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('v1_dataset/NF-BoT-IoT.csv')
print(f"Dataset loaded: {len(df)} total samples")

print("\nAttack distribution:")
for attack, count in df['Attack'].value_counts().items():
    print(f"  {attack}: {count}")

# Get one sample from each attack type
print("\n" + "=" * 80)
print("EXTRACTING REAL SAMPLES FROM EACH ATTACK TYPE")
print("=" * 80)

attack_samples = {}
for attack in ['Benign', 'DDoS', 'DoS', 'Reconnaissance', 'Theft']:
    df_attack = df[df['Attack'] == attack]
    if len(df_attack) > 0:
        # Get a sample from the middle of the dataset
        idx = len(df_attack) // 2
        sample = df_attack.iloc[idx]
        attack_samples[attack] = {
            "protocol": int(sample['PROTOCOL']),
            "l7_proto": float(sample['L7_PROTO']),
            "l4_src_port": int(sample['L4_SRC_PORT']),
            "l4_dst_port": int(sample['L4_DST_PORT']),
            "in_bytes": int(sample['IN_BYTES']),
            "in_pkts": int(sample['IN_PKTS']),
            "out_bytes": int(sample['OUT_BYTES']),
            "out_pkts": int(sample['OUT_PKTS']),
            "tcp_flags": int(sample['TCP_FLAGS']),
            "flow_duration_ms": int(sample['FLOW_DURATION_MILLISECONDS'])
        }
        print(f"\n{attack} sample extracted:")
        print(f"  PROTOCOL={sample['PROTOCOL']}, L7_PROTO={sample['L7_PROTO']}")
        print(f"  Ports: SRC={sample['L4_SRC_PORT']}, DST={sample['L4_DST_PORT']}")
        print(f"  IN: {sample['IN_BYTES']} bytes, {sample['IN_PKTS']} pkts")
        print(f"  OUT: {sample['OUT_BYTES']} bytes, {sample['OUT_PKTS']} pkts")
        print(f"  TCP_FLAGS={sample['TCP_FLAGS']}, DURATION={sample['FLOW_DURATION_MILLISECONDS']}ms")

# Test each sample against the API
print("\n" + "=" * 80)
print("TESTING SAMPLES AGAINST MULTI-AGENT API")
print("=" * 80)

api_url = "http://localhost:8000/predict"
results_summary = []

for attack_type, payload in attack_samples.items():
    print(f"\n{'-' * 80}")
    print(f"TESTING: {attack_type}")
    print(f"{'-' * 80}")
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        result = response.json()
        
        if result['success']:
            predicted = result['prediction']
            confidence = result['confidence']
            agreement = result['agreement']
            xgb_pred = result['xgboost_prediction']
            lstm_pred = result['lstm_prediction']
            
            correct = predicted == attack_type
            status = "✅ CORRECT" if correct else f"❌ WRONG (expected {attack_type})"
            
            print(f"  Prediction: {predicted} ({confidence*100:.1f}%) - {status}")
            print(f"  XGBoost:    {xgb_pred} ({result['xgboost_confidence']*100:.1f}%)")
            print(f"  LSTM:       {lstm_pred} ({result['lstm_confidence']*100:.1f}%)")
            print(f"  Agreement:  {agreement}")
            
            results_summary.append({
                'actual': attack_type,
                'predicted': predicted,
                'correct': correct,
                'confidence': confidence
            })
        else:
            print(f"  ❌ ERROR: {result.get('error', 'Unknown error')}")
            results_summary.append({
                'actual': attack_type,
                'predicted': 'ERROR',
                'correct': False,
                'confidence': 0.0
            })
    except Exception as e:
        print(f"  ❌ EXCEPTION: {str(e)}")
        results_summary.append({
            'actual': attack_type,
            'predicted': 'EXCEPTION',
            'correct': False,
            'confidence': 0.0
        })

# Final Summary
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

total = len(results_summary)
correct = sum(1 for r in results_summary if r['correct'])
accuracy = correct / total * 100 if total > 0 else 0

print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")

print("\nPer-Class Results:")
for result in results_summary:
    status = "✅" if result['correct'] else "❌"
    print(f"  {status} {result['actual']:15} -> {result['predicted']:15} ({result['confidence']*100:.1f}%)")

print("\n" + "=" * 80)

if correct < total:
    print("⚠️  WARNING: System is not detecting all attacks correctly!")
    print("   This needs to be investigated and fixed.")
else:
    print("✅ SUCCESS: All attacks detected correctly!")

print("=" * 80)

# Save results
with open('real_attack_test_results.json', 'w') as f:
    json.dump({
        'summary': {
            'total': total,
            'correct': correct,
            'accuracy': accuracy
        },
        'details': results_summary,
        'samples_tested': attack_samples
    }, f, indent=2)

print("\nResults saved to: real_attack_test_results.json")

