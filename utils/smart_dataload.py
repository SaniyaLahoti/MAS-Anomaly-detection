"""
Smart Dataset Sampler - Prioritizes Unique Attacks
Takes ALL rare/unique attacks and samples common ones
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

seed = 42

def create_smart_combined_dataset(v1_path, v2_path, target_size=50000, info=True):
    """
    Create a smart combined dataset that prioritizes unique attacks
    
    Strategy:
    - Take ALL samples of rare attacks (Ransomware, Injection, XSS, MITM, Password, Backdoor)
    - Sample common attacks proportionally (DDoS, DoS, Scanning, Benign)
    - Keep total size around target_size for fast loading
    """
    
    if info:
        print(f"Loading datasets for smart sampling...")
    
    # Load both datasets with row limit to avoid loading 16M rows
    df_v1 = pd.read_csv(v1_path, low_memory=True, nrows=min(target_size*4, 200000))
    df_v2 = pd.read_csv(v2_path, low_memory=True, nrows=min(target_size*4, 200000))
    
    df_v1.dropna(inplace=True)
    df_v2.dropna(inplace=True)
    
    if info:
        print(f"\nV1: {len(df_v1):,} rows, Attacks: {sorted(df_v1['Attack'].unique())}")
        print(f"V2: {len(df_v2):,} rows, Attacks: {sorted(df_v2['Attack'].unique())}")
    
    # Define attack categories
    rare_attacks = ['ransomware', 'injection', 'xss', 'mitm', 'password', 'backdoor']
    common_attacks = ['ddos', 'dos', 'scanning']
    
    # Combine datasets
    df_combined = pd.concat([df_v1, df_v2], ignore_index=True)
    
    # Get all unique attacks
    all_attacks = df_combined['Attack'].str.lower().unique()
    
    if info:
        print(f"\nAll attacks found: {sorted(all_attacks)}")
    
    # Strategy: Take samples from each attack type
    samples_per_type = {}
    
    # For rare attacks: take ALL available (up to 5000 each)
    for attack in rare_attacks:
        df_attack = df_combined[df_combined['Attack'].str.lower() == attack]
        if len(df_attack) > 0:
            samples_per_type[attack] = df_attack.sample(n=min(len(df_attack), 5000), random_state=seed)
    
    # For common attacks: take 2000 each for variety
    for attack in common_attacks:
        df_attack = df_combined[df_combined['Attack'].str.lower() == attack]
        if len(df_attack) > 0:
            samples_per_type[attack] = df_attack.sample(n=min(len(df_attack), 2000), random_state=seed)
    
    # For Benign: take 5000
    df_benign = df_combined[df_combined['Attack'].str.lower() == 'benign']
    if len(df_benign) > 0:
        samples_per_type['benign'] = df_benign.sample(n=min(len(df_benign), 5000), random_state=seed)
    
    # Combine all samples
    df_final = pd.concat(list(samples_per_type.values()), ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle
    
    # If we exceeded target_size, sample down
    if len(df_final) > target_size:
        df_final = df_final.sample(n=target_size, random_state=seed)
    
    if info:
        print(f"\n✅ Smart Combined Dataset:")
        print(f"   Total: {len(df_final):,} samples")
        print(f"   Attack distribution:")
        for attack, count in df_final['Attack'].value_counts().items():
            print(f"     {attack}: {count:,} ({count/len(df_final)*100:.1f}%)")
    
    # Remove non-numeric columns
    exclude_columns = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack', 'Label']
    
    # Import ALL canonical features to support both V1 and V2 agents
    from utils.feature_adapter import CanonicalFeatureBuilder
    all_features = CanonicalFeatureBuilder.ALL_CANONICAL_FEATURES
    
    if info:
        print(f"\n📋 Using ALL canonical features ({len(all_features)} features) to support all agents:")
        # print(f"   {all_features}")
    
    # Check which features are available in the dataset
    available_features = [f for f in all_features if f in df_final.columns]
    missing_features = [f for f in all_features if f not in df_final.columns]
    
    if missing_features:
        if info:
            print(f"\n⚠️  Missing features (will be filled with 0): {len(missing_features)} features")
            # print(f"   {missing_features}")
        # Add missing features as zeros
        for feat in missing_features:
            df_final[feat] = 0
    
    # Extract ALL canonical features in the correct order
    X = df_final[all_features]
    y = df_final['Label']
    attack_labels = df_final['Attack']
    
    if info:
        print(f"\n✅ Feature extraction:")
        print(f"   Selected features: {len(all_features)}")
        print(f"   Available in data: {len(available_features)}")
        print(f"   Filled with zeros: {len(missing_features)}")
    
    # Split and scale
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
        X, y, attack_labels,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    if info:
        print(f"\n✅ Train: {len(X_train):,}, Test: {len(X_test):,}")
        print(f"   Features: {X_train.shape[1]}")
    
    return X_train, y_train, X_test, y_test, attack_train, attack_test


def load_data_with_labels(cid=None, info=True, test_size=0.2, full=False):
    """Backward compatibility wrapper"""
    if cid is None or 'smart' in cid.lower():
        # Use smart combined dataset
        return create_smart_combined_dataset(
            v1_path='./v1_datasets/NF-ToN-IoT.csv',
            v2_path='./v2_datasets/NF-ToN-IoT-v2.csv.gz',
            target_size=50000,
            info=info
        )
    else:
        # Use original single dataset loader
        from utils.v1_dataload_enhanced import load_data_with_labels as original_loader
        return original_loader(cid, info, test_size, full)
