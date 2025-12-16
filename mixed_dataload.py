"""
Enhanced data loader that loads and mixes both v1 and v2 datasets
for comprehensive attack type coverage
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

seed = 42

not_applicable_features_v1 = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack', 'Label']
not_applicable_features_v2 = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'Attack', 'Label']


def load_mixed_datasets(v1_path, v2_path, v1_ratio=0.5, info=True, test_size=0.2):
    """
    Load and mix v1 and v2 datasets for comprehensive attack coverage
    
    Args:
        v1_path: Path to v1 dataset
        v2_path: Path to v2 dataset  
        v1_ratio: Ratio of v1 samples (0.5 = 50% v1, 50% v2)
        info: Print dataset info
        test_size: Test set size
        
    Returns:
        x_train, y_train, x_test, y_test, attack_train, attack_test, dataset_versions
    """
    # Load v1 dataset
    if info:
        print(f"Loading v1 dataset: {v1_path}")
    df_v1 = pd.read_csv(v1_path, low_memory=True)
    df_v1.dropna(inplace=True)
    df_v1['dataset_version'] = 'v1'
    
    # Load v2 dataset
    if info:
        print(f"Loading v2 dataset: {v2_path}")
    if v2_path.endswith('.gz'):
        df_v2 = pd.read_csv(v2_path, low_memory=True, nrows=len(df_v1))  # Match v1 size
    else:
        df_v2 = pd.read_csv(v2_path, low_memory=True)
    df_v2.dropna(inplace=True)
    df_v2['dataset_version'] = 'v2'
    
    if info:
        print(f"\nV1 Dataset:")
        print(f"  Samples: {len(df_v1)}")
        print(f"  Attack types: {sorted(df_v1['Attack'].unique())}")
        print(f"\nV2 Dataset:")
        print(f"  Samples: {len(df_v2)}")
        print(f"  Attack types: {sorted(df_v2['Attack'].unique())}")
    
    # Sample from each dataset according to ratio
    v1_sample_size = int(len(df_v1) * v1_ratio)
    v2_sample_size = int(len(df_v2) * (1 - v1_ratio))
    
    df_v1_sampled = df_v1.sample(n=min(v1_sample_size, len(df_v1)), random_state=seed)
    df_v2_sampled = df_v2.sample(n=min(v2_sample_size, len(df_v2)), random_state=seed)
    
    # Align features (v2 has more features, so we'll use only common ones for now)
    # Get common features, excluding non-numeric columns
    exclude_columns = ['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 
                      'Attack', 'Label', 'dataset_version']
    
    # Get all numeric columns from both datasets
    v1_numeric = [col for col in df_v1.columns if col not in exclude_columns]
    v2_numeric = [col for col in df_v2.columns if col not in exclude_columns]
    
    # Find common numeric features
    common_features = list(set(v1_numeric) & set(v2_numeric))
    
    # Ensure we only use numeric columns
    common_features = [f for f in common_features if df_v1[f].dtype in ['int64', 'float64']]
    
    if info:
        print(f"\nCommon numeric features: {len(common_features)}")
        print(f"Features: {sorted(common_features)[:10]}...")  # Show first 10
    
    # Keep only common features + Attack, Label, dataset_version
    df_v1_aligned = df_v1_sampled[common_features + ['Attack', 'Label', 'dataset_version']].copy()
    df_v2_aligned = df_v2_sampled[common_features + ['Attack', 'Label', 'dataset_version']].copy()
    
    # Combine datasets
    df_mixed = pd.concat([df_v1_aligned, df_v2_aligned], ignore_index=True)
    
    # Shuffle
    df_mixed = df_mixed.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    if info:
        print(f"\nMixed Dataset:")
        print(f"  Total samples: {len(df_mixed)}")
        print(f"  V1 samples: {len(df_v1_aligned)} ({len(df_v1_aligned)/len(df_mixed)*100:.1f}%)")
        print(f"  V2 samples: {len(df_v2_aligned)} ({len(df_v2_aligned)/len(df_mixed)*100:.1f}%)")
        print(f"  All attack types: {sorted(df_mixed['Attack'].unique())}")
        print(f"  Attack distribution:")
        for attack, count in df_mixed['Attack'].value_counts().items():
            print(f"    {attack}: {count} ({count/len(df_mixed)*100:.1f}%)")
    
    # Prepare features
    X = df_mixed[common_features]
    y = df_mixed['Label']
    attack_labels = df_mixed['Attack']
    dataset_versions = df_mixed['dataset_version']
    
    # Split and scale
    scaler = MinMaxScaler()
    X_train, X_test, y_train, y_test, attack_train, attack_test, version_train, version_test = train_test_split(
        X, y, attack_labels, dataset_versions,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test, attack_train, attack_test, version_test


def load_data_with_labels(cid, info=True, test_size=0.2, full=False):
    """Backward compatibility - loads single dataset"""
    from utils.v1_dataload_enhanced import load_data_with_labels as load_v1
    return load_v1(cid, info, test_size, full)
