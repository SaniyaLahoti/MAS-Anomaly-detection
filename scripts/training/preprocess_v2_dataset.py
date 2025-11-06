"""
Preprocess V2 Dataset (NF-BoT-IoT-v2.csv.gz)
- Extract 14 columns matching V1 format
- Fix column order (IN_PKTS and OUT_BYTES are swapped in V2)
- Handle class imbalance using hybrid sampling
- Save preprocessed data for training
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
import gzip
import warnings
warnings.filterwarnings('ignore')

def load_v2_dataset_chunked(file_path, sample_size=500000):
    """
    Load V2 dataset from compressed file using chunked reading
    This is memory-efficient for very large datasets (37M rows)
    
    Strategy: Read in chunks, perform stratified sampling on each chunk
    """
    print("=" * 80)
    print("LOADING V2 DATASET (CHUNKED)")
    print("=" * 80)
    
    print(f"\n📂 Loading from: {file_path}")
    print("⏳ Reading in chunks to manage memory...")
    
    # V1 column order (what we need)
    v1_columns = [
        'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
        'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
        'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'Label', 'Attack'
    ]
    
    chunk_size = 1000000  # 1M rows per chunk
    sampled_chunks = []
    total_rows_processed = 0
    chunk_count = 0
    
    # First pass: count rows per attack type
    print("\n🔍 First pass: Analyzing dataset structure...")
    attack_counts = {}
    
    with gzip.open(file_path, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=chunk_size, usecols=v1_columns):
            for attack in chunk['Attack'].unique():
                attack_counts[attack] = attack_counts.get(attack, 0) + len(chunk[chunk['Attack'] == attack])
            total_rows_processed += len(chunk)
            chunk_count += 1
            print(f"   Processed chunk {chunk_count}: {total_rows_processed:,} rows total")
    
    print(f"\n✅ Dataset structure:")
    print(f"   Total rows: {total_rows_processed:,}")
    print(f"   Attack types: {list(attack_counts.keys())}")
    
    # Calculate sampling rate to get approximately sample_size rows
    sampling_rate = min(1.0, sample_size / total_rows_processed)
    print(f"\n📊 Sampling rate: {sampling_rate:.4f} ({sampling_rate*100:.2f}%)")
    
    # Second pass: sample data
    print("\n🔍 Second pass: Sampling data...")
    chunk_count = 0
    
    with gzip.open(file_path, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=chunk_size, usecols=v1_columns):
            # Stratified sampling within chunk
            if sampling_rate < 1.0:
                chunk_sampled = chunk.sample(frac=sampling_rate, random_state=42)
            else:
                chunk_sampled = chunk
            
            sampled_chunks.append(chunk_sampled)
            chunk_count += 1
            print(f"   Sampled chunk {chunk_count}: {len(chunk_sampled):,} rows")
            
            # Early stop if we have enough data
            if sum(len(c) for c in sampled_chunks) >= sample_size * 1.2:
                print(f"   ✅ Collected sufficient data, stopping early")
                break
    
    # Combine chunks
    print(f"\n🔗 Combining {len(sampled_chunks)} chunks...")
    df = pd.concat(sampled_chunks, ignore_index=True)
    
    # Final sampling if we exceeded target
    if len(df) > sample_size:
        print(f"   Downsampling from {len(df):,} to {sample_size:,}...")
        df = df.sample(n=sample_size, random_state=42)
    
    print(f"\n✅ Dataset loaded and sampled: {df.shape}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    
    return df

def extract_v1_format_columns(df):
    """
    Extract only the 14 columns that match V1 format
    CRITICAL: V2 has IN_PKTS and OUT_BYTES in different order than V1!
    
    V1 order: ...,IN_BYTES,OUT_BYTES,IN_PKTS,OUT_PKTS,...
    V2 order: ...,IN_BYTES,IN_PKTS,OUT_BYTES,OUT_PKTS,...
    
    We need to reorder to match V1!
    """
    print("\n" + "=" * 80)
    print("EXTRACTING V1-FORMAT COLUMNS (14 columns)")
    print("=" * 80)
    
    # V1 column order (what we need)
    v1_columns = [
        'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT',
        'PROTOCOL', 'L7_PROTO', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
        'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'Label', 'Attack'
    ]
    
    print("\n🔍 Checking V2 column availability...")
    for col in v1_columns:
        if col in df.columns:
            print(f"   ✅ {col}")
        else:
            print(f"   ❌ {col} - MISSING!")
    
    # Extract columns in V1 order
    df_v1_format = df[v1_columns].copy()
    
    print(f"\n✅ Extracted {len(df_v1_format.columns)} columns")
    print(f"   Shape: {df_v1_format.shape}")
    
    return df_v1_format

def analyze_class_distribution(df):
    """
    Analyze attack type distribution
    """
    print("\n" + "=" * 80)
    print("CLASS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    attack_counts = df['Attack'].value_counts()
    total = len(df)
    
    print("\n📊 Attack Type Distribution:")
    print("-" * 80)
    for attack, count in attack_counts.items():
        percentage = (count / total) * 100
        print(f"   {attack:20s}: {count:>12,} ({percentage:>6.2f}%)")
    
    print("-" * 80)
    print(f"   {'TOTAL':20s}: {total:>12,} (100.00%)")
    
    return attack_counts

def stratified_sampling(df, sample_size=500000, random_state=42):
    """
    Stratified sampling to reduce dataset size while maintaining class proportions
    Uses stratification to ensure fair representation
    """
    print("\n" + "=" * 80)
    print(f"STRATIFIED SAMPLING (Target: {sample_size:,} rows)")
    print("=" * 80)
    
    # Calculate samples per class based on current proportions
    attack_counts = df['Attack'].value_counts()
    total = len(df)
    
    sampled_dfs = []
    
    print("\n📊 Sampling strategy:")
    print("-" * 80)
    
    for attack, count in attack_counts.items():
        proportion = count / total
        target_samples = int(sample_size * proportion)
        
        # If class has fewer samples than target, take all
        actual_samples = min(target_samples, count)
        
        df_class = df[df['Attack'] == attack]
        df_sampled = df_class.sample(n=actual_samples, random_state=random_state)
        sampled_dfs.append(df_sampled)
        
        print(f"   {attack:20s}: {count:>12,} → {actual_samples:>8,} ({actual_samples/count*100:>5.1f}%)")
    
    # Combine all sampled data
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # Shuffle
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print("-" * 80)
    print(f"   {'TOTAL':20s}: {total:>12,} → {len(df_sampled):>8,}")
    print(f"\n✅ Stratified sampling complete: {df_sampled.shape}")
    
    return df_sampled

def hybrid_balancing(df, target_per_class=25000, random_state=42):
    """
    Hybrid balancing: Oversample minority classes, undersample majority classes
    Similar to V1 approach
    """
    print("\n" + "=" * 80)
    print(f"HYBRID BALANCING (Target: {target_per_class:,} per class)")
    print("=" * 80)
    
    balanced_dfs = []
    attack_types = df['Attack'].unique()
    
    print("\n🔄 Balancing each class:")
    print("-" * 80)
    
    for attack in attack_types:
        df_class = df[df['Attack'] == attack]
        class_count = len(df_class)
        
        if class_count > target_per_class:
            # Undersample (too many samples)
            df_resampled = resample(
                df_class,
                replace=False,
                n_samples=target_per_class,
                random_state=random_state
            )
            action = "UNDERSAMPLED"
        else:
            # Oversample (too few samples)
            df_resampled = resample(
                df_class,
                replace=True,
                n_samples=target_per_class,
                random_state=random_state
            )
            action = "OVERSAMPLED"
        
        balanced_dfs.append(df_resampled)
        print(f"   {attack:20s}: {class_count:>12,} → {target_per_class:>8,} ({action})")
    
    # Combine and shuffle
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print("-" * 80)
    print(f"   {'TOTAL':20s}: {len(df):>12,} → {len(df_balanced):>8,}")
    print(f"\n✅ Hybrid balancing complete: {df_balanced.shape}")
    
    return df_balanced

def clean_data(df):
    """
    Clean data: handle missing values, infinities, etc.
    """
    print("\n" + "=" * 80)
    print("DATA CLEANING")
    print("=" * 80)
    
    initial_rows = len(df)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️  Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count} ({count/len(df)*100:.2f}%)")
        
        # Drop rows with missing values in critical columns
        df = df.dropna()
        print(f"\n   Dropped {initial_rows - len(df)} rows with missing values")
    else:
        print("\n✅ No missing values detected")
    
    # Replace infinities
    df = df.replace([np.inf, -np.inf], 0)
    
    print(f"\n✅ Data cleaning complete")
    print(f"   Final shape: {df.shape}")
    
    return df

def save_preprocessed_data(df, output_path):
    """
    Save preprocessed data
    """
    print("\n" + "=" * 80)
    print("SAVING PREPROCESSED DATA")
    print("=" * 80)
    
    df.to_csv(output_path, index=False)
    
    # Get file size
    import os
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    
    print(f"\n✅ Data saved to: {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   File size: {file_size:.2f} MB")

def main():
    """
    Main preprocessing pipeline for V2 dataset
    """
    print("\n" + "=" * 80)
    print("V2 DATASET PREPROCESSING PIPELINE")
    print("=" * 80)
    print("\nObjective: Extract V1-format columns, balance classes, prepare for training")
    print("=" * 80 + "\n")
    
    # Paths
    input_file = "../../datasets/v2_dataset/NF-BoT-IoT-v2.csv.gz"
    output_file = "../../datasets/v2_dataset/NF-BoT-IoT-v2-preprocessed.csv"
    
    # Step 1: Load V2 dataset with chunked reading and sampling (37M → 500K)
    # This extracts V1-format columns and samples data in a memory-efficient way
    df_sampled = load_v2_dataset_chunked(input_file, sample_size=500000)
    
    # Step 2: Analyze class distribution after sampling
    print("\n" + "=" * 80)
    print("DISTRIBUTION AFTER STRATIFIED SAMPLING")
    print("=" * 80)
    attack_counts = analyze_class_distribution(df_sampled)
    
    # Step 3: Hybrid balancing (similar to V1 approach)
    df_balanced = hybrid_balancing(df_sampled, target_per_class=25000)
    
    # Step 4: Clean data
    df_clean = clean_data(df_balanced)
    
    # Step 5: Final distribution check
    print("\n" + "=" * 80)
    print("FINAL DISTRIBUTION (BALANCED & CLEANED)")
    print("=" * 80)
    analyze_class_distribution(df_clean)
    
    # Step 6: Save preprocessed data
    save_preprocessed_data(df_clean, output_file)
    
    print("\n" + "=" * 80)
    print("✅ V2 PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Preprocessed data ready at: {output_file}")
    print(f"   Use this file for training hierarchical models")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

