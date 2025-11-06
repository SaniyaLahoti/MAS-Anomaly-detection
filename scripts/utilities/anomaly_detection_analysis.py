import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample
import xgboost as xgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Load and analyze the dataset
def load_and_analyze_dataset(file_path):
    """
    Load dataset and perform comprehensive analysis
    """
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Basic info
    print("\n" + "=" * 40)
    print("BASIC DATASET INFO")
    print("=" * 40)
    print(df.info())
    
    # Check for missing values
    print("\n" + "=" * 40)
    print("MISSING VALUES")
    print("=" * 40)
    missing_values = df.isnull().sum()
    missing_percent = (missing_values / len(df)) * 100
    missing_df = pd.DataFrame({
        'Column': missing_values.index,
        'Missing_Count': missing_values.values,
        'Missing_Percentage': missing_percent.values
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    # Target variable analysis
    print("\n" + "=" * 40)
    print("TARGET VARIABLE ANALYSIS")
    print("=" * 40)
    
    # Label distribution
    label_counts = df['Label'].value_counts()
    print("Label Distribution:")
    print(label_counts)
    print(f"Label percentages:")
    print(df['Label'].value_counts(normalize=True) * 100)
    
    # Attack type distribution
    attack_counts = df['Attack'].value_counts()
    print(f"\nAttack Type Distribution:")
    print(attack_counts)
    print(f"Attack type percentages:")
    print(df['Attack'].value_counts(normalize=True) * 100)
    
    # Feature analysis
    print("\n" + "=" * 40)
    print("FEATURE ANALYSIS")
    print("=" * 40)
    
    # Identify feature types
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variables from features
    if 'Label' in numeric_features:
        numeric_features.remove('Label')
    if 'Attack' in categorical_features:
        categorical_features.remove('Attack')
    
    print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Statistical summary
    print("\n" + "=" * 40)
    print("STATISTICAL SUMMARY")
    print("=" * 40)
    print(df[numeric_features].describe())
    
    return df, numeric_features, categorical_features

def preprocess_dataset(df, numeric_features, categorical_features):
    """
    Clean and preprocess the dataset
    """
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Handle missing values
    print("Handling missing values...")
    
    # For numeric features, fill with median
    for feature in numeric_features:
        if df_processed[feature].isnull().sum() > 0:
            median_val = df_processed[feature].median()
            df_processed[feature].fillna(median_val, inplace=True)
            print(f"  - Filled {feature} missing values with median: {median_val}")
    
    # For categorical features, fill with mode
    for feature in categorical_features:
        if df_processed[feature].isnull().sum() > 0:
            mode_val = df_processed[feature].mode()[0]
            df_processed[feature].fillna(mode_val, inplace=True)
            print(f"  - Filled {feature} missing values with mode: {mode_val}")
    
    # Handle IP addresses and ports (convert to numeric if needed)
    ip_port_features = ['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_SRC_PORT', 'L4_DST_PORT']
    features_to_drop = []
    
    for feature in ip_port_features:
        if feature in df_processed.columns:
            # For IP addresses, we'll drop them as they're not suitable for ML
            if 'ADDR' in feature:
                features_to_drop.append(feature)
                print(f"  - Will drop {feature} (IP addresses not suitable for ML)")
            # For ports, keep as numeric
            else:
                print(f"  - Keeping {feature} as numeric feature")
    
    # Drop IP address columns
    df_processed = df_processed.drop(columns=features_to_drop)
    
    # Update feature lists
    numeric_features = [f for f in numeric_features if f not in features_to_drop]
    categorical_features = [f for f in categorical_features if f not in features_to_drop]
    
    # Check for infinite values
    print("\nChecking for infinite values...")
    for feature in numeric_features:
        inf_count = np.isinf(df_processed[feature]).sum()
        if inf_count > 0:
            print(f"  - Found {inf_count} infinite values in {feature}")
            # Replace infinite values with max finite value
            max_finite = df_processed[feature][np.isfinite(df_processed[feature])].max()
            df_processed[feature] = df_processed[feature].replace([np.inf, -np.inf], max_finite)
            print(f"    Replaced with max finite value: {max_finite}")
    
    # Handle outliers (optional - using IQR method)
    print("\nHandling extreme outliers...")
    for feature in numeric_features:
        Q1 = df_processed[feature].quantile(0.25)
        Q3 = df_processed[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR  # Using 3*IQR for extreme outliers
        upper_bound = Q3 + 3 * IQR
        
        outliers = ((df_processed[feature] < lower_bound) | (df_processed[feature] > upper_bound)).sum()
        if outliers > 0:
            print(f"  - {feature}: {outliers} extreme outliers detected")
            # Cap outliers instead of removing them
            df_processed[feature] = df_processed[feature].clip(lower=lower_bound, upper=upper_bound)
    
    print(f"\nPreprocessed dataset shape: {df_processed.shape}")
    
    return df_processed, numeric_features, categorical_features

def handle_class_imbalance(df, target_column='Label'):
    """
    Handle class imbalance using resampling techniques
    """
    print("\n" + "=" * 60)
    print("HANDLING CLASS IMBALANCE")
    print("=" * 60)
    
    # Check current class distribution
    class_counts = df[target_column].value_counts()
    print("Current class distribution:")
    print(class_counts)
    print(f"Imbalance ratio: {class_counts.max() / class_counts.min():.2f}")
    
    # Separate majority and minority classes
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    df_majority = df[df[target_column] == majority_class]
    df_minority = df[df[target_column] == minority_class]
    
    print(f"Majority class ({majority_class}): {len(df_majority)} samples")
    print(f"Minority class ({minority_class}): {len(df_minority)} samples")
    
    # Calculate target size for balanced dataset
    # Use a balanced approach - not full upsampling to avoid overfitting
    target_size = int((len(df_majority) + len(df_minority)) * 0.4)  # 40% of total for each class
    
    print(f"Target size for each class: {target_size}")
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority, 
                                     replace=False,
                                     n_samples=target_size,
                                     random_state=42)
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                   replace=True,
                                   n_samples=target_size,
                                   random_state=42)
    
    # Combine the balanced dataset
    df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"\nBalanced dataset shape: {df_balanced.shape}")
    print("New class distribution:")
    print(df_balanced[target_column].value_counts())
    
    return df_balanced

def prepare_features_and_target(df, numeric_features, categorical_features):
    """
    Prepare features and target variables for ML
    """
    print("\n" + "=" * 60)
    print("FEATURE PREPARATION")
    print("=" * 60)
    
    # Prepare feature matrix X
    X = df[numeric_features + categorical_features].copy()
    
    # Handle categorical variables if any remain
    if categorical_features:
        print("Encoding categorical variables...")
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature].astype(str))
            print(f"  - Encoded {feature}")
    
    # Prepare target variable y
    y = df['Label'].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y

def split_and_scale_data(X, y, test_size=0.15, random_state=42):
    """
    Split data into train/test and apply scaling
    """
    print("\n" + "=" * 60)
    print("DATA SPLITTING AND SCALING")
    print("=" * 60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training class distribution:")
    print(y_train.value_counts())
    print(f"Test class distribution:")
    print(y_test.value_counts())
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_xgboost_model():
    """
    Create and configure XGBoost model for anomaly detection
    """
    print("\n" + "=" * 60)
    print("XGBOOST MODEL CONFIGURATION")
    print("=" * 60)
    
    # Create XGBoost classifier with optimized parameters for anomaly detection
    xgb_model = xgb.XGBClassifier(
        # Core parameters
        objective='binary:logistic',  # Binary classification
        eval_metric='auc',           # Area under ROC curve
        
        # Tree parameters
        max_depth=6,                 # Maximum depth of trees
        min_child_weight=1,          # Minimum sum of instance weight in child
        subsample=0.8,              # Subsample ratio of training instances
        colsample_bytree=0.8,       # Subsample ratio of columns when constructing each tree
        
        # Regularization
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        
        # Learning parameters
        learning_rate=0.1,          # Step size shrinkage
        n_estimators=100,           # Number of boosting rounds
        
        # Other parameters
        random_state=42,            # For reproducibility
        n_jobs=-1,                  # Use all available cores
        verbosity=1,                # Print messages
        use_label_encoder=False     # Avoid deprecation warning
    )
    
    print("XGBoost model configured with parameters:")
    print(f"  - Objective: {xgb_model.objective}")
    print(f"  - Eval metric: {xgb_model.eval_metric}")
    print(f"  - Max depth: {xgb_model.max_depth}")
    print(f"  - Learning rate: {xgb_model.learning_rate}")
    print(f"  - N estimators: {xgb_model.n_estimators}")
    print(f"  - Subsample: {xgb_model.subsample}")
    print(f"  - Colsample bytree: {xgb_model.colsample_bytree}")
    print(f"  - Regularization (L1/L2): {xgb_model.reg_alpha}/{xgb_model.reg_lambda}")
    
    return xgb_model

def main():
    """
    Main function to run the complete preprocessing pipeline
    """
    # File path
    file_path = "/Users/saniyalahoti/Downloads/MAS-LSTM-1/v1_dataset/NF-BoT-IoT.csv"
    
    try:
        # Step 1: Load and analyze dataset
        df, numeric_features, categorical_features = load_and_analyze_dataset(file_path)
        
        # Step 2: Preprocess dataset
        df_processed, numeric_features, categorical_features = preprocess_dataset(
            df, numeric_features, categorical_features
        )
        
        # Step 3: Handle class imbalance
        df_balanced = handle_class_imbalance(df_processed)
        
        # Step 4: Prepare features and target
        X, y = prepare_features_and_target(df_balanced, numeric_features, categorical_features)
        
        # Step 5: Split and scale data
        X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
        
        # Step 6: Create XGBoost model
        xgb_model = create_xgboost_model()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        print("✅ Dataset loaded and analyzed")
        print("✅ Missing values handled")
        print("✅ Class imbalance addressed")
        print("✅ Features prepared and scaled")
        print("✅ Train/test split completed")
        print("✅ XGBoost model configured")
        print("\nReady for training! (Training not started as requested)")
        
        # Save preprocessed data for future use
        print("\nSaving preprocessed data...")
        np.save('X_train.npy', X_train)
        np.save('X_test.npy', X_test)
        np.save('y_train.npy', y_train)
        np.save('y_test.npy', y_test)
        print("✅ Preprocessed data saved as .npy files")
        
        return {
            'model': xgb_model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'feature_names': list(X.columns)
        }
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
