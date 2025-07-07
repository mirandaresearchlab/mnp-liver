import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_filter_data(file_path, metadata_column, percentage_to_keep=50):
    """Load and filter CSV data, keeping specified percentage of '0' entries."""
    df = pd.read_csv(file_path, sep=",", header=0, dtype={metadata_column: 'string'})
    
    print(f"Unique values in '{metadata_column}':")
    print(df[metadata_column].value_counts())
    num_classes = len(df[metadata_column].value_counts())
    print(f"Number of classes: {num_classes}")

    # Filter rows with '0' in metadata_column
    zero_mask = df[metadata_column] == '0'
    num_zero_rows = zero_mask.sum()
    rows_to_keep = int(num_zero_rows * percentage_to_keep / 100)
    
    print(f"Rows with {metadata_column} == '0': {num_zero_rows}")
    print(f"Rows to keep ({percentage_to_keep:.2f}% of zeros): {rows_to_keep}")

    # Sample rows if any '0's exist
    if rows_to_keep > 0:
        zero_indices = df.index[zero_mask]
        zero_indices_to_keep = np.random.default_rng(42).choice(zero_indices, size=rows_to_keep, replace=False)
        keep_mask = ~zero_mask
        keep_mask[zero_indices_to_keep] = True
    else:
        keep_mask = ~zero_mask

    df_filtered = df[keep_mask].reset_index(drop=True)
    print(f"Filtered values in '{metadata_column}':")
    print(df_filtered[metadata_column].value_counts())
    
    del df  # Free memory
    return df_filtered, num_classes

def preprocess_dataframe(df, nan_threshold=0.0001):
    """Preprocess DataFrame by selecting features, handling NaN/Inf, and scaling."""
    feature_columns = [col for col in df.columns if not (col.startswith(('Metadata_', 'Image_')) or col.endswith('_ObjectNumber'))]
    print(f"Number of feature columns: {len(feature_columns)}")
    
    # X = df[feature_columns].astype('float32', copy=False)
    X = df[feature_columns]
    
    # Check for NaN and Inf
    nan_counts = X.isna().sum()
    inf_counts = np.isinf(X).sum()
    threshold = X.shape[0] * nan_threshold
    invalid_columns = set(nan_counts[nan_counts >= threshold].index).union(
        inf_counts[inf_counts >= threshold].index)
    valid_columns = [col for col in feature_columns if col not in invalid_columns]
    
    if invalid_columns:
        print("\nColumns with NaN or Inf:")
        print(f"{'Column':<60} {'NaN Count':>10} {'Inf Count':>10}")
        print("-" * 80)
        for col in sorted(invalid_columns):
            print(f"{col:<60} {nan_counts[col]:>10} {inf_counts[col]:>10}")
        print(f"Total columns with NaN or Inf: {len(invalid_columns)}")
    
    print(f"Number of valid columns: {len(valid_columns)}")
    if not valid_columns:
        raise ValueError("No valid columns remain after filtering.")
    
    X = X[valid_columns]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    
    nan_count_after_fill = X.isna().sum().sum()
    print(f"NaN count after filling: {nan_count_after_fill}")
    if nan_count_after_fill > 0:
        print("Warning: NaN values remain. Filling with zero.")
        X.fillna(0, inplace=True)
    
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("No rows/columns remain after preprocessing.")
    
    scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X).astype('float32')
    X_scaled = scaler.fit_transform(X)
    
    del X  # Free memory
    return X_scaled, valid_columns