import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import configure_paths, RANDOM_SEED, RANGE_N_CLUSTERS, USE_GMM, GMM_COVARIANCE_TYPES, get_range_n_components


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


def load_and_balance_data(file_path, metadata_column, well_column):
    """Load and balance a single DataFrame based on metadata and well columns.

    Args:
        df (pandas.DataFrame): The input DataFrame to process.
        csv_file (str): The file name corresponding to the DataFrame.
        metadata_column (str): Column name for metadata (e.g., Metadata_concentration_perliter).
        well_column (str): Column name for well identifier (e.g., well_column).

    Returns:
        tuple: (Filtered and balanced DataFrame, Number of unique classes in metadata_column).
    """

    # Load and filter CSV data, keeping specified percentage of '0' entries.
    df = pd.read_csv(file_path, sep=",", header=0, dtype={metadata_column: 'string'})
    print(f"\n--- File: {file_path} ---")

    # Total value counts for metadata_column
    conc_counts = df[metadata_column].value_counts()
    print(f"\nTotal value counts for '{metadata_column}':")
    print(conc_counts)
    num_classes = len(conc_counts)
    print(f"Number of unique {metadata_column} values: {num_classes}")

    # Total value counts for well_column
    well_counts = df[well_column].value_counts()
    print(f"\nTotal value counts for '{well_column}':")
    print(well_counts)
    print(f"Number of unique {well_column} values: {len(well_counts)}")

    # Group by metadata_column and get value counts of well_column
    grouped_counts = df.groupby(metadata_column)[well_column].value_counts()
    print(f"\nValue counts of '{well_column}' under each '{metadata_column}':")
    print(grouped_counts)

    # Print total number of unique wells per concentration
    for conc in df[metadata_column].unique():
        num_wells = len(df[df[metadata_column] == conc][well_column].unique())
        print(f"Number of unique wells for {metadata_column} = {conc}: {num_wells}")

    # Delete grouped_counts to free memory
    del grouped_counts

    # Find the minimum count of rows among all metadata_column values
    min_count = conc_counts.min()
    print(f"\nMinimum row count among {metadata_column} values: {min_count}")

    # Delete conc_counts and well_counts to free memory
    del conc_counts, well_counts

    # Initialize an empty list to store indices to keep
    keep_indices = []

    # Process each concentration value
    for conc in df[metadata_column].unique():
        # Subset for the current concentration
        conc_mask = df[metadata_column] == conc
        conc_df = df[conc_mask]

        # Get the proportion of each well in this concentration
        well_proportions = conc_df[well_column].value_counts(normalize=True)

        # Calculate number of rows to keep for each well to maintain proportions
        rows_to_keep = (well_proportions * min_count).round().astype(int)

        # Ensure we don't exceed available rows for each well
        for well in rows_to_keep.index:
            available_rows = len(conc_df[conc_df[well_column] == well])
            rows_to_keep[well] = min(rows_to_keep[well], available_rows)

        # Randomly sample rows for each well
        for well in rows_to_keep.index:
            well_mask = conc_df[well_column] == well
            well_indices = conc_df[well_mask].index
            if rows_to_keep[well] > 0:
                selected_indices = np.random.default_rng(42).choice(
                    well_indices, size=rows_to_keep[well], replace=False
                )
                keep_indices.extend(selected_indices)

        # Delete intermediate variables to free memory
        del conc_df, well_proportions, rows_to_keep, well_indices

    # Create filtered DataFrame
    df_filtered = df.loc[keep_indices].reset_index(drop=True)

    # Print results after filtering
    print(f"\nValue counts for '{metadata_column}' after balancing to lowest count ({min_count}):")
    print(df_filtered[metadata_column].value_counts())
    print(f"\nValue counts of '{well_column}' under each '{metadata_column}' after balancing:")
    print(df_filtered.groupby(metadata_column)[well_column].value_counts())

    # Delete keep_indices to free memory
    del keep_indices

    return df_filtered, num_classes


def preprocess_dataframe(df, nan_threshold=0.0001):
    """Preprocess DataFrame by selecting features, handling NaN/Inf, and scaling."""
    feature_columns = [col for col in df.columns if not (col.startswith(('Metadata_', 'Image_')) or col.endswith('_ObjectNumber'))]
    print(f"Number of feature columns: {len(feature_columns)}")
    
    # X = df[feature_columns].astype('float32', copy=False)
    X = df[feature_columns].copy()
    
    # Check for NaN and Inf
    nan_counts = X.isna().sum()
    inf_counts = np.isinf(X).sum()

    # Identify columns with NaN or Inf exceeding threshold
    threshold = X.shape[0] * nan_threshold
    invalid_columns = set(nan_counts[nan_counts >= threshold].index).union(
        inf_counts[inf_counts >= threshold].index
    )
    valid_columns = [col for col in feature_columns if col not in invalid_columns]
    
    # Print columns with NaN or Inf if any
    columns_with_nan_or_inf = set(nan_counts[nan_counts > 0].index).union(
        inf_counts[inf_counts > 0].index
    )
    if columns_with_nan_or_inf:
        print("\nColumns with at least one NaN or Inf value:")
        print(f"{'Column':<60} {'NaN Count':>10} {'Inf Count':>10}")
        print("-" * 80)
        for col in sorted(columns_with_nan_or_inf):
            print(f"{col:<60} {nan_counts[col]:>10} {inf_counts[col]:>10}")
        print(f"Total columns with NaN or Inf: {len(columns_with_nan_or_inf)}")
    
    print(f"Number of valid columns: {len(valid_columns)}")
    if not valid_columns:
        raise ValueError("No valid columns remain after filtering.")
    
    # Select valid columns in-place
    # X.drop(columns=[col for col in X.columns if col not in valid_columns], inplace=True)
    X = X[valid_columns]

    # Check NaN and Inf counts after filtering
    nan_counts_after_filter = X.isna().sum().sum()
    inf_counts_after_filter = np.isinf(X).sum().sum()
    print(f"\nNaN count after filtering columns: {nan_counts_after_filter}")
    print(f"Inf count after filtering columns: {inf_counts_after_filter}")
    
    # Replace inf with NaN and fill NaN with median only if necessary
    if nan_counts_after_filter > 0 or inf_counts_after_filter > 0:
        X = X.replace([np.inf, -np.inf], np.nan)
        medians = X.median()
        X = X.fillna(medians)
        
        # Check for remaining NaN values
        nan_count_after_fill = X.isna().sum().sum()
        print(f"NaN count after filling with median: {nan_count_after_fill}")
        if nan_count_after_fill > 0:
            print("Warning: Some NaN values remain. Filling with zero.")
            X = X.fillna(0)
    
    # Check if data is valid
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("No rows/columns remain after preprocessing.")
    
    # Normalize using Median Absolute Deviation (MAD)
    # Calculate median for each column
    medians = X.median()
    # Calculate MAD for each column: median(|x - median(x)|)
    mad = np.abs(X - medians).median()
    # Avoid division by zero by setting small MAD values to 1
    mad = mad.where(mad > 0, 1.0)
    # Normalize: (x - median) / MAD
    X_normalized = (X - medians) / mad
    
    del X  # Free memory
    return X_normalized, valid_columns

def concatenate_dataframes(file_paths, metadata_column, well_column):
    # Initialize list to store processed data for each file
    preprocessed_data = {}
    preprocessed_results = []

    # Loop through each file path
    for file_path in file_paths:
        # Load and filter data
        df_filtered, num_classes = load_and_balance_data(file_path, metadata_column, well_column)

        # Preprocess data
        X_normalized, valid_columns = preprocess_dataframe(df_filtered)
        preprocessed_results.append({'X_normalized': X_normalized, 'valid_columns': valid_columns, 'df': df_filtered})
        print(f"Preprocessed {file_path} with {len(valid_columns)} valid columns.\n")

        del df_filtered
        del X_normalized  # Free memory
        del valid_columns  # Free memory

    # Concatenate preprocessed data
    concatenated_df = pd.concat([res['df'] for res in preprocessed_results], ignore_index=True)
    # Find common valid columns across all DataFrames
    common_valid_columns = list(set.intersection(*[set(res['valid_columns']) for res in preprocessed_results]))
    # Concatenate X_normalized arrays, selecting only common valid columns
    if common_valid_columns:
        # Ensure X_normalized is a numpy array and get indices for common valid columns
        concatenated_X_normalized = np.concatenate([
            np.array(res['X_normalized'])[:, [res['valid_columns'].index(col) for col in common_valid_columns]]
            for res in preprocessed_results
        ], axis=0)
    else:
        raise ValueError("No common valid columns found across DataFrames.")
    preprocessed_data['concatenated_df'] = {
        'X_normalized': concatenated_X_normalized,
        'valid_columns': common_valid_columns,
        'df': concatenated_df
    }

    num_classes = len(concatenated_df[metadata_column].value_counts())
    print(f"\nConcatenated DataFrame shape: {concatenated_df.shape}")
    print(f"Concatenated X_normalized shape: {concatenated_X_normalized.shape}")
    print(f"Number of common valid columns: {len(common_valid_columns)}\n")

    # Delete intermediate variables to free memory
    del preprocessed_results
    del concatenated_df
    del common_valid_columns

    return preprocessed_data, num_classes
