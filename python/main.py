import numpy as np
import sys
import datetime
from pathlib import Path
from config import configure_paths, RANDOM_SEED, PERCENTAGE_TO_KEEP, RANGE_N_CLUSTERS, USE_GMM, GMM_COVARIANCE_TYPES, get_range_n_components
from preprocessing import load_and_filter_data, preprocess_dataframe
from visualization import plot_dimensionality_reduction
from clustering import perform_clustering_analysis

# Set random seed
np.random.seed(RANDOM_SEED)

def setup_logging(save_dir):
    """Set up logging to write print statements to a single log file with dynamic timestamp."""
    date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f"log_{date_time_str}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    log_handle = open(log_file, 'a', encoding='utf-8')  # Use 'a' for append mode
    sys.stdout = log_handle
    return log_handle

def main():
    """Main function to execute the analysis pipeline."""
    # Load configuration
    save_dir, file_path, metadata_column = configure_paths()
    
    # Initialize logging
    log_handle = setup_logging(save_dir)
    
    # Load and filter data
    df_filtered, num_classes = load_and_filter_data(file_path, metadata_column, PERCENTAGE_TO_KEEP)
    
    # Get RANGE_N_COMPONENTS based on num_classes
    RANGE_N_COMPONENTS = get_range_n_components(num_classes)
    print(f"RANGE_N_COMPONENTS set to: {RANGE_N_COMPONENTS}")
    
    # Preprocess data
    X_scaled, valid_columns = preprocess_dataframe(df_filtered)
    preprocessed_data = {'df': {'X_scaled': X_scaled, 'valid_columns': valid_columns, 'df': df_filtered}}
    
    # Dimensionality reduction with LDA
    method = 'LDA'
    for name, data in preprocessed_data.items():
        if metadata_column in data['df'].columns:
            # 2D categorical plot
            plot_dimensionality_reduction(
                data['X_scaled'], data['df'], data['valid_columns'], metadata_column,
                method, f"{method} of {file_path.name}", continuous=False, n_components=2,
                save_path=str(save_dir / f"{file_path.name}_{method}")
            )
            # 3D categorical plot
            plot_dimensionality_reduction(
                data['X_scaled'], data['df'], data['valid_columns'], metadata_column,
                method, f"{method} of {file_path.name}", continuous=False, n_components=3,
                save_path=str(save_dir / f"{file_path.name}_{method}")
            )
        else:
            print(f"Warning: {metadata_column} not found in {name}.")
    
    # Clustering analysis
    for name, data in preprocessed_data.items():
        if metadata_column in data['df'].columns:
            perform_clustering_analysis(
                data['X_scaled'], data['df'], metadata_column, file_path.name, save_dir,
                RANGE_N_CLUSTERS, RANGE_N_COMPONENTS, USE_GMM, GMM_COVARIANCE_TYPES
            )
        else:
            print(f"Warning: {metadata_column} not found in {name}.")
    
    # Free memory
    del df_filtered, X_scaled, preprocessed_data
    
    # Close log file
    if log_handle:
        log_handle.close()
        sys.stdout = sys.__stdout__  # Restore original stdout

if __name__ == "__main__":
    main()