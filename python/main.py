import numpy as np
import sys
import datetime
from pathlib import Path
from config import configure_paths, RANDOM_SEED, RANGE_N_CLUSTERS, USE_GMM, GMM_COVARIANCE_TYPES, get_range_n_components
from preprocessing import concatenate_dataframes, load_and_balance_data, preprocess_dataframe
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
    save_dir, file_paths, metadata_column, well_column, cell_type, dim_reduction = configure_paths()

    # Initialize logging
    log_handle = setup_logging(save_dir)

    # Preprcess and concatenate dataframes
    preprocessed_data, num_classes = concatenate_dataframes(file_paths, metadata_column, well_column)

    # Get RANGE_N_COMPONENTS based on num_classes
    RANGE_N_COMPONENTS = get_range_n_components(num_classes)
    print(f"RANGE_N_COMPONENTS: {RANGE_N_COMPONENTS}")
    
    # Dimensionality reduction
    for name, data in preprocessed_data.items():
        if metadata_column in data['df'].columns:
            # 2D categorical plot
            plot_dimensionality_reduction(
                data['X_normalized'], data['df'], data['valid_columns'], metadata_column,
                dim_reduction, f"{dim_reduction} of {cell_type}", continuous=False, n_components=2,
                save_path=str(save_dir / f"{cell_type}_{dim_reduction}")
            )
            # 3D categorical plot
            plot_dimensionality_reduction(
                data['X_normalized'], data['df'], data['valid_columns'], metadata_column,
                dim_reduction, f"{dim_reduction} of {cell_type}", continuous=False, n_components=3,
                save_path=str(save_dir / f"{cell_type}_{dim_reduction}")
            )
        else:
            print(f"Warning: {metadata_column} not found in {name}.")
    
    # Clustering analysis
    for name, data in preprocessed_data.items():
        if metadata_column in data['df'].columns:
            perform_clustering_analysis(
                data['X_normalized'], data['df'], metadata_column, cell_type, save_dir,
                RANGE_N_CLUSTERS, RANGE_N_COMPONENTS, USE_GMM, GMM_COVARIANCE_TYPES
            )
        else:
            print(f"Warning: {metadata_column} not found in {name}.")
    
    # Free memory
    del preprocessed_data
    
    # Close log file
    if log_handle:
        log_handle.close()
        sys.stdout = sys.__stdout__  # Restore original stdout

if __name__ == "__main__":
    main()