from pathlib import Path
import sys
import datetime
import os

def configure_paths():
    """Configure paths for local or Docker environment."""
    save_dir = Path("/home/jen-hungwang/Documents/mnp-liver/results/")
    csv_dir = Path("/home/jen-hungwang/Documents/mnp-liver/csv")
    # save_dir = Path("/storage/jenhung/results/mnp/")  # Uncomment for Docker
    # csv_dir = Path("/storage/jenhung/data/mnp_liver")  # Uncomment for Docker
    csv_path = csv_dir / "hep" # hep or huh
    csv_data = "df_SingleCell_AO_HEPG2_110341.csv" # "df_HUH7_SingleCell_102912.csv", "df_HUH7_SingleCell_110341.csv", "df_HUH7_SingleCell_191735.csv"
    metadata_column = "Metadata_concentration_perliter" # "Metadata_concentration_perliter_x"
    return save_dir, csv_path / csv_data, metadata_column

def setup_logging(save_dir):
    """Set up logging to write print statements to a log file with dynamic timestamp."""
    date_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f"log_{date_time_str}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    log_handle = open(log_file, 'w', encoding='utf-8')
    sys.stdout = log_handle
    return log_handle

def get_range_n_components(num_classes):
    """Generate RANGE_N_COMPONENTS based on num_classes."""
    # Ensure max components does not exceed num_classes - 1 (LDA constraint)
    max_components = num_classes - 1 if num_classes > 1 else 1
    # Generate range from 2 to max_components (or [2] if max_components < 2)
    return list(range(2, max_components + 1)) if max_components >= 2 else [2]

# Global settings
RANDOM_SEED = 42
PERCENTAGE_TO_KEEP = 50
RANGE_N_CLUSTERS = [2, 3]
USE_GMM = True
GMM_COVARIANCE_TYPES = ['full', 'tied'] # ['full', 'tied', 'diag', 'spherical']  # Uncomment for more covariance types