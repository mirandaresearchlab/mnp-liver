from pathlib import Path

def configure_paths():
    """Configure paths for local or Docker environment."""
    save_dir = Path("/home/jen-hungwang/Documents/mnp-liver/results/")
    csv_dir = Path("/home/jen-hungwang/Documents/mnp-liver/csv")
    # save_dir = Path("/storage/jenhung/results/mnp/")  # Uncomment for Docker
    # csv_dir = Path("/storage/jenhung/data/mnp_liver")  # Uncomment for Docker
    cell_type = "HepG2"  # or "HUH7"
    csv_path = csv_dir / cell_type  # hep or huh
    csv_data = [
        "df_SingleCell_AO_HEPG2_102912.csv",
        "df_SingleCell_AO_HEPG2_110341.csv",  
        "df_SingleCell_AO_HEPG2_231222.csv"
    ]
    metadata_column = "Metadata_concentration_perliter"  # "Metadata_concentration_perliter_x"
    well_column = "Metadata_Well"
    dim_reduction = "LDA"  # or "PCA", "UMAP", etc.
    return save_dir, [csv_path / file for file in csv_data], metadata_column, well_column, cell_type, dim_reduction

def get_range_n_components(num_classes):
    """Generate RANGE_N_COMPONENTS based on num_classes."""
    # Ensure max components does not exceed num_classes - 1 (LDA constraint)
    max_components = num_classes - 1 if num_classes > 1 else 1
    # Generate range from 2 to max_components (or [2] if max_components < 2)
    return list(range(2, max_components + 1)) if max_components >= 2 else [2]
    # return [max_components]

# Global settings
RANDOM_SEED = 42
# PERCENTAGE_TO_KEEP = 50
RANGE_N_CLUSTERS = [3, 4]
USE_GMM = True
GMM_COVARIANCE_TYPES = ['full'] # ['full', 'tied', 'diag', 'spherical']  # Uncomment for more covariance types