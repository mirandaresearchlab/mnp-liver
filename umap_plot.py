import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the CSV file
f = "/Users/jen-hung/Desktop/df_SingleCell_AO_HEPG2_110341.csv"
df = pd.read_csv(f, sep=",", header=0)

# Print basic info about the DataFrame
print("Initial DataFrame shape:", df.shape)
print("Columns:", df.columns.tolist())

# Select numerical feature columns (starting with 'Cells_', excluding 'Metadata_' and 'Image_')
feature_columns = [col for col in df.columns if not col.startswith(('Metadata_', 'Image_'))]
print("Selected feature columns:", feature_columns)
print("Number of feature columns:", len(feature_columns))

# Extract features
X = df[feature_columns]

# Check initial row count
print("Initial number of rows in X:", X.shape[0])

# Check for invalid values (NaN, inf)
print("NaN count per column:\n", X.isna().sum())
print("Inf count per column:\n", np.isinf(X).sum())

# Filter out columns with too many NaN or inf values (e.g., >50% missing)
threshold = X.shape[0] * 0.5
valid_columns = [col for col in X.columns if X[col].isna().sum() < threshold and np.isinf(X[col]).sum() < threshold]
print("Valid columns after filtering (>50% valid data):", valid_columns)
print("Number of valid columns:", len(valid_columns))

if not valid_columns:
    raise ValueError("No valid columns remain after filtering. All selected columns have excessive NaN or inf values.")

X = X[valid_columns]

# Replace inf with NaN
X = X.replace([np.inf, -np.inf], np.nan)

# Option 1: Fill NaN with column median (more robust than mean)
X = X.fillna(X.median())

# Check if any NaN values remain
nan_count_after_fill = X.isna().sum().sum()
print("NaN count after filling with median:", nan_count_after_fill)

# If NaN values remain, try filling with zero or drop rows
if nan_count_after_fill > 0:
    print("Warning: Some columns still have NaN values after filling with median. Filling with zero.")
    X = X.fillna(0)
    nan_count_after_fill = X.isna().sum().sum()
    print("NaN count after filling with zero:", nan_count_after_fill)

# Check if X is empty
if X.shape[0] == 0:
    raise ValueError("No rows remain after preprocessing. Check your data for excessive NaN or inf values.")

# Check if X has at least one column
if X.shape[1] == 0:
    raise ValueError("No columns remain after preprocessing. Check your data for valid feature columns.")

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_embedding = umap_model.fit_transform(X_scaled)

# Create a DataFrame for the UMAP results
umap_df = pd.DataFrame(umap_embedding, columns=['UMAP1', 'UMAP2'])

# Add metadata for coloring (e.g., Metadata_Site)
if 'Metadata_Site' in df.columns:
    umap_df['Metadata'] = df['Metadata_concentration_perliter'].astype(str)
elif 'Metadata_QCFlag' in df.columns:
    umap_df['Metadata'] = df['Metadata_QCFlag'].astype(str)
else:
    print("No metadata column found (Metadata_Site or Metadata_QCFlag). Using index as metadata.")
    umap_df['Metadata'] = df.index.astype(str)

# Plot the UMAP
plt.figure(figsize=(10, 8))
sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Metadata', palette='deep', s=50)
plt.title('UMAP of Single-Cell Data')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend(title='Metadata', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('umap_plot.png')
plt.show()