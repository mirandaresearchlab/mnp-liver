import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from utils import convert_concentration
from config import setup_logging
import os

def perform_clustering_analysis(X_scaled, df, metadata_column, csv_data, save_dir, 
                               range_n_clusters=[2, 4, 5], range_n_components=[4], 
                               use_gmm=True, gmm_covariance_types=['full', 'tied']):
    """Perform clustering analysis with silhouette plots and concentration distribution."""
    if save_dir:
        setup_logging(save_dir)
    
    X_scaled = X_scaled.astype('float32', copy=False)
    df[metadata_column] = df[metadata_column].apply(lambda x: x.lower().strip())

    for n_comp in range_n_components:
        try:
            reducer = LDA(n_components=n_comp)
            X_reduced = reducer.fit_transform(X_scaled, df[metadata_column]).astype('float32')
        except ValueError as e:
            print(f"Error with LDA n_components={n_comp}: {e}")
            continue

        covariance_iter = gmm_covariance_types if use_gmm else [None]
        for gmm_covariance in covariance_iter:
            for n_clusters in range_n_clusters:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7), gridspec_kw={'width_ratios': [1, 1, 1.5]})
                
                try:
                    if use_gmm:
                        clusterer = GaussianMixture(n_components=n_clusters, random_state=10, covariance_type=gmm_covariance)
                        algo_name = f"GMM_{gmm_covariance}"
                    else:
                        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
                        algo_name = "KMeans"

                    cluster_labels = clusterer.fit_predict(X_reduced)
                    centers = clusterer.means_ if use_gmm else clusterer.cluster_centers_
                    centers = centers.astype('float32')
                except Exception as e:
                    print(f"Error in clustering for n_components={n_comp}, n_clusters={n_clusters}, {algo_name}: {e}")
                    plt.close(fig)
                    continue

                silhouette_avg = silhouette_score(X_reduced, cluster_labels)
                print(f"For n_components={n_comp}, n_clusters={n_clusters}, {algo_name}, silhouette_score={silhouette_avg:.4f}")

                # Silhouette Plot
                ax1.set_xlim([-0.1, 1])
                ax1.set_ylim([0, len(X_reduced) + (n_clusters + 1) * 10])
                sample_silhouette_values = silhouette_samples(X_reduced, cluster_labels)
                y_lower = 10
                for i in range(n_clusters):
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
                    ith_cluster_silhouette_values.sort()
                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    color = cm.nipy_spectral(float(i) / n_clusters)
                    ax1.fill_betweenx(
                        np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7
                    )
                    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
                    y_lower = y_upper + 10

                ax1.set_title("Silhouette Plot")
                ax1.set_xlabel("Silhouette Coefficient Values")
                ax1.set_ylabel("Cluster Label")
                ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
                ax1.set_yticks([])
                ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

                # Cluster Scatter Plot
                colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
                ax2.scatter(X_reduced[:, 0], X_reduced[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
                ax2.scatter(centers[:, 0], centers[:, 1], marker="o", c="white", alpha=1, s=200, edgecolor="k")
                for i, c in enumerate(centers):
                    ax2.scatter(c[0], c[1], marker=f"${i}$", alpha=1, s=50, edgecolor="k")
                
                ax2.set_title(f"Cluster Visualization ({algo_name})")
                ax2.set_xlabel("Feature Space (1st Dimension)")
                ax2.set_ylabel("Feature Space (2nd Dimension)")

                # Concentration Distribution Plot
                unique_clusters = np.unique(cluster_labels[cluster_labels != -1])
                concentration_values = df[metadata_column].unique()
                converted_values = np.array([convert_concentration(val) for val in concentration_values])
                sort_indices = np.argsort(converted_values)[::-1]
                sorted_concentration_values = concentration_values[sort_indices]

                for idx, cluster in enumerate(unique_clusters):
                    cluster_mask = cluster_labels == cluster
                    cluster_concentrations = df[metadata_column][cluster_mask]
                    counts = np.zeros(len(sorted_concentration_values), dtype='int32')
                    for i, conc in enumerate(sorted_concentration_values):
                        counts[i] = np.sum(cluster_concentrations == conc)
                    
                    ax3.bar(
                        np.arange(len(sorted_concentration_values)) + idx * 0.2,
                        counts, width=0.2, label=f'Cluster {cluster}',
                        color=cm.nipy_spectral(float(cluster) / n_clusters), alpha=0.7
                    )

                ax3.set_title("Concentration Distribution Across Clusters")
                ax3.set_xlabel("Concentration Levels (High to Low)")
                ax3.set_ylabel("Count")
                ax3.set_xticks(np.arange(len(sorted_concentration_values)))
                ax3.set_xticklabels(sorted_concentration_values, rotation=45, ha='right')
                ax3.legend()

                plt.suptitle(
                    f"{csv_data} - Analysis with {algo_name}, Silhouette={silhouette_avg:.4f}, "
                    f"n_clusters={n_clusters}, LDA={n_comp}D",
                    fontsize=14, fontweight="bold"
                )

                filename = f"{algo_name}_{n_clusters}clusters_{n_comp}D_{csv_data}.png"
                plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
                plt.close(fig)
                del fig

        del X_reduced