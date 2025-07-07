import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap.umap_ as umap
from utils import convert_concentration
from config import setup_logging

def plot_dimensionality_reduction(X_scaled, df, valid_columns, metadata_column, method_name, title, 
                                 tsne_perplexity=30, n_neighbors=15, min_dist=0.1, continuous=False, 
                                 n_components=2, save_path=None, save_dir=None):
    """Plot dimensionality reduction results using Plotly."""
    if save_dir:
        setup_logging(save_dir)
    
    X_scaled = X_scaled.astype('float32', copy=False)
    
    if method_name == 'PCA':
        reducer = PCA(n_components=n_components)
        x_label, y_label = 'PC1', 'PC2'
        z_label = 'PC3' if n_components == 3 else None
    elif method_name == 't-SNE':
        reducer = TSNE(n_components=n_components, perplexity=tsne_perplexity, learning_rate='auto', random_state=42)
        x_label, y_label = 't-SNE 1', 't-SNE 2'
        z_label = 't-SNE 3' if n_components == 3 else None
    elif method_name == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
        x_label, y_label = 'UMAP 1', 'UMAP 2'
        z_label = 'UMAP 3' if n_components == 3 else None
    elif method_name == 'LDA':
        reducer = LDA(n_components=n_components)
        x_label, y_label = 'LD1', 'LD2'
        z_label = 'LD3' if n_components == 3 else None
    else:
        raise ValueError(f"Unsupported method: {method_name}")

    try:
        X_reduced = (reducer.fit_transform(X_scaled, df[metadata_column]) if method_name == 'LDA' 
                     else reducer.fit_transform(X_scaled)).astype('float32')
    except Exception as e:
        print(f"Error in {method_name} reduction: {e}")
        return None

    is_3d = n_components == 3
    fig = go.Figure()
    title_suffix = ' (Continuous)' if continuous else ' (Categorical)'
    
    if continuous:
        concentrations = df[metadata_column].apply(convert_concentration).astype('float32')
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2], mode='markers',
                marker=dict(color=concentrations, colorscale='Viridis', showscale=True, 
                            colorbar=dict(title=f'{metadata_column} (g)'))
            ))
        else:
            fig.add_trace(go.Scatter(
                x=X_reduced[:, 0], y=X_reduced[:, 1], mode='markers',
                marker=dict(color=concentrations, colorscale='Viridis', showscale=True, 
                            colorbar=dict(title=f'{metadata_column} (g)'))
            ))
        del concentrations
    else:
        labels = df[metadata_column].astype('string', copy=False)
        unique_labels = sorted(labels.unique(), key=convert_concentration, reverse=True)
        color_map = {label: f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' 
                     for label, (r, g, b) in zip(unique_labels, sns.color_palette('tab10', len(unique_labels)))}
        
        for label in unique_labels:
            mask = labels == label
            if is_3d:
                fig.add_trace(go.Scatter3d(
                    x=X_reduced[mask, 0], y=X_reduced[mask, 1], z=X_reduced[mask, 2], 
                    mode='markers', marker=dict(color=color_map[label], size=3),
                    name=str(label), showlegend=True
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=X_reduced[mask, 0], y=X_reduced[mask, 1], 
                    mode='markers', marker=dict(color=color_map[label], size=3),
                    name=str(label), showlegend=True
                ))
        del labels
    
    layout = dict(
        title=f"{title}{title_suffix} ({'3D' if is_3d else '2D'}, {len(valid_columns)} features)",
        **({'scene': dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label)} if is_3d 
           else {'xaxis_title': x_label, 'yaxis_title': y_label})
    )
    fig.update_layout(**layout, showlegend=True, legend=dict(itemsizing='constant'))

    if save_path:
        fig.write_image(f"{save_path}_{'3D' if is_3d else '2D'}.png", width=1200, height=800)
    
    del X_reduced
    return fig