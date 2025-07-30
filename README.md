# 🧬 Single-Cell Morphological Data Analysis

This repository provides a pipeline for downstream analysis of single-cell morphological profiles. It includes steps for **dimensionality reduction** and **unsupervised clustering**, enabling the exploration of phenotypic heterogeneity at the single-cell level.

---

## 📦 1. Installation (via [uv](https://docs.astral.sh/uv/getting-started/installation/))

First, install `uv` globally by following the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Then, from the root directory of this project, set up the local virtual environment and install all dependencies:

```bash
uv sync
```

### Activating the Project Environment

Once dependencies are installed, you can use the environment in one of two ways:

**Option 1: Activate the environment manually**

```bash
source .venv/bin/activate
```

**Option 2: Run Python scripts using `uv`**

```bash
uv run python your_script.py
```

## 🚀 2. Running the Feature Analysis Pipeline

You can explore and execute the analysis using either Jupyter notebook or standalone Python script:

### Option A: Jupyter Notebook

For interactive exploration, open the following notebook:

```bash
./notebooks/singlecell_feature_analysis.ipynb
```

To launch Jupyter Lab:

```bash
uv run jupyter lab
```

### Option B: Python Script

To run the full pipeline non-interactively:

```bash
python ./python/main.py
```