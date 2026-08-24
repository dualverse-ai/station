#!/bin/bash
# Setup conda environment for batch integration with OpenProblems 2.0 compatible versions

set -e  # Exit on error

# Check if conda is available, if not try to initialize it
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH, attempting to initialize..."

    # Try common conda installation paths
    CONDA_PATHS=(
        "~/miniconda3/etc/profile.d/conda.sh"
        "~/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
        "/usr/local/anaconda3/etc/profile.d/conda.sh"
    )

    CONDA_FOUND=false
    for conda_path in "${CONDA_PATHS[@]}"; do
        # Expand tilde
        expanded_path="${conda_path/#\~/$HOME}"
        if [[ -f "$expanded_path" ]]; then
            echo "Found conda at: $expanded_path"
            source "$expanded_path"
            CONDA_FOUND=true
            break
        fi
    done

    if [[ "$CONDA_FOUND" == false ]]; then
        echo "ERROR: Could not find conda installation. Please ensure conda is installed and accessible."
        echo "Tried the following paths:"
        for conda_path in "${CONDA_PATHS[@]}"; do
            echo "  - $conda_path"
        done
        exit 1
    fi

    # Verify conda is now available
    if ! command -v conda &> /dev/null; then
        echo "ERROR: Conda initialization failed. Please check your conda installation."
        exit 1
    fi
fi

echo "Using conda: $(which conda)"
echo "Conda version: $(conda --version)"

echo "=== Setting up batch_integration conda environment ==="

# Install R and system dependencies
echo "Installing R and system dependencies..."
sudo apt-get update
sudo apt install r-base r-base-dev -y
sudo apt install libfontconfig1-dev libfreetype6-dev libcurl4-openssl-dev -y

# Create new conda environment
echo "Creating conda environment 'batch_integration'..."
conda create -n batch_integration --clone station -y

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate batch_integration

# Install base packages with compatible versions (OpenProblems 2.0 requirements)
echo "Installing NumPy < 2.0 and SciPy <= 1.13..."
pip install "numpy<2.0" "scipy<=1.13"

# Install core scientific packages
echo "Installing core packages..."
pip install pandas matplotlib seaborn

# Install single-cell packages
echo "Installing single-cell packages..."
pip install anndata scanpy

# Install R interface with compatible versions
echo "Installing R interface packages..."
pip install "rpy2>=3.5.2"
pip install "anndata2ri==1.3.1"  # Compatible with numpy<2

# Install scib with specific version used by OpenProblems (compile from source to avoid GLIBC issues)
echo "Installing scib 1.1.7 from source..."
mkdir -p ~/lib
cd ~/lib
git clone https://github.com/theislab/scib.git
cd scib
git checkout v1.1.7
pip install -e .
cd -  # Return to original directory

# Install additional dependencies
echo "Installing additional dependencies..."
pip install scikit-learn scikit-misc numba
pip install umap-learn pynndescent
pip install leidenalg igraph
pip install pydot deprecated
pip install h5py zarr
pip install joblib networkx
pip install statsmodels patsy
# pip install session-info2  # Requires Python 3.10+, skip for Python 3.9

# Install JAX and libraries mentioned in research spec
echo "Installing JAX ecosystem..."
pip install jax jaxlib
pip install flax optax

# Install GPU acceleration for clustering
echo "Installing rapids-singlecell for GPU clustering..."
pip install rapids-singlecell

# Install other libraries mentioned in spec
echo "Installing additional libraries from spec..."
pip install scikit-learn  # already installed but ensure it's there

# Install R packages for kBET
echo "Installing R packages and kBET..."
mkdir -p ~/R/library
cd ~/R/library
wget https://github.com/theislab/kBET/archive/refs/heads/master.zip -O kBET-master.zip
unzip kBET-master.zip
sudo R -e "install.packages(c('FNN', 'RColorBrewer', 'ggplot2'), repos='https://cran.r-project.org')"
sudo R -e "install.packages('kBET-master', repos = NULL, type = 'source')"
cd -  # Return to original directory

# Test the installation
echo "Testing installation..."
python -c "
import numpy as np
import scipy
import anndata
import scanpy
import scib
import anndata2ri
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'AnnData: {anndata.__version__}')
print(f'Scanpy: {scanpy.__version__}')
print(f'anndata2ri: {anndata2ri.__version__}')
print('✓ All packages installed successfully!')
"

# Test R packages
echo "Testing R packages..."
R -e "library(kBET); library(FNN); print('✓ R packages installed successfully!')"

echo "=== Environment setup complete! ==="
echo "To use this environment:"
echo "  conda activate batch_integration"
echo "  python your_script.py"