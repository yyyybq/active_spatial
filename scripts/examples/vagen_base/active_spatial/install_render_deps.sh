#!/bin/bash
# Install dependencies for local Gaussian Splatting rendering

set -e

echo "==================================="
echo "Installing Rendering Dependencies"
echo "==================================="

# Install gsplat
echo "Installing gsplat..."
pip install gsplat

# Install plyfile (required by ply_gaussian_loader)
echo "Installing plyfile..."
pip install plyfile

# Copy ply_gaussian_loader from ViewSuite
echo "Copying ply_gaussian_loader..."
VIEWSUITE_PATH="/scratch/by2593/project/Active_Spatial/ViewSuite"
if [ -f "$VIEWSUITE_PATH/ply_gaussian_loader.py" ]; then
    cp "$VIEWSUITE_PATH/ply_gaussian_loader.py" \
       "$(python -c 'import site; print(site.getsitepackages()[0])')/ply_gaussian_loader.py"
    echo "✓ ply_gaussian_loader copied"
else
    echo "✗ ply_gaussian_loader.py not found in ViewSuite"
    echo "  Please check the path: $VIEWSUITE_PATH"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import gsplat; print('✓ gsplat installed:', gsplat.__version__)"
python -c "import ply_gaussian_loader; print('✓ ply_gaussian_loader installed')"

echo ""
echo "==================================="
echo "Installation Complete! ✓"
echo "==================================="
