#!/usr/bin/env python3
"""
Setup script for Redtape Business Analysis
Installs all required packages
"""

import subprocess
import sys

def install_packages():
    """Install required packages for the analysis"""
    
    packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'plotly',
        'openpyxl',  # For Excel file reading
        'xlrd',      # Additional Excel support
        'scipy',     # For statistical analysis
        'scikit-learn'  # For advanced analytics
    ]
    
    print("📦 Installing required packages for business analysis...")
    print("="*50)
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")
    
    print("\n✅ Package installation completed!")
    print("\n🚀 You can now run the analysis scripts:")
    print("   1. python explore_data_structure.py")
    print("   2. python comprehensive_analysis.py")

if __name__ == "__main__":
    install_packages()
