#!/usr/bin/env python3
"""
UrbanSense One-Click Setup Script
Automatically installs dependencies and runs the demo
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    print("🏙️  UrbanSense One-Click Setup")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher required")
        sys.exit(1)

    print("✅ Python version check passed")

    # Install required dependencies
    print("📦 Installing core dependencies...")
    try:
        required_packages = [
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "networkx>=2.6.0",
            "scikit-learn>=1.0.0",
            "scipy>=1.7.0",
            "matplotlib>=3.5.0",
            "faker>=13.0.0",
            "geopy>=2.2.0",
            "tqdm>=4.62.0",
            "colorama>=0.4.4"
        ]

        for package in required_packages:
            print(f"  Installing {package.split('>=')[0]}...")
            install_package(package)

        print("✅ Core dependencies installed")

        # Optional dependencies
        print("\n📊 Installing visualization dependencies...")
        viz_packages = ["plotly>=5.0.0", "dash>=2.0.0"]

        for package in viz_packages:
            try:
                print(f"  Installing {package.split('>=')[0]}...")
                install_package(package)
            except:
                print(f"  ⚠️  Failed to install {package.split('>=')[0]} (optional)")

        print("✅ Setup completed successfully!")

        # Run demo
        print("\n🚀 Running UrbanSense demo...")
        print("=" * 50)

        # Try full demo first, fallback to simplified
        try:
            subprocess.run([sys.executable, "demo.py", "--sensors", "50", "--duration", "0.25"])
        except:
            print("\n⚡ Running simplified demo...")
            subprocess.run([sys.executable, "demo.py", "--no-dashboard", "--sensors", "20"])

    except Exception as e:
        print(f"❌ Setup failed: {e}")
        print("\n🔧 Manual installation:")
        print("pip install numpy pandas networkx scikit-learn scipy matplotlib faker geopy tqdm colorama")
        print("python demo.py --no-dashboard --sensors 20")
        sys.exit(1)

if __name__ == "__main__":
    main()