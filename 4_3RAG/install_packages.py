#!/usr/bin/env python3
"""
Install script for RAG system dependencies
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🔧 RAG System Package Installer")
    print("=" * 40)
    
    packages = [
        "transformers",
        "torch", 
        "sentence-transformers",
        "faiss-cpu",
        "langchain",
        "numpy",
        "beautifulsoup4",
        "requests"
    ]
    
    print("📦 Installing required packages...")
    
    failed_packages = []
    for package in packages:
        print(f"   Installing {package}...")
        if install_package(package):
            print(f"   ✅ {package} installed successfully")
        else:
            print(f"   ❌ Failed to install {package}")
            failed_packages.append(package)
    
    print("\n" + "=" * 40)
    if not failed_packages:
        print("🎉 All packages installed successfully!")
        print("\n🚀 Now you can run:")
        print("   python improved_test.py")
        print("   python RAG_app.py")
    else:
        print(f"❌ Failed to install: {', '.join(failed_packages)}")
        print("\n💡 Try running manually:")
        for package in failed_packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()
