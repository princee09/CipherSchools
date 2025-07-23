import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing requirements. Please install manually.")

def run_dashboard():
    """Run the Streamlit dashboard"""
    try:
        subprocess.run(["streamlit", "run", "dashboard.py"])
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing requirements first...")
        install_requirements()
        subprocess.run(["streamlit", "run", "dashboard.py"])

if __name__ == "__main__":
    print("🚀 Starting Product Purchase Prediction Dashboard...")
    print("📦 Installing/checking requirements...")
    install_requirements()
    print("🌐 Launching dashboard...")
    run_dashboard()