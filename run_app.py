#!/usr/bin/env python3
"""
Launcher script for the Routing Optimization System UI
"""

import sys
import subprocess
import argparse
import os

def run_basic_app():
    """Run the basic Streamlit app."""
    print("🚀 Starting Basic Routing Optimization System...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

def run_enhanced_app():
    """Run the enhanced Streamlit app with real-time features."""
    print("🚀 Starting Enhanced Routing Optimization System...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "enhanced_app.py"])

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def main():
    parser = argparse.ArgumentParser(description="Routing Optimization System Launcher")
    parser.add_argument(
        "--mode", 
        choices=["basic", "enhanced"], 
        default="enhanced",
        help="Choose which version to run (default: enhanced)"
    )
    parser.add_argument(
        "--install", 
        action="store_true",
        help="Install dependencies before running"
    )
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
    
    print("\n" + "="*60)
    print("🔄 ROUTING OPTIMIZATION SYSTEM")
    print("="*60)
    
    if args.mode == "basic":
        print("📊 Running Basic Version")
        print("Features: Data generation, basic analytics, static charts")
        run_basic_app()
    else:
        print("🚀 Running Enhanced Version")
        print("Features: Real-time data, ML insights, advanced analytics")
        run_enhanced_app()

if __name__ == "__main__":
    main()
