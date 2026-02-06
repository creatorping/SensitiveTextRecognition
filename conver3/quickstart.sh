#!/bin/bash

# Quick start script for Nested Privacy Entity Recognition

echo "=========================================="
echo "Nested Privacy Entity Recognition System"
echo "=========================================="

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Analyze dataset
echo ""
echo "Analyzing dataset..."
python utils.py

# Train model
echo ""
echo "Starting training..."
python run.py --mode train

# Evaluate model
echo ""
echo "Evaluating model..."
python run.py --mode eval

# Run benchmark
echo ""
echo "Running speed benchmark..."
python benchmark.py

# Run demo
echo ""
echo "Running demo..."
python demo.py

echo ""
echo "=========================================="
echo "All tasks completed!"
echo "=========================================="
