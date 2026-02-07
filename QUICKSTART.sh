#!/usr/bin/env bash
# Quick Start Guide for LocalML-Commuter

echo "LocalML-Commuter - Quick Start"
echo "=============================="
echo ""

# 1. Check Python
echo "[1] Checking Python..."
python --version

# 2. Install dependencies
echo "[2] Installing dependencies..."
pip install -r requirements.txt

# 3. Get API key
echo "[3] Set up OpenWeatherMap API"
echo "   Visit: https://openweathermap.org/api"
echo "   Sign up and copy your API key"
echo "   Then run: export OPENWEATHER_API_KEY='your_key_here'"
echo ""

# 4. View available commands
echo "[4] Available commands:"
python -m commute.cli --help
echo ""

echo "Quick Examples:"
echo "==============="
echo ""
echo "1. Get commute estimates (no API key needed for fallback):"
echo "   python -m commute.cli estimate"
echo ""
echo "2. Get a prediction from the trained model:"
echo "   export OPENWEATHER_API_KEY='your_key'"
echo "   python -m commute.cli predict"
echo ""
echo "3. Log a commute with your choice:"
echo "   python -m commute.cli fetch --choice=uber --annoyance=3"
echo ""
echo "4. Retrain the model with new data:"
echo "   python -m commute.cli train"
echo ""
echo "All set! Read README.md for detailed documentation."
