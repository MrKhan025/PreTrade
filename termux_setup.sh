#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "===================================================="
echo " PreTrade — Android Termux Setup & Launcher"
echo "===================================================="

# 1. Update Termux packages and install prebuilt ARM64 scientific libraries
pkg update -y
pkg install -y tur-repo
pkg install -y python python-pip python-numpy python-scipy python-pandas git clang libffi openssl freetype libjpeg-turbo zlib

# 2. Install pure-Python dependencies (skipping heavy C/Fortran rebuilds)
pip install --upgrade pip
pip install --no-build-isolation -r requirements-termux.txt || pip install -r requirements-termux.txt

# 3. Configure Finnhub API key
export FINNHUB_API_KEY="dbg5c3hr01qidncm8hpgdbg5c3hr01qidncm8hq0"

echo ""
echo "Setup complete! Starting PreTrade on http://localhost:8501 ..."
streamlit run app.py --server.headless=true --server.address=127.0.0.1 --server.port=8501
