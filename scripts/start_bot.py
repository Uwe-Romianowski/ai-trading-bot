#!/usr/bin/env python3
# Start Script for AI Trading Bot

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("AI TRADING BOT v3.5")
    print("=" * 60)
    
    try:
        from main import main as run_bot
        run_bot()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
