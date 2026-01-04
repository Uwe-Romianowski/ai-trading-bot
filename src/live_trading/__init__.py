# src/live_trading/__init__.py
"""
Live Trading Module - Phase E
"""

from .mt5_client import MT5LiveClient, quick_test as mt5_quick_test
from .order_executor import MT5OrderExecutor
from .live_bridge import LiveTradingBridge, simple_test as bridge_simple_test

__all__ = [
    'MT5LiveClient',
    'MT5OrderExecutor', 
    'LiveTradingBridge',
    'mt5_quick_test',
    'bridge_simple_test'
]

__version__ = '1.0.0'