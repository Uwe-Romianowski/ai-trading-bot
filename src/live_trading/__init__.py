# src/live_trading/__init__.py
"""
Live Trading Module - Phase E Final mit SL/TP
"""

from .mt5_client import MT5LiveClient, quick_test as mt5_quick_test
from .order_executor import MT5OrderExecutor
from .live_bridge import LiveTradingBridge, test_ml_integration

__all__ = [
    'MT5LiveClient',
    'MT5OrderExecutor', 
    'LiveTradingBridge',
    'test_ml_integration',
    'mt5_quick_test'
]

__version__ = '1.2.0'
__author__ = 'AI Trading Bot Team'
__description__ = 'ML Live Trading mit SL/TP Monitoring'