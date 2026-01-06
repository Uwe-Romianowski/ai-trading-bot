"""
Paper Trading Module - Enhanced Version
"""

from .ml_integration import MLTradingEngine, train_ml_model
from .portfolio import Portfolio

# Enhanced ML Engine
try:
    from .enhanced_ml_engine import EnhancedMLTradingEngine, run_enhanced_ml_trading
    ENHANCED_ML_AVAILABLE = True
except ImportError:
    ENHANCED_ML_AVAILABLE = False
    EnhancedMLTradingEngine = None
    run_enhanced_ml_trading = None

# Paper Bridge (falls existiert)
try:
    from .paper_bridge import PaperTradingBridge, run_paper_trading_session
    PAPER_BRIDGE_AVAILABLE = True
except ImportError:
    PAPER_BRIDGE_AVAILABLE = False
    PaperTradingBridge = None
    run_paper_trading_session = None

__all__ = [
    'MLTradingEngine',
    'train_ml_model',
    'Portfolio'
]

if ENHANCED_ML_AVAILABLE:
    __all__.extend(['EnhancedMLTradingEngine', 'run_enhanced_ml_trading'])

if PAPER_BRIDGE_AVAILABLE:
    __all__.extend(['PaperTradingBridge', 'run_paper_trading_session'])

__version__ = '3.0.0'
__author__ = 'AI Trading Bot Team'
__description__ = 'Enhanced Paper Trading mit echtem ML'