import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TradingEngine:
    """Vereinfachte Trading Engine für Paper Trading."""
    
    def __init__(self, config: dict):
        self.config = config
        self.positions = {}
        
    def analyze_market(self, symbol: str, candles: pd.DataFrame) -> Dict:
        """Einfache Marktanalyse für Paper Trading."""
        try:
            if candles.empty or len(candles) < 5:
                return {'action': 'HOLD', 'confidence': 0.5}
            
            # Einfache Preistrend-Analyse
            recent_close = candles['close'].iloc[-1]
            sma_5 = candles['close'].iloc[-5:].mean()
            
            # Signal basierend auf Preis vs SMA
            if recent_close > sma_5 * 1.005:  # 0.5% über SMA
                return {'action': 'BUY', 'confidence': 0.6}
            elif recent_close < sma_5 * 0.995:  # 0.5% unter SMA
                return {'action': 'SELL', 'confidence': 0.6}
            else:
                return {'action': 'HOLD', 'confidence': 0.5}
                
        except Exception as e:
            logger.error(f"Fehler in analyze_market für {symbol}: {str(e)}")
            return {'action': 'HOLD', 'confidence': 0.5}
    
    def execute_trade(self, symbol: str, action: str, volume: float = 0.1) -> bool:
        """Simuliert Trade-Execution für Paper Trading."""
        try:
            trade_id = f"{symbol}_{action}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.positions[trade_id] = {
                'symbol': symbol,
                'action': action,
                'volume': volume,
                'timestamp': pd.Timestamp.now(),
                'status': 'SIMULATED'
            }
            
            logger.info(f"PAPER TRADE: {action} {symbol} (Volume: {volume})")
            return True
            
        except Exception as e:
            logger.error(f"Fehler in execute_trade: {str(e)}")
            return False