#!/usr/bin/env python3
"""
Performance Tracker - Vereinfachte Version
"""

import logging
import numpy as np
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Trackt und analysiert Trading Performance"""
    
    def __init__(self):
        """Initialisiert den Performance Tracker"""
        self.logger = logging.getLogger(__name__)
        
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.performance_score = 0.0
        
        self.logger.info("Performance Tracker initialisiert")
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Zeichnet einen Trade auf.
        
        Args:
            trade_data: Dictionary mit Trade-Informationen
                - profit: Gewinn/Verlust
                - confidence: ML-Confidence (optional)
                - symbol: Symbol (optional)
                - action: BUY/SELL (optional)
        """
        try:
            # Extrahiere Profit
            profit = 0.0
            if isinstance(trade_data, dict):
                profit = float(trade_data.get('profit', 0.0))
            elif isinstance(trade_data, (int, float)):
                profit = float(trade_data)
            
            # Speichere Trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'profit': profit,
                'data': trade_data if isinstance(trade_data, dict) else {}
            }
            
            self.trades.append(trade_record)
            self.total_trades += 1
            
            if profit > 0:
                self.winning_trades += 1
            elif profit < 0:
                self.losing_trades += 1
            
            self.total_profit += profit
            
            # Aktualisiere Statistiken
            if self.total_trades > 0:
                self.win_rate = self.winning_trades / self.total_trades
            
            # Performance Score
            self.performance_score = self._calculate_performance_score()
            
            self.logger.debug(
                f"Trade aufgezeichnet: Profit={profit:.2f}, "
                f"Total Trades={self.total_trades}, "
                f"Win Rate={self.win_rate:.1%}"
            )
            
        except Exception as e:
            self.logger.error(f"Fehler beim Aufzeichnen des Trades: {e}")
    
    def _calculate_performance_score(self) -> float:
        """Berechnet Performance Score"""
        if self.total_trades < 3:
            return 0.0
        
        try:
            score = self.win_rate * 70  # 70% Gewichtung auf Win Rate
            
            # 30% Gewichtung auf Profitabilität
            if self.total_trades > 0:
                avg_profit = self.total_profit / self.total_trades
                score += avg_profit * 3
            
            return max(0.0, min(100.0, score))
        except:
            return 0.0
    
    def get_performance_metrics(self, window: int = None) -> Dict[str, Any]:
        """Gibt Performance-Metriken zurück"""
        try:
            if not self.trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'total_profit': 0.0,
                    'average_profit': 0.0,
                    'sharpe_ratio': 0.0,
                    'performance_score': self.performance_score
                }
            
            if window is None or window > len(self.trades):
                trades_to_analyze = self.trades
            else:
                trades_to_analyze = self.trades[-window:]
            
            profits = [t['profit'] for t in trades_to_analyze]
            total_trades = len(profits)
            winning = sum(1 for p in profits if p > 0)
            
            win_rate = winning / total_trades if total_trades > 0 else 0.0
            total_profit = sum(profits)
            average_profit = total_profit / total_trades if total_trades > 0 else 0.0
            
            # Vereinfachter Sharpe Ratio
            if len(profits) > 1:
                returns = np.array(profits)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-10)
            else:
                sharpe = 0.0
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'average_profit': average_profit,
                'sharpe_ratio': float(sharpe),
                'performance_score': self.performance_score
            }
            
        except Exception as e:
            self.logger.error(f"Fehler in get_performance_metrics: {e}")
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'average_profit': 0.0,
                'sharpe_ratio': 0.0,
                'performance_score': 0.0
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt Status zurück"""
        return {
            'total_trades': self.total_trades,
            'win_rate': self.win_rate,
            'performance_score': self.performance_score,
            'total_profit': self.total_profit,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'trade_history_count': len(self.trades)
        }
    
    def clear_history(self):
        """Löscht die Trade-Historie"""
        self.trades = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.win_rate = 0.0
        self.performance_score = 0.0
        self.logger.info("Trade-Historie gelöscht")
