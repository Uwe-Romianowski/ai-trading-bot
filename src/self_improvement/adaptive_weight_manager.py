#!/usr/bin/env python3
"""
Adaptive Weight Manager - Reparierte Version
Passt ML vs Rule Gewichtung basierend auf Performance an
"""

import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AdaptiveWeightManager:
    """Verwaltet und passt Gewichtung zwischen ML und Regel-basierten Signalen an"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den Adaptive Weight Manager.
        
        Args:
            config: Konfigurations-Dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Extrahiere Konfiguration SICHER
        try:
            # Konvertiere alle Werte zu float
            self.initial_ml_weight = float(config.get('initial_ml_weight', 0.7))
            self.min_ml_weight = float(config.get('min_ml_weight', 0.3))
            self.max_ml_weight = float(config.get('max_ml_weight', 0.9))
            self.adjustment_step = float(config.get('adjustment_step', 0.05))
            
            # Sicherstellen dass Werte gültig sind
            self.initial_ml_weight = max(self.min_ml_weight, min(self.max_ml_weight, self.initial_ml_weight))
            
            self.current_ml_weight = self.initial_ml_weight
            self.current_rule_weight = 1.0 - self.current_ml_weight
            
            # Extrahiere max_history sicher
            max_history_raw = config.get('max_history_days', 30)
            if isinstance(max_history_raw, dict):
                self.max_history = 30
                self.logger.warning("max_history_days war ein dict, verwende Default 30")
            else:
                self.max_history = int(float(max_history_raw))
            
            self.performance_history = []
            
            self.logger.info(
                f"Adaptive Weight Manager initialisiert: "
                f"ML={self.current_ml_weight:.2f}, "
                f"Rules={self.current_rule_weight:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Fehler in Konfiguration: {e}, verwende Defaults")
            # Fallback-Werte
            self.initial_ml_weight = 0.7
            self.min_ml_weight = 0.3
            self.max_ml_weight = 0.9
            self.adjustment_step = 0.05
            self.max_history = 30
            self.current_ml_weight = self.initial_ml_weight
            self.current_rule_weight = 1.0 - self.current_ml_weight
    
    def calculate_new_weights(self, performance_metrics: Dict[str, Any]) -> Dict[str, float]:
        """
        Berechnet neue Gewichtung basierend auf Performance.
        
        Args:
            performance_metrics: Dictionary mit Performance-Metriken
            
        Returns:
            Dictionary mit neuen Gewichten
        """
        try:
            # Extrahiere Metriken sicher
            win_rate = float(performance_metrics.get('win_rate', 0.5))
            sharpe_ratio = float(performance_metrics.get('sharpe_ratio', 0.0))
            total_trades = int(performance_metrics.get('total_trades', 0))
            
            # Nur anpassen wenn genug Trades
            if total_trades < 10:
                return {
                    'ml_weight': self.current_ml_weight,
                    'rule_weight': self.current_rule_weight
                }
            
            # Einfache Anpassungslogik
            adjustment = 0.0
            
            # Basierend auf Win Rate
            if win_rate > 0.6:
                adjustment = self.adjustment_step  # Erhöhe ML Vertrauen
            elif win_rate < 0.4:
                adjustment = -self.adjustment_step  # Reduziere ML Vertrauen
            
            # Basierend auf Sharpe Ratio
            if sharpe_ratio > 1.0:
                adjustment += 0.02
            elif sharpe_ratio < -0.5:
                adjustment -= 0.02
            
            # Neue Gewichte berechnen
            new_ml_weight = self.current_ml_weight + adjustment
            new_ml_weight = max(self.min_ml_weight, min(self.max_ml_weight, new_ml_weight))
            new_rule_weight = 1.0 - new_ml_weight
            
            # Nur aktualisieren wenn Änderung signifikant
            if abs(new_ml_weight - self.current_ml_weight) > 0.01:
                self.current_ml_weight = new_ml_weight
                self.current_rule_weight = new_rule_weight
                
                self.logger.info(
                    f"Gewichte angepasst: "
                    f"ML={self.current_ml_weight:.2f} (Δ{adjustment:+.2f}), "
                    f"Rules={self.current_rule_weight:.2f}, "
                    f"Win Rate={win_rate:.1%}"
                )
            
            return {
                'ml_weight': self.current_ml_weight,
                'rule_weight': self.current_rule_weight
            }
            
        except Exception as e:
            self.logger.error(f"Fehler in calculate_new_weights: {e}")
            return {
                'ml_weight': self.current_ml_weight,
                'rule_weight': self.current_rule_weight
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt aktuellen Status zurück"""
        return {
            'current_ml_weight': self.current_ml_weight,
            'current_rule_weight': self.current_rule_weight,
            'initial_ml_weight': self.initial_ml_weight,
            'adjustment_step': self.adjustment_step,
            'performance_history_count': len(self.performance_history)
        }
