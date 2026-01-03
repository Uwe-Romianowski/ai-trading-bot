#!/usr/bin/env python3
"""
Market Regime Detector - Vereinfachte Version
Erkennt verschiedene Marktregimes für adaptive Strategien
"""

import logging
import numpy as np
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Erkennt und klassifiziert Marktregimes"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialisiert den Market Regime Detector.
        
        Args:
            config: Konfigurations-Dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # Extrahiere Konfiguration sicher
        try:
            self.atr_period = int(float(config.get('atr_period', 14)))
            self.trend_period = int(float(config.get('trend_strength_period', 20)))
            self.volatility_threshold = float(config.get('volatility_threshold', 1.5))
            
            self.regime_history = []
            self.current_regime = 'NEUTRAL'
            
            self.logger.info(
                f"Market Regime Detector initialisiert: "
                f"ATR={self.atr_period}, Trend={self.trend_period}"
            )
            
        except Exception as e:
            self.logger.error(f"Fehler in Konfiguration: {e}, verwende Defaults")
            self.atr_period = 14
            self.trend_period = 20
            self.volatility_threshold = 1.5
            self.regime_history = []
            self.current_regime = 'NEUTRAL'
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Erkennt das aktuelle Marktregime.
        
        Args:
            price_data: DataFrame mit Preis-Daten
            
        Returns:
            Dictionary mit Regime-Informationen
        """
        try:
            if price_data.empty or len(price_data) < 20:
                return {
                    'regime': 'NEUTRAL',
                    'trend': 'SIDEWAYS',
                    'volatility': 'NORMAL',
                    'confidence': 0.5
                }
            
            prices = price_data['close'].values
            
            # Trend berechnen (einfache Methode)
            if len(prices) >= self.trend_period:
                recent = prices[-self.trend_period:]
                old = prices[-self.trend_period*2:-self.trend_period]
                
                if len(recent) > 0 and len(old) > 0:
                    recent_avg = np.mean(recent)
                    old_avg = np.mean(old)
                    trend_strength = (recent_avg - old_avg) / old_avg
                else:
                    trend_strength = 0.0
            else:
                trend_strength = 0.0
            
            # Volatilität berechnen (einfache Methode)
            if len(prices) >= 10:
                returns = np.diff(prices[-10:]) / prices[-11:-1]
                volatility = np.std(returns) * np.sqrt(252)  # Annualisiert
            else:
                volatility = 0.1
            
            # Regime bestimmen
            trend = 'BULLISH' if trend_strength > 0.01 else 'BEARISH' if trend_strength < -0.01 else 'SIDEWAYS'
            vol = 'HIGH' if volatility > self.volatility_threshold else 'NORMAL'
            
            # Kombiniertes Regime
            if trend == 'BULLISH' and vol == 'HIGH':
                regime = 'TRENDING_BULL_HIGH_VOL'
            elif trend == 'BULLISH':
                regime = 'TRENDING_BULL'
            elif trend == 'BEARISH' and vol == 'HIGH':
                regime = 'TRENDING_BEAR_HIGH_VOL'
            elif trend == 'BEARISH':
                regime = 'TRENDING_BEAR'
            elif vol == 'HIGH':
                regime = 'RANGING_HIGH_VOL'
            else:
                regime = 'RANGING'
            
            # Confidence berechnen
            confidence = min(0.9, abs(trend_strength) * 10 + (0.3 if vol == 'HIGH' else 0.1))
            
            self.current_regime = regime
            self.regime_history.append({
                'timestamp': pd.Timestamp.now(),
                'regime': regime,
                'trend': trend,
                'volatility': vol,
                'confidence': confidence
            })
            
            # History begrenzen
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return {
                'regime': regime,
                'trend': trend,
                'volatility': vol,
                'trend_strength': float(trend_strength),
                'volatility_value': float(volatility),
                'confidence': float(confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Fehler in detect_regime: {e}")
            return {
                'regime': 'NEUTRAL',
                'trend': 'SIDEWAYS',
                'volatility': 'NORMAL',
                'confidence': 0.5
            }
    
    def get_recommended_weights(self, regime_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Gibt empfohlene Gewichte basierend auf Regime zurück.
        
        Args:
            regime_info: Regime-Informationen
            
        Returns:
            Dictionary mit empfohlenen Gewichten
        """
        regime = regime_info.get('regime', 'NEUTRAL')
        
        # Einfache Empfehlungen
        if 'HIGH_VOL' in regime:
            # Bei hoher Volatilität weniger ML Vertrauen
            return {'ml_weight': 0.5, 'rule_weight': 0.5}
        elif 'TRENDING' in regime:
            # Im Trend mehr ML Vertrauen
            return {'ml_weight': 0.8, 'rule_weight': 0.2}
        elif 'RANGING' in regime:
            # Seitwärts weniger ML Vertrauen
            return {'ml_weight': 0.4, 'rule_weight': 0.6}
        else:
            # Neutral
            return {'ml_weight': 0.7, 'rule_weight': 0.3}
    
    def get_status(self) -> Dict[str, Any]:
        """Gibt Status zurück"""
        return {
            'current_regime': self.current_regime,
            'regime_history_count': len(self.regime_history),
            'atr_period': self.atr_period,
            'trend_period': self.trend_period,
            'volatility_threshold': self.volatility_threshold
        }
