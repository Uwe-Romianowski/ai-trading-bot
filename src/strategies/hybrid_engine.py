# src/strategies/hybrid_engine.py
import logging
from typing import Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class HybridEngine:
    """Kombiniert ML-Signale mit regelbasierten Signalen"""
    
    def __init__(self, ml_generator=None, ml_weight: float = 0.7, rule_weight: float = 0.3):
        """
        Initialisiert die Hybrid Engine.
        
        Args:
            ml_generator: MLSignalGenerator Instanz (optional)
            ml_weight: Gewichtung für ML-Signale (0.0-1.0)
            rule_weight: Gewichtung für Regel-Signale (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.ml_generator = ml_generator
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        
        # Normalisiere Gewichte
        total_weight = ml_weight + rule_weight
        if total_weight > 0:
            self.ml_weight = ml_weight / total_weight
            self.rule_weight = rule_weight / total_weight
        
        self.logger.info(f"Hybrid Engine initialisiert (ML: {self.ml_weight:.2f}, Rules: {self.rule_weight:.2f})")
    
    def generate_rule_signal(self, symbol: str, current_price: float) -> Dict:
        """Generiert ein regelbasiertes Signal"""
        # Einfache regelbasierte Strategie
        # Hier könnten komplexe Regeln implementiert werden
        
        import random
        from datetime import datetime
        
        # Beispiel: Zufälliges Signal für Demo
        signals = ['BUY', 'SELL', 'HOLD']
        weights = [0.3, 0.3, 0.4]  # 30% BUY, 30% SELL, 40% HOLD
        signal = np.random.choice(signals, p=weights)
        
        # Simuliere Confidence
        confidence = random.uniform(0.4, 0.8)
        
        return {
            'action': signal,
            'confidence': confidence,
            'source': 'RULES',
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': current_price
        }
    
    def combine_signals(self, ml_signal: Optional[Dict] = None, 
                       symbol: str = "EURUSD", 
                       current_price: float = 0.0) -> Dict:
        """
        Kombiniert ML- und Regel-Signale basierend auf Gewichtung.
        
        Args:
            ml_signal: ML-Signal (optional)
            symbol: Trading Symbol
            current_price: Aktueller Preis
            
        Returns:
            Kombiniertes Signal
        """
        from datetime import datetime
        
        # Standard-Signal
        default_signal = {
            'action': 'HOLD',
            'confidence': 0.0,
            'source': 'HYBRID',
            'ml_confidence': 0.0,
            'rule_confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'price': current_price
        }
        
        # Generiere Regel-Signal
        rule_signal = self.generate_rule_signal(symbol, current_price)
        
        # Falls kein ML-Signal, verwende nur Regeln
        if ml_signal is None or 'error' in ml_signal:
            self.logger.warning("Kein ML-Signal verfügbar, verwende nur Regeln")
            default_signal['action'] = rule_signal['action']
            default_signal['confidence'] = rule_signal['confidence']
            default_signal['source'] = 'RULES_ONLY'
            default_signal['rule_confidence'] = rule_signal['confidence']
            return default_signal
        
        # Extrahiere ML-Signal
        ml_action = ml_signal.get('signal', 'HOLD')
        ml_confidence = ml_signal.get('confidence', 0.0)
        
        # Kombiniere basierend auf Gewichtung
        if ml_action == 'HOLD' and rule_signal['action'] == 'HOLD':
            default_signal['action'] = 'HOLD'
            default_signal['confidence'] = 0.0
        elif ml_action == rule_signal['action']:
            # Beide Signale stimmen überein
            default_signal['action'] = ml_action
            default_signal['confidence'] = (
                ml_confidence * self.ml_weight + 
                rule_signal['confidence'] * self.rule_weight
            )
        else:
            # Signale widersprechen sich
            ml_score = ml_confidence * self.ml_weight
            rule_score = rule_signal['confidence'] * self.rule_weight
            
            if ml_score > rule_score:
                default_signal['action'] = ml_action
                default_signal['confidence'] = ml_confidence
                default_signal['source'] = 'ML_DOMINANT'
            else:
                default_signal['action'] = rule_signal['action']
                default_signal['confidence'] = rule_signal['confidence']
                default_signal['source'] = 'RULES_DOMINANT'
        
        default_signal['ml_confidence'] = ml_confidence
        default_signal['rule_confidence'] = rule_signal['confidence']
        
        # Logge Entscheidung bei hoher Confidence
        if default_signal['confidence'] > 0.65 and default_signal['action'] != 'HOLD':
            self.logger.info(
                f"Hybrid Signal: {symbol} {default_signal['action']} "
                f"(Confidence: {default_signal['confidence']:.2%}, "
                f"Source: {default_signal['source']})"
            )
        
        return default_signal
    
    def update_weights(self, ml_weight: float, rule_weight: float):
        """Aktualisiert die Gewichtung"""
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        
        # Normalisiere
        total_weight = ml_weight + rule_weight
        if total_weight > 0:
            self.ml_weight = ml_weight / total_weight
            self.rule_weight = rule_weight / total_weight
        
        self.logger.info(f"Gewichte aktualisiert: ML={self.ml_weight:.2f}, Rules={self.rule_weight:.2f}")
    
    def get_status(self) -> Dict:
        """Gibt Status der Hybrid Engine zurück"""
        return {
            'ml_weight': self.ml_weight,
            'rule_weight': self.rule_weight,
            'ml_available': self.ml_generator is not None,
            'total_weight': self.ml_weight + self.rule_weight
        }