"""
Hybrid Decision Engine - Kombiniert Regel- und ML-basierte Signale
"""
import logging
from typing import Dict, Optional
import numpy as np
import yaml

class HybridDecisionEngine:
    """
    Kombiniert regelbasierte und ML-basierte Trading-Signale.
    """
    
    def __init__(self, config_path: str = "config/bot_config.yaml"):
        """Initialisiert die Hybrid Decision Engine."""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_config()
        
        # Standard-Gewichtungen
        self.ml_weight = self.config.get('ml_weight', 0.3)  # 30% ML, 70% Regeln
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        
        # Performance Tracking
        self.decisions_made = 0
        self.ml_overrides = 0
        self.rule_overrides = 0
        
        self.logger.info(f"Hybrid Decision Engine initialisiert (ML Weight: {self.ml_weight})")
    
    def _load_config(self) -> dict:
        """Lädt die Konfiguration."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Konfiguration konnte nicht geladen werden: {e}")
            return {}
    
    def decide(self, rule_signal: Optional[Dict] = None, 
               ml_signal: Optional[Dict] = None, 
               ml_weight: Optional[float] = None) -> Dict:
        """
        Entscheidet basierend auf Regel- und ML-Signalen.
        
        Args:
            rule_signal: Regel-basiertes Signal
            ml_signal: ML-basiertes Signal
            ml_weight: Gewichtung für ML (0.0-1.0)
        
        Returns:
            dict: Finales Hybrid-Signal
        """
        self.decisions_made += 1
        
        # Standardwerte
        ml_weight = ml_weight if ml_weight is not None else self.ml_weight
        rule_signal = rule_signal or {'action': 'HOLD', 'confidence': 0.5}
        ml_signal = ml_signal or {'action': 'HOLD', 'confidence': 0.5}
        
        # Extrahiere Aktionen und Confidence
        rule_action = rule_signal.get('action', 'HOLD')
        ml_action = ml_signal.get('action', 'HOLD')
        rule_confidence = rule_signal.get('confidence', 0.5)
        ml_confidence = ml_signal.get('confidence', 0.5)
        
        # 1. Wenn beide HOLD → HOLD
        if rule_action == 'HOLD' and ml_action == 'HOLD':
            final_action = 'HOLD'
            final_confidence = (rule_confidence + ml_confidence) / 2
        
        # 2. Wenn eine Seite HOLD, andere nicht → Nicht-HOLD mit angepasster Confidence
        elif rule_action == 'HOLD' and ml_action != 'HOLD':
            if ml_confidence >= self.confidence_threshold:
                final_action = ml_action
                final_confidence = ml_confidence * ml_weight
                self.ml_overrides += 1
            else:
                final_action = 'HOLD'
                final_confidence = (rule_confidence + ml_confidence * ml_weight) / 2
        
        elif ml_action == 'HOLD' and rule_action != 'HOLD':
            if rule_confidence >= self.confidence_threshold:
                final_action = rule_action
                final_confidence = rule_confidence * (1 - ml_weight)
                self.rule_overrides += 1
            else:
                final_action = 'HOLD'
                final_confidence = (rule_confidence * (1 - ml_weight) + ml_confidence) / 2
        
        # 3. Wenn beide BUY/SELL → Gewichtete Entscheidung
        else:
            # Gleiche Richtung
            if rule_action == ml_action:
                final_action = rule_action
                final_confidence = (rule_confidence * (1 - ml_weight) + 
                                   ml_confidence * ml_weight)
            
            # Gegensätzliche Richtung
            else:
                # Wähle höhere Confidence mit Gewichtung
                rule_strength = rule_confidence * (1 - ml_weight)
                ml_strength = ml_confidence * ml_weight
                
                if rule_strength > ml_strength:
                    final_action = rule_action
                    final_confidence = rule_strength
                    self.rule_overrides += 1
                else:
                    final_action = ml_action
                    final_confidence = ml_strength
                    self.ml_overrides += 1
        
        # Ergebnis zusammenstellen
        result = {
            'action': final_action,
            'confidence': float(final_confidence),
            'rule_signal': rule_signal,
            'ml_signal': ml_signal,
            'ml_weight': ml_weight,
            'decision_id': self.decisions_made,
            'engine_type': 'hybrid'
        }
        
        return result
    
    def set_ml_weight(self, weight: float):
        """Setzt die ML-Gewichtung."""
        if 0.0 <= weight <= 1.0:
            self.ml_weight = weight
            self.logger.info(f"ML-Gewichtung auf {weight:.1%} gesetzt")
        else:
            raise ValueError("ML-Gewichtung muss zwischen 0.0 und 1.0 liegen")
    
    def get_stats(self) -> Dict:
        """Gibt Entscheidungs-Statistiken zurück."""
        return {
            'decisions_made': self.decisions_made,
            'ml_overrides': self.ml_overrides,
            'rule_overrides': self.rule_overrides,
            'ml_weight': self.ml_weight,
            'confidence_threshold': self.confidence_threshold,
            'ml_override_rate': self.ml_overrides / self.decisions_made if self.decisions_made > 0 else 0
        }
    
    def reset_stats(self):
        """Setzt die Statistiken zurück."""
        self.decisions_made = 0
        self.ml_overrides = 0
        self.rule_overrides = 0
        self.logger.info("Statistiken zurückgesetzt")