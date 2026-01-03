"""
Self-Improvement Module für autonomen Trading Bot
"""

import pandas as pd  # ✅ FIXED: Pandas Import hinzugefügt
import numpy as np   # ✅ FIXED: NumPy Import hinzugefügt
from datetime import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from .performance_tracker import PerformanceTracker
from .adaptive_weight_manager import AdaptiveWeightManager
from .market_regime_detector import MarketRegimeDetector

class SelfImprovementManager:
    """
    Hauptmanager für alle Selbstoptimierungs-Komponenten
    """
    
    def __init__(self, config: dict = None):
        """
        Initialisiert den Self-Improvement Manager
        
        Args:
            config: Konfigurations-Dictionary
        """
        self.config = config or {}
        
        # Initialisiere Komponenten
        self.performance_tracker = PerformanceTracker(
            max_history_days=self.config.get('max_history_days', 30),
            min_trades_for_analysis=self.config.get('min_trades_for_analysis', 10)
        )
        
        self.adaptive_weight_manager = AdaptiveWeightManager(
            initial_ml_weight=self.config.get('initial_ml_weight', 0.7),
            min_ml_weight=self.config.get('min_ml_weight', 0.3),
            max_ml_weight=self.config.get('max_ml_weight', 0.9),
            adjustment_step=self.config.get('adjustment_step', 0.05)
        )
        
        self.market_regime_detector = MarketRegimeDetector(
            atr_period=self.config.get('atr_period', 14),
            trend_strength_period=self.config.get('trend_strength_period', 20),
            volatility_threshold=self.config.get('volatility_threshold', 1.5)
        )
        
        # State
        self.last_analysis_time = None
        self.auto_adjust_enabled = True
        
        print("[OK] Self-Improvement Manager initialisiert")
        print(f"     - ML Weight: {self.adaptive_weight_manager.ml_weight:.2f}")
        print(f"     - Auto-Adjust: {'ENABLED' if self.auto_adjust_enabled else 'DISABLED'}")
    
    def record_trade(self, 
                    symbol: str, 
                    direction: str, 
                    result_pips: float, 
                    ml_confidence: float,
                    entry_price: float,
                    exit_price: float,
                    duration_minutes: int = None):
        """
        Zeichnet einen Trade für alle Komponenten auf
        
        Args:
            symbol: Trading-Symbol
            direction: 'BUY' oder 'SELL'
            result_pips: Ergebnis in Pips
            ml_confidence: ML-Confidence (0-1)
            entry_price: Einstiegspreis
            exit_price: Ausstiegspreis
            duration_minutes: Trade-Dauer
        """
        # Performance Tracker
        self.performance_tracker.record_trade(
            symbol=symbol,
            direction=direction,
            result_pips=result_pips,
            confidence=ml_confidence,
            ml_weight=self.adaptive_weight_manager.ml_weight,
            entry_price=entry_price,
            exit_price=exit_price,
            duration_minutes=duration_minutes
        )
        
        # Adaptive Weight Manager
        self.adaptive_weight_manager.record_trade_result(
            symbol=symbol,
            result_pips=result_pips,
            ml_confidence=ml_confidence,
            used_ml_weight=self.adaptive_weight_manager.ml_weight
        )
        
        # Auto-Adjust nach jedem Trade prüfen
        if self.auto_adjust_enabled:
            self._try_auto_adjust()
    
    def _try_auto_adjust(self):
        """Versucht automatische Anpassung basierend auf Performance"""
        should_adjust, reason, new_weight = self.adaptive_weight_manager.auto_adjust_based_on_performance(
            self.performance_tracker,
            window_size=30
        )
        
        if should_adjust:
            print(f"⚙️  AUTO-ADJUST: {reason}")
            print(f"   ML Weight: {self.adaptive_weight_manager.ml_weight:.2f} → {new_weight:.2f}")
    
    def analyze_market_regime(self, candles: dict):
        """
        Analysiert Marktregime für alle Symbole
        
        Args:
            candles: Dictionary mit Symbol->DataFrame Mappings
        """
        regime_results = {}
        
        for symbol, df in candles.items():
            regime = self.market_regime_detector.detect_regime(df)
            regime_results[symbol] = regime
            
            # Trading-Empfehlungen basierend auf Regime
            recommendations = self.market_regime_detector.get_trading_recommendations(regime)
            
            if regime['confidence'] > 0.6:  # Nur bei hoher Confidence anzeigen
                print(f"📊 {symbol}: {regime['regime']} (Conf: {regime['confidence']:.2f})")
                print(f"   Empfehlung: {recommendations['action']}")
        
        return regime_results
    
    def get_performance_dashboard(self) -> dict:
        """Gibt Performance-Dashboard zurück"""
        performance = self.performance_tracker.get_performance_summary()
        weight_info = self.adaptive_weight_manager.get_adjustment_summary()
        confidence_analysis = self.performance_tracker.get_confidence_analysis()
        
        return {
            'performance': performance,
            'weight_management': weight_info,
            'confidence_analysis': confidence_analysis,
            'market_regime_stats': self.market_regime_detector.get_regime_statistics(),
            'timestamp': pd.Timestamp.now().isoformat()  # ✅ FIXED: pd verfügbar
        }
    
    def print_dashboard(self):
        """Gibt Performance-Dashboard in der Konsole aus"""
        try:
            dashboard = self.get_performance_dashboard()
            
            print("\n" + "="*60)
            print("🤖 SELF-IMPROVEMENT DASHBOARD")
            print("="*60)
            
            # Performance Summary
            perf = dashboard['performance'].get('overall', {})
            if perf:
                print(f"\n📈 PERFORMANCE SUMMARY")
                print(f"   Trades: {perf.get('total_trades', 0)}")
                print(f"   Win Rate: {perf.get('win_rate_percent', 0):.1f}%")
                print(f"   Net Pips: {perf.get('net_profit_pips', 0):.1f}")
                print(f"   Profit Factor: {perf.get('profit_factor', 0):.2f}")
                print(f"   Current Streak: {perf.get('current_streak', 0)}")
            
            # Weight Management
            weights = dashboard['weight_management']
            print(f"\n⚖️  WEIGHT MANAGEMENT")
            print(f"   ML Weight: {self.adaptive_weight_manager.ml_weight:.2f}")
            print(f"   Rules Weight: {self.adaptive_weight_manager.rules_weight:.2f}")
            print(f"   Total Adjustments: {weights.get('total_adjustments', 0)}")
            
            # Confidence Analysis
            conf = dashboard['confidence_analysis']
            if 'overall_accuracy' in conf:
                print(f"\n🎯 CONFIDENCE ANALYSIS")
                print(f"   Overall Accuracy: {conf['overall_accuracy']:.1%}")
                if 'recommendation' in conf:
                    print(f"   Recommendation: {conf['recommendation']}")
            
            # Market Regime
            regime_stats = dashboard['market_regime_stats']
            if regime_stats.get('total_samples', 0) > 0:
                print(f"\n🌍 MARKET REGIME")
                print(f"   Most Common: {regime_stats.get('most_common_regime', 'UNKNOWN')}")
                print(f"   Stability: {regime_stats.get('regime_stability', 0):.1%}")
            
            print("="*60)
        except Exception as e:
            print(f"⚠️  Dashboard Error: {str(e)}")
            print("   ℹ️  Using basic dashboard...")
            self._print_basic_dashboard()
    
    def _print_basic_dashboard(self):
        """Basic Dashboard Fallback"""
        print("\n📊 BASIC DASHBOARD")
        print(f"   ML Weight: {self.adaptive_weight_manager.ml_weight:.2f}")
        print(f"   Trades: {self.performance_tracker.metrics['total_trades']}")
        print(f"   Auto-Adjust: {'ON' if self.auto_adjust_enabled else 'OFF'}")
    
    def save_state(self, filepath: str = "data/self_improvement_state.json"):
        """Speichert den aktuellen Zustand"""
        import json
        from datetime import datetime
        
        state = {
            'performance_tracker': {
                'trade_count': len(self.performance_tracker.trade_history),
                'metrics': self.performance_tracker.metrics
            },
            'adaptive_weights': {
                'ml_weight': self.adaptive_weight_manager.ml_weight,
                'adjustment_count': len(self.adaptive_weight_manager.adjustment_history)
            },
            'market_regime': {
                'samples': len(self.market_regime_detector.regime_history)
            },
            'last_analysis': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'export_timestamp': datetime.now().isoformat()
        }
        
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"[OK] State gespeichert: {filepath}")
        return filepath
    
    def load_state(self, filepath: str = "data/self_improvement_state.json"):
        """Lädt gespeicherten Zustand"""
        try:
            import json
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            print(f"[OK] State geladen: {filepath}")
            return state
        except FileNotFoundError:
            print(f"[INFO] Kein State gefunden: {filepath}")
            return None
    
    def set_auto_adjust(self, enabled: bool):
        """Aktiviert/Deaktiviert automatische Anpassungen"""
        self.auto_adjust_enabled = enabled
        status = "ENABLED" if enabled else "DISABLED"
        print(f"[OK] Auto-Adjust: {status}")
    
    def get_recommended_position_size(self, 
                                     symbol: str, 
                                     ml_confidence: float, 
                                     base_size: float = 0.01,
                                     market_regime: str = None) -> float:
        """
        Gibt empfohlene Positionsgröße zurück
        
        Args:
            symbol: Trading-Symbol
            ml_confidence: ML-Confidence (0-1)
            base_size: Basis-Positionsgröße
            market_regime: Optionales Marktregime
            
        Returns:
            Empfohlene Positionsgröße
        """
        # Basis-Position basierend auf Confidence
        position = self.performance_tracker.get_recommended_position_size(
            ml_confidence, base_size
        )
        
        # Anpassung basierend auf Marktregime
        if market_regime:
            regime_multipliers = {
                'HIGH_VOLATILITY': 0.5,
                'STRONG_UPTREND': 1.2,
                'STRONG_DOWNTREND': 1.2,
                'TRENDING': 1.0,
                'RANGING': 0.8,
                'MEAN_REVERTING': 0.7,
                'NEUTRAL': 0.9
            }
            
            multiplier = regime_multipliers.get(market_regime, 1.0)
            position *= multiplier
        
        return round(position, 4)  # Runde auf 4 Dezimalstellen für Lots
    
    def run_daily_analysis(self):
        """Führt tägliche Analyse durch (für Scheduler)"""
        print("\n" + "="*60)
        print("📊 TÄGLICHE SELBST-ANALYSE")
        print("="*60)
        
        # 1. Performance Report speichern
        report_path = self.performance_tracker.save_performance_report()
        print(f"[OK] Performance Report: {report_path}")
        
        # 2. Gewichtungs-Analyse
        weight_performance = self.adaptive_weight_manager.get_performance_by_weight_range()
        print(f"[INFO] Weight Performance Analysis abgeschlossen")
        
        # 3. Confidence-Analyse
        conf_analysis = self.performance_tracker.get_confidence_analysis()
        if 'recommendation' in conf_analysis:
            print(f"[RECOMMENDATION] {conf_analysis['recommendation']}")
        
        self.last_analysis_time = pd.Timestamp.now()
        print(f"[OK] Tägliche Analyse abgeschlossen")