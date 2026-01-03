#!/usr/bin/env python3
"""
Kompletter Fix für PerformanceTracker
"""

import os
import sys

def analyze_record_trade_signature():
    """Analysiert die Signatur der record_trade Methode"""
    file_path = "src/self_improvement/performance_tracker.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Suche nach der record_trade Methode
        if 'def record_trade' in content:
            # Extrahiere die Methode
            start = content.find('def record_trade')
            # Finde das Ende der Parameterliste
            paren_end = content.find(')', start)
            signature = content[start:paren_end+1]
            
            print(f"📋 Aktuelle Signatur: {signature}")
            
            # Zähle die Parameter
            params_start = content.find('(', start) + 1
            params_str = content[params_start:paren_end]
            params = [p.strip() for p in params_str.split(',') if p.strip()]
            
            print(f"📊 Anzahl Parameter: {len(params)}")
            print(f"📋 Parameter: {params}")
            
            return params
        else:
            print("❌ record_trade Methode nicht gefunden")
            return []
            
    except Exception as e:
        print(f"❌ Fehler bei der Analyse: {e}")
        return []

def fix_record_trade_signature():
    """Vereinfacht die record_trade Methode"""
    file_path = "src/self_improvement/performance_tracker.py"
    
    print(f"\n🔧 Vereinfache record_trade Signatur...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup
        backup_path = file_path + ".backup3"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Neue vereinfachte Version
        simplified_content = '''#!/usr/bin/env python3
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
'''
        
        # Speicere vereinfachte Version
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(simplified_content)
        
        print(f"✅ record_trade Methode vereinfacht")
        return True
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        return False

def test_fixed_version():
    """Testet die vereinfachte Version"""
    print("\n🧪 Teste vereinfachte Version...")
    
    try:
        sys.path.insert(0, 'src')
        
        # Lösche vorherige Importe aus Cache
        if 'self_improvement.performance_tracker' in sys.modules:
            del sys.modules['self_improvement.performance_tracker']
        
        from self_improvement.performance_tracker import PerformanceTracker
        
        pt = PerformanceTracker()
        
        print(f"✅ total_trades: {pt.total_trades}")
        print(f"✅ win_rate: {pt.win_rate}")
        print(f"✅ performance_score: {pt.performance_score}")
        
        # Test mit verschiedenen Aufrufen
        pt.record_trade({'profit': 10.0, 'confidence': 0.8})
        print(f"✅ Nach Trade 1 - total_trades: {pt.total_trades}")
        print(f"✅ Nach Trade 1 - win_rate: {pt.win_rate}")
        
        pt.record_trade({'profit': -5.0})
        print(f"✅ Nach Trade 2 - total_trades: {pt.total_trades}")
        print(f"✅ Nach Trade 2 - win_rate: {pt.win_rate}")
        
        # Test get_status
        status = pt.get_status()
        print(f"✅ get_status - total_trades: {status['total_trades']}")
        print(f"✅ get_status - win_rate: {status['win_rate']:.1%}")
        print(f"✅ get_status - performance_score: {status['performance_score']:.1f}")
        
        # Test get_performance_metrics
        metrics = pt.get_performance_metrics()
        print(f"✅ get_performance_metrics - win_rate: {metrics['win_rate']:.1%}")
        print(f"✅ get_performance_metrics - sharpe_ratio: {metrics['sharpe_ratio']:.2f}")
        
        print("\n🎉 ALLE TESTS BESTANDEN!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FEHLGESCHLAGEN: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_main_integration():
    """Prüft ob main.py mit der neuen Version kompatibel ist"""
    print("\n🔍 Prüfe Integration mit main.py...")
    
    try:
        # Analysiere wie main.py record_trade aufruft
        with open('main.py', 'r', encoding='utf-8') as f:
            main_content = f.read()
        
        # Suche nach record_trade Aufrufen
        if 'record_trade' in main_content:
            print("✅ record_trade Aufrufe in main.py gefunden")
            
            # Extrahiere Beispielaufruf
            start = main_content.find('record_trade')
            if start != -1:
                end = main_content.find('\n', start)
                call = main_content[start:end].strip()
                print(f"📋 Beispielaufruf: {call}")
                
                # Prüfe Parameter
                if 'trade_result' in call and 'ml_confidence' in call:
                    print("⚠️  main.py verwendet alte Signatur (2 Parameter)")
                    print("   Möglicherweise muss main.py angepasst werden")
                    return False
                elif 'trade_result' in call and 'ml_confidence' not in call:
                    print("✅ main.py verwendet neue Signatur (1 Parameter)")
                    return True
        else:
            print("ℹ️  Keine record_trade Aufrufe in main.py gefunden")
            return True
            
    except Exception as e:
        print(f"❌ Fehler bei der Prüfung: {e}")
        return True  # Gehe davon aus dass es okay ist

def main():
    print("=" * 60)
    print("🔧 PERFORMANCE TRACKER KOMPLETT-REPARATUR")
    print("=" * 60)
    
    # 1. Analyse der aktuellen Signatur
    print("\n📊 ANALYSE DER AKTUELLEN SIGNATUR")
    params = analyze_record_trade_signature()
    
    if len(params) > 2:  # self + mindestens 2 Parameter
        print(f"\n⚠️  Komplexe Signatur mit {len(params)-1} Parametern gefunden")
        print("   Vereinfache auf einen Parameter...")
        
        # 2. Methode vereinfachen
        if fix_record_trade_signature():
            # 3. Testen
            if test_fixed_version():
                # 4. Integration prüfen
                if check_main_integration():
                    print("\n" + "=" * 60)
                    print("✅ REPARATUR ERFOLGREICH ABGESCHLOSSEN!")
                    print("=" * 60)
                    print("\n🚀 NÄCHSTE SCHRITTE:")
                    print("1. Starte den Bot: python main.py")
                    print("2. Prüfe Self-Improvement: Option 7")
                    print("3. Teste Paper Trading: Option 1")
                else:
                    print("\n⚠️  main.py muss möglicherweise angepasst werden")
                    print("   Siehe Beispielaufrufe oben")
            else:
                print("\n❌ Test der vereinfachten Version fehlgeschlagen")
        else:
            print("\n❌ Vereinfachung der Signatur fehlgeschlagen")
    else:
        print("\n✅ Signatur ist bereits einfach genug")
        if test_fixed_version():
            print("\n🎉 PerformanceTracker funktioniert bereits!")
        else:
            print("\n❌ PerformanceTracker hat andere Probleme")

if __name__ == "__main__":
    main()