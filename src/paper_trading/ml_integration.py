"""
ML INTEGRATION FÜR PAPER TRADING - VOLLSTÄNDIGE VERSION
=======================================================
Führt ML-Signale in echte Paper-Trades um.
"""

from datetime import datetime
from typing import Tuple, Optional, Dict
from enum import Enum
import random

class SignalType(Enum):
    """ML-Signal Typen"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MLTradingEngine:
    """
    Haupt-Engine, die ML-Signale automatisch in Paper-Trades umwandelt.
    """
    
    def __init__(self, 
                 paper_portfolio,
                 ml_generator=None,
                 mt5_client=None):
        """
        Initialisiert die ML Trading Engine.
        """
        self.portfolio = paper_portfolio
        self.ml_generator = ml_generator
        self.mt5_client = mt5_client
        
        self.symbol = "EURUSD"
        self.confidence_threshold = 0.65
        
        # Trading-Parameter
        self.stop_loss_pips = 20  # 20 Pips Stop-Loss
        self.take_profit_pips = 40  # 40 Pips Take-Profit
        self.trade_size = 0.1  # Standard: 0.1 Lots
        
        # Statistik
        self.signals_generated = 0
        self.trades_executed = 0
        self.last_signal = None
        
        print(f"✅ ML Trading Engine für {self.symbol}")
        print(f"   Confidence Threshold: {self.confidence_threshold:.0%}")
        print(f"   Stop-Loss: {self.stop_loss_pips} Pips")
        print(f"   Take-Profit: {self.take_profit_pips} Pips")
    
    def generate_signal(self) -> Tuple[SignalType, float, Dict]:
        """
        Generiert ein ML-Signal basierend auf aktuellen Marktdaten.
        """
        self.signals_generated += 1
        
        try:
            # Versuche echten ML-Generator zu verwenden
            if self.ml_generator and hasattr(self.ml_generator, 'generate_signal'):
                ml_signal, confidence = self.ml_generator.generate_signal()
            else:
                # Fallback: Simuliertes Signal
                ml_signal, confidence = self._generate_simulated_signal()
            
            # Signal in Enum umwandeln
            if ml_signal == "BUY" and confidence >= self.confidence_threshold:
                signal_type = SignalType.BUY
            elif ml_signal == "SELL" and confidence >= self.confidence_threshold:
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD
                confidence = max(0.3, confidence)  # Mindest-Confidence für HOLD
            
            # Aktuellen Preis holen (simuliert oder von MT5)
            current_price = self._get_current_price()
            
            # Stop-Loss und Take-Profit berechnen
            pip_value = 0.0001  # Für EURUSD
            if signal_type == SignalType.BUY:
                stop_loss = current_price - (self.stop_loss_pips * pip_value)
                take_profit = current_price + (self.take_profit_pips * pip_value)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price + (self.stop_loss_pips * pip_value)
                take_profit = current_price - (self.take_profit_pips * pip_value)
            else:
                stop_loss = None
                take_profit = None
            
            signal_details = {
                'timestamp': datetime.now().isoformat(),
                'price': current_price,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_strength': self._calculate_signal_strength(confidence)
            }
            
            self.last_signal = signal_type
            
            return signal_type, confidence, signal_details
            
        except Exception as e:
            print(f"❌ Signal-Generierungsfehler: {e}")
            return SignalType.HOLD, 0.0, {}
    
    def execute_trade(self, 
                     signal_type: SignalType, 
                     confidence: float,
                     signal_details: Dict) -> bool:
        """
        Führt einen Trade basierend auf ML-Signal aus.
        Gibt True zurück wenn Trade ausgeführt wurde.
        """
        # Kein Trade bei HOLD
        if signal_type == SignalType.HOLD:
            print(f"   ⏸️  HOLD Signal (Confidence: {confidence:.1%})")
            return False
        
        # Prüfe ob Position bereits offen
        if self.symbol in self.portfolio.positions:
            print(f"   ⚠️  Position in {self.symbol} bereits offen")
            return self._manage_existing_position(signal_type, confidence, signal_details)
        
        # Hole Trade-Parameter
        entry_price = signal_details.get('price', 1.0850)
        stop_loss = signal_details.get('stop_loss')
        take_profit = signal_details.get('take_profit')
        
        print(f"\n   🎯 TRADE PARAMETER:")
        print(f"      Entry:       {entry_price:.5f}")
        print(f"      Confidence:  {confidence:.1%}")
        if stop_loss:
            print(f"      Stop-Loss:   {stop_loss:.5f}")
        if take_profit:
            print(f"      Take-Profit: {take_profit:.5f}")
        
        # Order-Typ festlegen
        from src.paper_trading.portfolio import OrderType
        order_type = OrderType.BUY if signal_type == SignalType.BUY else OrderType.SELL
        
        # Position öffnen
        try:
            order = self.portfolio.open_position(
                symbol=self.symbol,
                order_type=order_type,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                signal_confidence=confidence,
                custom_size=self.trade_size  # Feste Positionsgröße
            )
            
            if order:
                self.trades_executed += 1
                print(f"   ✅ {signal_type.value} Position eröffnet")
                print(f"      Order ID: {order.order_id}")
                print(f"      Size: {order.quantity} Lots")
                return True
            else:
                print(f"   ❌ Position konnte nicht eröffnet werden")
                return False
                
        except Exception as e:
            print(f"   ❌ Trade-Fehler: {e}")
            return False
    
    def _manage_existing_position(self, 
                                 signal_type: SignalType,
                                 confidence: float,
                                 signal_details: Dict) -> bool:
        """
        Verwaltet eine bereits offene Position.
        """
        current_order = self.portfolio.positions[self.symbol]
        current_price = signal_details.get('price', 1.0850)
        
        # Prüfe ob entgegengesetztes Signal vorhanden
        from src.paper_trading.portfolio import OrderType
        current_is_buy = current_order.order_type == OrderType.BUY
        new_is_sell = signal_type == SignalType.SELL
        
        if (current_is_buy and new_is_sell) or (not current_is_buy and signal_type == SignalType.BUY):
            # Entgegengesetztes Signal - Position schließen
            print(f"   🔄 ENTGEGENGESETZTES SIGNAL: Schließe Position")
            
            pnl = self.portfolio.close_position(self.symbol, current_price)
            if pnl is not None:
                pnl_sign = "+" if pnl > 0 else ""
                print(f"   💰 Position geschlossen mit P&L: {pnl_sign}{pnl:.2f} USD")
                return True
        
        return False
    
    def run_single_iteration(self) -> Dict:
        """
        Führt eine komplette Trading-Iteration durch.
        """
        print(f"\n   🔄 TRADING ITERATION #{self.signals_generated + 1}")
        print(f"   {'-'*40}")
        
        # 1. Signal generieren
        signal_type, confidence, details = self.generate_signal()
        
        # 2. Trade ausführen
        trade_executed = self.execute_trade(signal_type, confidence, details)
        
        # 3. Aktuellen Portfolio-Status anzeigen
        if trade_executed:
            summary = self.portfolio.get_portfolio_summary()
            print(f"\n   📊 AKTUELLER STATUS:")
            print(f"      Balance: {summary['current_balance']:.2f} USD")
            print(f"      Positions: {summary['open_positions']}")
            print(f"      Total Trades: {summary['total_trades']}")
        
        return {
            'signal': signal_type.value,
            'confidence': confidence,
            'trade_executed': trade_executed,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict:
        """Gibt Trading-Statistiken zurück."""
        summary = self.portfolio.get_portfolio_summary()
        
        return {
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'current_balance': self.portfolio.current_balance,
            'total_pnl': summary['performance_metrics']['total_pnl'],
            'total_trades': summary['total_trades'],
            'win_rate': self._calculate_win_rate(),
            'portfolio_id': self.portfolio.portfolio_id
        }
    
    def _generate_simulated_signal(self) -> Tuple[str, float]:
        """Generiert ein simuliertes ML-Signal für Tests."""
        signals = ["BUY", "SELL", "HOLD"]
        weights = [0.35, 0.35, 0.30]
        signal = random.choices(signals, weights=weights)[0]
        
        # Confidence basierend auf Signal-Typ
        if signal == "BUY":
            confidence = random.uniform(0.45, 0.85)
        elif signal == "SELL":
            confidence = random.uniform(0.45, 0.85)
        else:
            confidence = random.uniform(0.30, 0.60)
        
        return signal, confidence
    
    def _get_current_price(self) -> float:
        """Holt aktuellen Preis (simuliert für Demo)."""
        # Basispreis mit etwas Zufall
        base_price = 1.0850
        random_change = random.uniform(-0.0010, 0.0010)
        return round(base_price + random_change, 5)
    
    def _calculate_signal_strength(self, confidence: float) -> str:
        """Berechnet die Signal-Stärke."""
        if confidence >= 0.80:
            return "SEHR STARK"
        elif confidence >= 0.70:
            return "STARK"
        elif confidence >= 0.60:
            return "MITTEL"
        elif confidence >= 0.50:
            return "SCHWACH"
        else:
            return "SEHR SCHWACH"
    
    def _calculate_win_rate(self) -> float:
        """Berechnet die Win-Rate."""
        if not self.portfolio.trade_history:
            return 0.0
        
        winning_trades = [t for t in self.portfolio.trade_history if t.pnl and t.pnl > 0]
        return len(winning_trades) / len(self.portfolio.trade_history)


def test_ml_trading_engine():
    """Testet die ML Trading Engine."""
    print("🧪 TEST ML TRADING ENGINE")
    print("="*50)
    
    # Portfolio importieren
    from src.paper_trading.portfolio import PaperPortfolio
    
    # Portfolio erstellen
    portfolio = PaperPortfolio(initial_balance=10000.0)
    
    # Engine erstellen
    engine = MLTradingEngine(paper_portfolio=portfolio)
    
    # Mehrere Iterationen testen
    print("\n🚀 STARTE TEST MIT 5 ITERATIONEN")
    print("="*50)
    
    for i in range(5):
        print(f"\n📊 ITERATION {i+1}/5")
        result = engine.run_single_iteration()
        
        if i < 4:  # Nicht nach der letzten Iteration warten
            print(f"\n⏱️  Warte 2 Sekunden...")
            import time
            time.sleep(2)
    
    # Finale Statistik
    print("\n" + "="*50)
    print("📈 FINALE STATISTIK")
    print("="*50)
    
    stats = engine.get_statistics()
    print(f"Signale generiert: {stats['signals_generated']}")
    print(f"Trades ausgeführt: {stats['trades_executed']}")
    print(f"Aktuelle Balance: {stats['current_balance']:.2f} USD")
    print(f"Gesamt-P&L: {stats['total_pnl']:.2f} USD")
    print(f"Win Rate: {stats['win_rate']:.1%}")
    
    # Portfolio-Report
    print("\n" + "="*50)
    portfolio.print_detailed_report()
    
    # Report speichern
    portfolio.save_performance_report()
    
    print("\n🎉 TEST ERFOLGREICH ABGESCHLOSSEN!")
    return engine


if __name__ == "__main__":
    test_ml_trading_engine()