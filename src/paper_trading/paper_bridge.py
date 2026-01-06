"""
PAPER TRADING BRIDGE v4.2 - MIT TRADE HISTORY
"""

import os
import sys
import json
import csv
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import traceback

# Importiere interne Module
try:
    from .enhanced_ml_engine import EnhancedMLTradingEngine
    from .portfolio import Portfolio
    from .ml_integration import MLTradingEngine
except ImportError:
    # Fallback für direkten Aufruf
    try:
        from enhanced_ml_engine import EnhancedMLTradingEngine
        from portfolio import Portfolio
        from ml_integration import MLTradingEngine
    except ImportError:
        print("⚠️  Einige Module nicht verfügbar")


class PaperTradingBridge:
    """
    Bridge zwischen ML Engine und Paper Trading.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """Initialisiert die Paper Trading Bridge."""
        self.portfolio = Portfolio(initial_balance)
        self.ml_engine = EnhancedMLTradingEngine(self.portfolio)
        self.trading_active = False
        self.iteration_count = 0
        self.max_iterations = 10
        self.symbol = "EURUSD"
        
        # Trade History Einstellungen
        self.history_settings = {
            'save_to_csv': True,
            'csv_file': 'data/paper_trading/trade_history_detailed.csv',
            'save_to_json': True,
            'json_file': 'data/paper_trading/trade_history.json',
            'backup_interval': 10,  # Backup alle 10 Trades
            'compress_old_files': True
        }
        
        # Performance Tracking
        self.performance_data = {
            'session_start': datetime.now().isoformat(),
            'total_iterations': 0,
            'signals_generated': 0,
            'trades_executed': 0,
            'total_volume': 0.0,
            'total_commission': 0.0,
            'session_pnl': 0.0,
            'peak_balance': initial_balance,
            'lowest_balance': initial_balance,
            'drawdown': 0.0
        }
        
        # Erstelle benötigte Verzeichnisse
        self.create_directories()
        
    def create_directories(self):
        """Erstellt benötigte Verzeichnisse."""
        directories = [
            'data/paper_trading',
            'data/paper_trading/backups',
            'data/paper_trading/logs',
            'data/performance_reports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def start_trading_session(self, iterations: int = 10, symbol: str = "EURUSD"):
        """
        Startet eine Paper Trading Session.
        
        Args:
            iterations: Anzahl der Iterationen
            symbol: Trading-Symbol
        """
        print(f"\n{'='*80}")
        print(f"🚀 PAPER TRADING SESSION STARTEN")
        print(f"{'='*80}")
        
        self.max_iterations = iterations
        self.symbol = symbol
        self.trading_active = True
        self.iteration_count = 0
        
        # Session-Log Datei
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"data/paper_trading/logs/session_{session_id}.log"
        
        print(f"📝 Session ID: {session_id}")
        print(f"🔢 Iterationen: {iterations}")
        print(f"💱 Symbol: {symbol}")
        print(f"💰 Startkapital: ${self.portfolio.initial_balance:,.2f}")
        print(f"{'='*80}\n")
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"Paper Trading Session Start: {datetime.now().isoformat()}\n")
                f.write(f"Iterations: {iterations}, Symbol: {symbol}\n")
                f.write(f"Initial Balance: ${self.portfolio.initial_balance:,.2f}\n")
                f.write("="*80 + "\n")
                
            # Haupt-Trading Loop
            while self.trading_active and self.iteration_count < self.max_iterations:
                self.iteration_count += 1
                self.performance_data['total_iterations'] += 1
                
                print(f"\n📊 ITERATION {self.iteration_count}/{self.max_iterations}")
                print("-" * 40)
                
                # 1. Hole Marktdaten (simuliert)
                market_data = self.get_market_data()
                
                # 2. Aktualisiere Preis in ML Engine
                self.ml_engine.update_current_price(market_data['price'])
                
                # 3. Generiere ML-Signal
                signal, confidence = self.ml_engine.generate_signal()
                self.performance_data['signals_generated'] += 1
                
                # 4. Prüfe SL/TP für offene Positionen
                closed_trades = self.portfolio.check_sltp_levels({
                    self.symbol: market_data['price']
                })
                
                # Verarbeite geschlossene Trades
                for trade in closed_trades:
                    self.process_closed_trade(trade)
                    
                # 5. Entscheide über neue Position
                trade_executed = False
                if confidence >= 65 and signal != "HOLD":
                    # Prüfe ob bereits maximale Positionen offen
                    if len(self.portfolio.positions) < 3:  # Max 3 Positionen
                        trade_executed = self.execute_trade(signal, confidence, market_data['price'])
                        
                # 6. Zeige Status
                self.display_status(market_data['price'], signal, confidence, trade_executed)
                
                # 7. Speichere Zwischenstand
                if self.iteration_count % 5 == 0:
                    self.save_session_state()
                    
                # 8. Kurze Pause
                time.sleep(2)
                
            # Session beenden
            self.end_trading_session()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Session durch Benutzer abgebrochen")
            self.end_trading_session()
        except Exception as e:
            print(f"\n❌ Fehler in Trading Session: {e}")
            traceback.print_exc()
            self.end_trading_session()
            
    def get_market_data(self) -> Dict:
        """
        Holt simulierte Marktdaten.
        
        Returns:
            Dict: Marktdaten
        """
        # Simulierte Preis-Bewegung
        base_price = 1.10000
        price_change = random.uniform(-0.001, 0.001)
        current_price = base_price + price_change
        
        spread = 0.0002  # 2 Pips
        bid = current_price - (spread / 2)
        ask = current_price + (spread / 2)
        
        return {
            'symbol': self.symbol,
            'bid': bid,
            'ask': ask,
            'price': current_price,  # Mid-Price
            'spread': spread,
            'timestamp': datetime.now().isoformat(),
            'volume': random.randint(100, 1000)
        }
        
    def execute_trade(self, signal: str, confidence: float, price: float) -> bool:
        """
        Führt einen Trade aus.
        
        Args:
            signal: BUY oder SELL
            confidence: Confidence Level
            price: Aktueller Preis
            
        Returns:
            bool: True wenn Trade ausgeführt wurde
        """
        try:
            print(f"\n🎯 TRADE AUSFÜHREN:")
            print(f"   Signal: {signal}")
            print(f"   Confidence: {confidence:.1f}%")
            print(f"   Preis: {price:.5f}")
            
            # Bestimme Entry-Preis (Bid für SELL, Ask für BUY)
            if signal == 'BUY':
                entry_price = price + 0.0001  # Simulierter Ask
            else:  # SELL
                entry_price = price - 0.0001  # Simulierter Bid
                
            # Berechne Position-Größe basierend auf Confidence
            base_volume = 0.01  # 0.01 Lots
            if confidence > 80:
                volume = base_volume * 2  # 0.02 Lots
            elif confidence > 70:
                volume = base_volume * 1.5  # 0.015 Lots
            else:
                volume = base_volume  # 0.01 Lots
                
            # Berechne SL/TP basierend auf Volatilität
            sl_pips = 30
            tp_pips = 60
            
            if confidence > 75:
                # Aggressiver bei hoher Confidence
                sl_pips = 20
                tp_pips = 40
                
            # Erstelle Position
            position_data = {
                'symbol': self.symbol,
                'type': signal,
                'entry_price': entry_price,
                'volume': volume,
                'sl_pips': sl_pips,
                'tp_pips': tp_pips,
                'confidence': confidence,
                'commission': volume * 0.5,  # Simulierte Commission
                'signal_source': 'ML_ENGINE'
            }
            
            # Öffne Position
            success = self.portfolio.open_position(position_data)
            
            if success:
                self.performance_data['trades_executed'] += 1
                self.performance_data['total_volume'] += volume
                self.performance_data['total_commission'] += position_data['commission']
                
                print(f"✅ Trade ausgeführt: {signal} {self.symbol} @ {entry_price:.5f}")
                print(f"   Volume: {volume} Lots, SL: {sl_pips}pips, TP: {tp_pips}pips")
                return True
            else:
                print(f"❌ Trade konnte nicht ausgeführt werden")
                return False
                
        except Exception as e:
            print(f"❌ Fehler bei Trade-Ausführung: {e}")
            traceback.print_exc()
            return False
            
    def process_closed_trade(self, trade: Dict):
        """
        Verarbeitet einen geschlossenen Trade.
        
        Args:
            trade: Trade-Daten
        """
        try:
            print(f"\n🔔 TRADE GESCHLOSSEN:")
            print(f"   ID: {trade.get('position_id', 'N/A')}")
            print(f"   Typ: {trade.get('type', 'N/A')}")
            print(f"   Entry: {trade.get('entry_price', 0):.5f}")
            print(f"   Exit: {trade.get('exit_price', 0):.5f}")
            print(f"   P&L: ${trade.get('net_profit', 0):.2f}")
            print(f"   Grund: {trade.get('close_reason', 'N/A')}")
            print(f"   Dauer: {trade.get('hold_time', 'N/A')}")
            
            # Update Performance Data
            self.performance_data['session_pnl'] += trade.get('net_profit', 0)
            
            # Update Peak/Low Balance
            current_balance = self.portfolio.balance
            self.performance_data['peak_balance'] = max(
                self.performance_data['peak_balance'],
                current_balance
            )
            self.performance_data['lowest_balance'] = min(
                self.performance_data['lowest_balance'],
                current_balance
            )
            
            # Berechne Drawdown
            peak = self.performance_data['peak_balance']
            current = current_balance
            if peak > 0:
                drawdown = ((peak - current) / peak) * 100
                self.performance_data['drawdown'] = max(
                    self.performance_data['drawdown'],
                    drawdown
                )
                
            # Speichere detaillierte Trade-History
            self.save_detailed_trade_history(trade)
            
        except Exception as e:
            print(f"❌ Fehler bei Trade-Verarbeitung: {e}")
            
    def save_detailed_trade_history(self, trade: Dict):
        """
        Speichert detaillierte Trade-History.
        """
        try:
            # CSV Datei
            if self.history_settings['save_to_csv']:
                csv_file = self.history_settings['csv_file']
                file_exists = os.path.isfile(csv_file)
                
                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    fieldnames = [
                        'session_id', 'trade_id', 'symbol', 'direction',
                        'entry_price', 'exit_price', 'entry_time', 'exit_time',
                        'hold_time', 'volume', 'profit', 'net_profit',
                        'commission', 'swap', 'sl_price', 'tp_price',
                        'close_reason', 'confidence', 'signal_source',
                        'balance_before', 'balance_after', 'equity_before', 'equity_after'
                    ]
                    
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                        
                    # Berechne zusätzliche Daten
                    trade_data = {
                        'session_id': self.performance_data['session_start'][:19].replace(':', ''),
                        'trade_id': trade.get('position_id', ''),
                        'symbol': trade.get('symbol', ''),
                        'direction': trade.get('type', ''),
                        'entry_price': trade.get('entry_price', 0),
                        'exit_price': trade.get('exit_price', 0),
                        'entry_time': trade.get('open_time', ''),
                        'exit_time': trade.get('close_time', ''),
                        'hold_time': trade.get('hold_time', ''),
                        'volume': trade.get('volume', 0),
                        'profit': trade.get('profit', 0),
                        'net_profit': trade.get('net_profit', 0),
                        'commission': trade.get('commission', 0),
                        'swap': trade.get('swap', 0),
                        'sl_price': trade.get('sl', 0),
                        'tp_price': trade.get('tp', 0),
                        'close_reason': trade.get('close_reason', ''),
                        'confidence': trade.get('confidence', 0),
                        'signal_source': trade.get('signal_source', 'ML'),
                        'balance_before': self.portfolio.balance - trade.get('net_profit', 0),
                        'balance_after': self.portfolio.balance,
                        'equity_before': self.portfolio.equity - trade.get('net_profit', 0),
                        'equity_after': self.portfolio.equity
                    }
                    
                    writer.writerow(trade_data)
                    
            # JSON Datei
            if self.history_settings['save_to_json']:
                json_file = self.history_settings['json_file']
                
                # Lade existierende Trades oder erstelle neue Liste
                trades_list = []
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            trades_list = json.load(f)
                    except:
                        trades_list = []
                        
                # Füge neuen Trade hinzu
                trades_list.append(trade)
                
                # Speichere zurück (nur letzten 100 Trades behalten)
                if len(trades_list) > 100:
                    trades_list = trades_list[-100:]
                    
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(trades_list, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Trade-History: {e}")
            
    def display_status(self, current_price: float, signal: str, confidence: float, trade_executed: bool = False):
        """
        Zeigt aktuellen Status an.
        """
        # Portfolio Summary
        portfolio_summary = self.portfolio.get_summary()
        
        print(f"\n📈 MARKT:")
        print(f"   Symbol: {self.symbol}")
        print(f"   Preis:  {current_price:.5f}")
        
        print(f"\n📡 SIGNAL:")
        print(f"   Typ:       {signal}")
        print(f"   Confidence: {confidence:.1f}%")
        
        print(f"\n💰 PORTFOLIO:")
        print(f"   Balance:   ${portfolio_summary['balance']:,.2f}")
        print(f"   Equity:    ${portfolio_summary['equity']:,.2f}")
        print(f"   P&L:       ${portfolio_summary['net_profit']:,.2f}")
        print(f"   Positionen: {portfolio_summary['open_positions']}")
        print(f"   Win Rate:  {portfolio_summary['win_rate']}%")
        
        print(f"\n📊 PERFORMANCE:")
        print(f"   Trades:    {self.performance_data['trades_executed']}")
        print(f"   Session P&L: ${self.performance_data['session_pnl']:.2f}")
        print(f"   Drawdown:  {self.performance_data['drawdown']:.1f}%")
        
        # Zeige Trade-Entscheidung
        if trade_executed:
            print(f"\n✅ TRADE AUSGEFÜHRT")
        elif confidence >= 65 and signal != "HOLD":
            print(f"\n⏸️  KEIN TRADE - Max. Positionen erreicht")
        elif confidence < 65:
            print(f"\n⏸️  KEIN TRADE - Confidence zu niedrig ({confidence:.1f}% < 65%)")
        else:
            print(f"\n⏸️  KEIN TRADE - HOLD Signal")
            
        print(f"\n⏳ FORTSCHRITT: {self.iteration_count}/{self.max_iterations}")
        print()  # Leere Zeile für bessere Lesbarkeit
        
    def save_session_state(self):
        """Speichert Session-Zustand."""
        try:
            state_file = f"data/paper_trading/session_state_{self.performance_data['session_start'][:10]}.json"
            
            state = {
                'session_data': self.performance_data,
                'portfolio_summary': self.portfolio.get_summary(),
                'iteration': self.iteration_count,
                'max_iterations': self.max_iterations,
                'last_update': datetime.now().isoformat()
            }
            
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Session-Zustand gespeichert: {state_file}")
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Session-Zustands: {e}")
            
    def end_trading_session(self):
        """Beendet die Trading Session."""
        print(f"\n{'='*80}")
        print(f"🛑 PAPER TRADING SESSION BEENDEN")
        print(f"{'='*80}")
        
        self.trading_active = False
        
        # Finale Performance-Berechnung
        portfolio_summary = self.portfolio.get_summary()
        
        print(f"\n📊 FINALE PERFORMANCE:")
        print(f"   Startkapital:   ${self.portfolio.initial_balance:,.2f}")
        print(f"   Endkapital:     ${portfolio_summary['balance']:,.2f}")
        print(f"   Gesamt-P&L:     ${portfolio_summary['net_profit']:,.2f}")
        print(f"   Return:         {(portfolio_summary['net_profit'] / self.portfolio.initial_balance) * 100:.2f}%")
        print(f"   Trades:         {portfolio_summary['total_trades']}")
        print(f"   Win Rate:       {portfolio_summary['win_rate']}%")
        print(f"   Profit Faktor:  {portfolio_summary['profit_factor']:.2f}")
        print(f"   Max Drawdown:   {self.performance_data['drawdown']:.1f}%")
        
        print(f"\n📈 STATISTIKEN:")
        print(f"   Iterationen:    {self.iteration_count}")
        print(f"   Signale:        {self.performance_data['signals_generated']}")
        print(f"   Ausgeführte Trades: {self.performance_data['trades_executed']}")
        print(f"   Total Volume:   {self.performance_data['total_volume']:.3f} Lots")
        print(f"   Commission:     ${self.performance_data['total_commission']:.2f}")
        
        print(f"\n⏱️  SESSION DETAILS:")
        print(f"   Start:          {self.performance_data['session_start'][11:19]}")
        print(f"   Ende:           {datetime.now().strftime('%H:%M:%S')}")
        
        # Speichere Session-Report
        self.save_session_report()
        
        print(f"\n💾 Daten gespeichert in:")
        print(f"   Trade History:  data/paper_trading/trade_history_detailed.csv")
        print(f"   Session Report: data/performance_reports/session_{self.performance_data['session_start'][:10]}.json")
        
        print(f"\n✅ Session erfolgreich beendet!")
        print(f"{'='*80}")
        
    def save_session_report(self):
        """Speichert detaillierten Session-Report."""
        try:
            report_file = f"data/performance_reports/session_{self.performance_data['session_start'][:10]}.json"
            
            # Erstelle Report
            report = {
                'session_info': {
                    'id': self.performance_data['session_start'][:19].replace(':', ''),
                    'start_time': self.performance_data['session_start'],
                    'end_time': datetime.now().isoformat(),
                    'duration_minutes': self.iteration_count * 2,  # Geschätzt
                    'iterations': self.iteration_count,
                    'symbol': self.symbol
                },
                'performance': {
                    'initial_balance': self.portfolio.initial_balance,
                    'final_balance': self.portfolio.balance,
                    'total_pnl': self.portfolio.get_summary()['net_profit'],
                    'return_percent': (self.portfolio.get_summary()['net_profit'] / self.portfolio.initial_balance) * 100,
                    'win_rate': self.portfolio.get_summary()['win_rate'],
                    'profit_factor': self.portfolio.get_summary()['profit_factor'],
                    'max_drawdown': self.performance_data['drawdown'],
                    'sharpe_ratio': self.calculate_sharpe_ratio(),
                    'total_trades': self.performance_data['trades_executed'],
                    'total_volume': self.performance_data['total_volume']
                },
                'ml_performance': {
                    'signals_generated': self.performance_data['signals_generated'],
                    'signals_executed': self.performance_data['trades_executed'],
                    'execution_rate': (self.performance_data['trades_executed'] / max(1, self.performance_data['signals_generated'])) * 100
                },
                'trade_summary': self.portfolio.get_summary(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Speichere Report
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            print(f"✅ Session-Report gespeichert: {report_file}")
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Session-Reports: {e}")
            
    def calculate_sharpe_ratio(self) -> float:
        """Berechnet Sharpe Ratio (vereinfacht)."""
        try:
            if len(self.portfolio.trade_history) < 2:
                return 0.0
                
            # Sammle Returns aus Trades
            returns = []
            for trade in self.portfolio.trade_history:
                if trade.get('profit', 0) != 0:
                    return_percent = (trade['profit'] / self.portfolio.initial_balance) * 100
                    returns.append(return_percent)
                    
            if len(returns) < 2:
                return 0.0
                
            import statistics
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns) if len(returns) > 1 else 0
            
            # Annahme: Risk-Free Rate = 0% für Paper Trading
            risk_free_rate = 0
            
            if std_return > 0:
                sharpe = (mean_return - risk_free_rate) / std_return
                return round(sharpe, 2)
            else:
                return 0.0
                
        except:
            return 0.0


def run_paper_trading_session(iterations: int = 5, symbol: str = "EURUSD"):
    """
    Startet eine Paper Trading Session.
    
    Args:
        iterations: Anzahl der Iterationen
        symbol: Trading-Symbol
    """
    bridge = PaperTradingBridge(initial_balance=10000.0)
    bridge.start_trading_session(iterations=iterations, symbol=symbol)
    return True


# Test-Funktion
def test_paper_bridge():
    """Testet die Paper Trading Bridge."""
    print("🧪 Teste Paper Trading Bridge...")
    
    # Kurztest mit 3 Iterationen
    run_paper_trading_session(iterations=3, symbol="EURUSD")
    
    return True


if __name__ == "__main__":
    test_paper_bridge()