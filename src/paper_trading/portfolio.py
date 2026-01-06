"""
PORTFOLIO MANAGEMENT v4.2 - MIT AUTO-SL/TP
"""

import os
import json
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback


class Portfolio:
    """
    Portfolio Management mit Auto-SL/TP und Trade-History.
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        """Initialisiert das Portfolio."""
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self.positions = []  # Aktive Positionen
        self.trade_history = []  # Abgeschlossene Trades
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # SL/TP Einstellungen
        self.sltp_settings = {
            'default_sl_pips': 30,
            'default_tp_pips': 60,
            'trailing_stop_enabled': False,
            'trailing_stop_distance': 20,
            'break_even_enabled': True,
            'break_even_at': 20
        }
        
        # Dateien für Persistenz
        self.trade_history_file = "data/paper_trading/trade_history.csv"
        self.portfolio_state_file = "data/paper_trading/portfolio_state.json"
        
        # Lade gespeicherte Daten wenn vorhanden
        self.load_portfolio_state()
        
    def open_position(self, position_data: Dict) -> bool:
        """
        Öffnet eine neue Position.
        
        Args:
            position_data: Position-Daten
            
        Returns:
            bool: True wenn erfolgreich
        """
        try:
            # Validiere Position-Daten
            required_fields = ['symbol', 'type', 'entry_price', 'volume']
            for field in required_fields:
                if field not in position_data:
                    print(f"❌ Fehlendes Feld: {field}")
                    return False
                    
            # Berechne Stop Loss und Take Profit falls nicht angegeben
            if 'sl' not in position_data:
                sl_pips = position_data.get('sl_pips', self.sltp_settings['default_sl_pips'])
                if position_data['type'] == 'BUY':
                    position_data['sl'] = position_data['entry_price'] - (sl_pips * 0.0001)
                else:  # SELL
                    position_data['sl'] = position_data['entry_price'] + (sl_pips * 0.0001)
                    
            if 'tp' not in position_data:
                tp_pips = position_data.get('tp_pips', self.sltp_settings['default_tp_pips'])
                if position_data['type'] == 'BUY':
                    position_data['tp'] = position_data['entry_price'] + (tp_pips * 0.0001)
                else:  # SELL
                    position_data['tp'] = position_data['entry_price'] - (tp_pips * 0.0001)
                    
            # Füge Metadaten hinzu
            position_data['id'] = f"pos_{len(self.positions)+1:04d}"
            position_data['open_time'] = datetime.now().isoformat()
            position_data['status'] = 'OPEN'
            position_data['commission'] = position_data.get('commission', 0.0)
            position_data['swap'] = 0.0
            
            # Berechne Margin (vereinfacht)
            position_value = position_data['volume'] * 100000  # 1 Lot = 100,000
            leverage = 30  # Angenommener Hebel
            position_data['margin'] = position_value / leverage
            
            # Prüfe ob genug Margin verfügbar
            if position_data['margin'] > self.equity * 0.8:  # Max 80% Equity als Margin
                print(f"❌ Nicht genug Margin verfügbar")
                return False
                
            # Füge Position hinzu
            self.positions.append(position_data)
            
            # Aktualisiere Portfolio
            self.update_portfolio_values()
            
            # Speichere Zustand
            self.save_portfolio_state()
            
            print(f"✅ Position eröffnet: {position_data['id']} "
                  f"{position_data['type']} {position_data['symbol']} "
                  f"@{position_data['entry_price']:.5f}")
                  
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Öffnen der Position: {e}")
            traceback.print_exc()
            return False
            
    def close_position(self, position_id: str, exit_price: float, reason: str = "MANUAL") -> Dict:
        """
        Schließt eine Position.
        
        Args:
            position_id: ID der Position
            exit_price: Exit-Preis
            reason: Grund für Schließung
            
        Returns:
            Dict: Trade-Daten
        """
        try:
            # Finde Position
            position = None
            position_index = -1
            
            for i, pos in enumerate(self.positions):
                if pos['id'] == position_id:
                    position = pos
                    position_index = i
                    break
                    
            if position is None:
                print(f"❌ Position {position_id} nicht gefunden")
                return {}
                
            # Berechne Profit/Loss
            profit = self.calculate_position_profit(position, exit_price)
            
            # Erstelle Trade-Eintrag
            trade = {
                'position_id': position_id,
                'symbol': position['symbol'],
                'type': position['type'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'volume': position['volume'],
                'profit': profit,
                'open_time': position['open_time'],
                'close_time': datetime.now().isoformat(),
                'hold_time': self.calculate_hold_time(position['open_time']),
                'sl': position.get('sl', 0),
                'tp': position.get('tp', 0),
                'sl_hit': position.get('sl_hit', False),
                'tp_hit': position.get('tp_hit', False),
                'close_reason': reason,
                'commission': position.get('commission', 0.0),
                'swap': position.get('swap', 0.0),
                'net_profit': profit - position.get('commission', 0.0) + position.get('swap', 0.0)
            }
            
            # Entferne Position
            self.positions.pop(position_index)
            
            # Aktualisiere Balance
            self.balance += trade['net_profit']
            
            # Füge zu Trade-History hinzu
            self.trade_history.append(trade)
            
            # Aktualisiere Performance-Statistiken
            self.update_performance_stats(trade)
            
            # Aktualisiere Portfolio
            self.update_portfolio_values()
            
            # Speichere Trade-History
            self.save_trade_history(trade)
            
            # Speichere Portfolio-Zustand
            self.save_portfolio_state()
            
            print(f"✅ Position geschlossen: {position_id} "
                  f"Profit: ${trade['net_profit']:.2f} "
                  f"Reason: {reason}")
                  
            return trade
            
        except Exception as e:
            print(f"❌ Fehler beim Schließen der Position: {e}")
            traceback.print_exc()
            return {}
            
    def check_sltp_levels(self, market_prices: Dict[str, float]) -> List[Dict]:
        """
        Prüft alle Positionen auf SL/TP Erreichung.
        
        Args:
            market_prices: Dict mit Symbol: Preis
            
        Returns:
            List: Geschlossene Positionen
        """
        closed_positions = []
        
        try:
            for position in self.positions[:]:  # Kopie für sicheres Iterieren
                symbol = position['symbol']
                current_price = market_prices.get(symbol)
                
                if current_price is None:
                    continue
                    
                # Prüfe SL/TP
                sl_hit = False
                tp_hit = False
                close_reason = ""
                
                if position['type'] == 'BUY':
                    if current_price <= position.get('sl', 0):
                        sl_hit = True
                        close_reason = "SL_HIT"
                    elif current_price >= position.get('tp', 0):
                        tp_hit = True
                        close_reason = "TP_HIT"
                else:  # SELL
                    if current_price >= position.get('sl', 0):
                        sl_hit = True
                        close_reason = "SL_HIT"
                    elif current_price <= position.get('tp', 0):
                        tp_hit = True
                        close_reason = "TP_HIT"
                        
                # Prüfe Trailing Stop
                if self.sltp_settings['trailing_stop_enabled']:
                    if self.check_trailing_stop(position, current_price):
                        close_reason = "TRAILING_STOP"
                        sl_hit = True
                        
                # Prüfe Break Even
                if self.sltp_settings['break_even_enabled']:
                    self.check_break_even(position, current_price)
                        
                # Schließe Position wenn SL/TP erreicht
                if sl_hit or tp_hit:
                    position['sl_hit'] = sl_hit
                    position['tp_hit'] = tp_hit
                    
                    trade = self.close_position(
                        position['id'],
                        current_price,
                        close_reason
                    )
                    
                    if trade:
                        closed_positions.append(trade)
                        
        except Exception as e:
            print(f"❌ Fehler bei SL/TP Prüfung: {e}")
            traceback.print_exc()
            
        return closed_positions
        
    def check_trailing_stop(self, position: Dict, current_price: float) -> bool:
        """
        Prüft und aktualisiert Trailing Stop.
        
        Args:
            position: Position-Daten
            current_price: Aktueller Preis
            
        Returns:
            bool: True wenn Trailing Stop erreicht
        """
        try:
            if 'trailing_stop' not in position:
                # Initialisiere Trailing Stop
                if position['type'] == 'BUY':
                    position['trailing_stop'] = position['entry_price'] - (
                        self.sltp_settings['trailing_stop_distance'] * 0.0001
                    )
                    position['highest_price'] = position['entry_price']
                else:  # SELL
                    position['trailing_stop'] = position['entry_price'] + (
                        self.sltp_settings['trailing_stop_distance'] * 0.0001
                    )
                    position['lowest_price'] = position['entry_price']
                    
            # Update Trailing Stop für BUY
            if position['type'] == 'BUY':
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    new_stop = current_price - (
                        self.sltp_settings['trailing_stop_distance'] * 0.0001
                    )
                    if new_stop > position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                        # print(f"⚠️  Trailing Stop angehoben auf: {new_stop:.5f}")
                        
                # Prüfe ob Preis unter Trailing Stop gefallen
                if current_price <= position['trailing_stop']:
                    return True
                    
            # Update Trailing Stop für SELL
            else:
                if current_price < position['lowest_price']:
                    position['lowest_price'] = current_price
                    new_stop = current_price + (
                        self.sltp_settings['trailing_stop_distance'] * 0.0001
                    )
                    if new_stop < position['trailing_stop']:
                        position['trailing_stop'] = new_stop
                        # print(f"⚠️  Trailing Stop gesenkt auf: {new_stop:.5f}")
                        
                # Prüfe ob Preis über Trailing Stop gestiegen
                if current_price >= position['trailing_stop']:
                    return True
                    
        except Exception as e:
            print(f"❌ Fehler bei Trailing Stop: {e}")
            
        return False
        
    def check_break_even(self, position: Dict, current_price: float):
        """
        Bewegt SL auf Break Even wenn Gewinn-Schwelle erreicht.
        """
        try:
            if 'break_even_moved' in position and position['break_even_moved']:
                return
                
            profit_pips = self.calculate_pip_profit(position, current_price)
            break_even_pips = self.sltp_settings['break_even_at']
            
            if profit_pips >= break_even_pips:
                # Bewege SL auf Entry Price (Break Even)
                if position['type'] == 'BUY':
                    position['sl'] = position['entry_price']
                else:  # SELL
                    position['sl'] = position['entry_price']
                    
                position['break_even_moved'] = True
                print(f"⚠️  Break Even aktiviert für Position {position['id']}")
                
        except Exception as e:
            print(f"❌ Fehler bei Break Even: {e}")
            
    def calculate_position_profit(self, position: Dict, exit_price: float) -> float:
        """
        Berechnet Profit/Loss für eine Position.
        """
        try:
            entry_price = position['entry_price']
            volume = position['volume']
            
            if position['type'] == 'BUY':
                price_diff = exit_price - entry_price
            else:  # SELL
                price_diff = entry_price - exit_price
                
            # Standard Forex Berechnung
            pip_size = 0.0001
            if 'JPY' in position['symbol']:
                pip_size = 0.01
                
            profit_pips = price_diff / pip_size
            
            # Pip-Wert für Standard Paare
            pip_value_per_lot = 10  # USD für EURUSD
            
            profit = profit_pips * pip_value_per_lot * volume
            
            return round(profit, 2)
            
        except Exception as e:
            print(f"❌ Fehler in Profit-Berechnung: {e}")
            return 0.0
            
    def calculate_pip_profit(self, position: Dict, current_price: float) -> float:
        """
        Berechnet Profit in Pips.
        """
        try:
            if position['type'] == 'BUY':
                price_diff = current_price - position['entry_price']
            else:  # SELL
                price_diff = position['entry_price'] - current_price
                
            pip_size = 0.0001
            if 'JPY' in position['symbol']:
                pip_size = 0.01
                
            return price_diff / pip_size
            
        except:
            return 0.0
            
    def calculate_hold_time(self, open_time_str: str) -> str:
        """
        Berechnet die Haltedauer.
        """
        try:
            open_time = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
            close_time = datetime.now()
            duration = close_time - open_time
            
            # Formatieren als Stunden:Minuten:Sekunden
            total_seconds = int(duration.total_seconds())
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
        except:
            return "N/A"
            
    def update_portfolio_values(self):
        """Aktualisiert Portfolio-Werte."""
        try:
            # Berechne aktuelles Equity
            self.equity = self.balance
            
            # Füge unrealisierten Profit hinzu
            # (wird in Verbindung mit Engine berechnet)
            
        except Exception as e:
            print(f"❌ Fehler bei Portfolio-Update: {e}")
            
    def update_performance_stats(self, trade: Dict):
        """Aktualisiert Performance-Statistiken."""
        try:
            profit = trade.get('net_profit', 0)
            
            self.performance_stats['total_trades'] += 1
            
            if profit > 0:
                self.performance_stats['winning_trades'] += 1
                self.performance_stats['total_profit'] += profit
                self.performance_stats['largest_win'] = max(
                    self.performance_stats['largest_win'],
                    profit
                )
            else:
                self.performance_stats['losing_trades'] += 1
                self.performance_stats['total_loss'] += abs(profit)
                self.performance_stats['largest_loss'] = min(
                    self.performance_stats['largest_loss'],
                    profit
                )
                
            # Berechne Win Rate
            total = self.performance_stats['winning_trades'] + self.performance_stats['losing_trades']
            if total > 0:
                self.performance_stats['win_rate'] = (
                    self.performance_stats['winning_trades'] / total
                ) * 100
                
            # Berechne Profit Factor
            if self.performance_stats['total_loss'] > 0:
                self.performance_stats['profit_factor'] = (
                    self.performance_stats['total_profit'] / self.performance_stats['total_loss']
                )
                
        except Exception as e:
            print(f"❌ Fehler bei Stats-Update: {e}")
            
    def save_trade_history(self, trade: Dict):
        """Speichert Trade in CSV Datei."""
        try:
            # Erstelle Verzeichnis falls nicht existiert
            os.makedirs(os.path.dirname(self.trade_history_file), exist_ok=True)
            
            file_exists = os.path.isfile(self.trade_history_file)
            
            with open(self.trade_history_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'position_id', 'symbol', 'type',
                    'entry_price', 'exit_price', 'volume', 'profit',
                    'net_profit', 'hold_time', 'close_reason',
                    'sl', 'tp', 'sl_hit', 'tp_hit',
                    'commission', 'swap', 'open_time', 'close_time'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                    
                # Prepare trade data for CSV
                csv_trade = {
                    'timestamp': datetime.now().isoformat(),
                    'position_id': trade.get('position_id', ''),
                    'symbol': trade.get('symbol', ''),
                    'type': trade.get('type', ''),
                    'entry_price': trade.get('entry_price', 0),
                    'exit_price': trade.get('exit_price', 0),
                    'volume': trade.get('volume', 0),
                    'profit': trade.get('profit', 0),
                    'net_profit': trade.get('net_profit', 0),
                    'hold_time': trade.get('hold_time', ''),
                    'close_reason': trade.get('close_reason', ''),
                    'sl': trade.get('sl', 0),
                    'tp': trade.get('tp', 0),
                    'sl_hit': trade.get('sl_hit', False),
                    'tp_hit': trade.get('tp_hit', False),
                    'commission': trade.get('commission', 0),
                    'swap': trade.get('swap', 0),
                    'open_time': trade.get('open_time', ''),
                    'close_time': trade.get('close_time', '')
                }
                
                writer.writerow(csv_trade)
                
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Trade-History: {e}")
            
    def save_portfolio_state(self):
        """Speichert Portfolio-Zustand."""
        try:
            state = {
                'balance': self.balance,
                'equity': self.equity,
                'positions': self.positions,
                'performance_stats': self.performance_stats,
                'last_updated': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(self.portfolio_state_file), exist_ok=True)
            
            with open(self.portfolio_state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Portfolio-Zustands: {e}")
            
    def load_portfolio_state(self):
        """Lädt Portfolio-Zustand."""
        try:
            if os.path.exists(self.portfolio_state_file):
                with open(self.portfolio_state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                self.balance = state.get('balance', self.balance)
                self.equity = state.get('equity', self.equity)
                self.positions = state.get('positions', [])
                self.performance_stats = state.get('performance_stats', self.performance_stats)
                
                print(f"✅ Portfolio-Zustand geladen: Balance ${self.balance:.2f}")
                
        except Exception as e:
            print(f"❌ Fehler beim Laden des Portfolio-Zustands: {e}")
            
    def get_summary(self) -> Dict:
        """Gibt Portfolio-Zusammenfassung zurück."""
        return {
            'balance': round(self.balance, 2),
            'equity': round(self.equity, 2),
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'winning_trades': self.performance_stats['winning_trades'],
            'losing_trades': self.performance_stats['losing_trades'],
            'win_rate': round(self.performance_stats['win_rate'], 2),
            'total_profit': round(self.performance_stats['total_profit'], 2),
            'total_loss': round(self.performance_stats['total_loss'], 2),
            'net_profit': round(self.performance_stats['total_profit'] - self.performance_stats['total_loss'], 2),
            'profit_factor': round(self.performance_stats['profit_factor'], 2)
        }
        
    def clear_portfolio(self):
        """Setzt Portfolio zurück."""
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.positions = []
        self.trade_history = []
        self.performance_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Lösche gespeicherte Dateien
        try:
            if os.path.exists(self.portfolio_state_file):
                os.remove(self.portfolio_state_file)
            print("✅ Portfolio zurückgesetzt")
        except:
            pass


# Test-Funktion
def test_portfolio():
    """Testet die Portfolio-Funktionalität."""
    print("🧪 Teste Portfolio mit Auto-SL/TP...")
    
    portfolio = Portfolio(initial_balance=10000.0)
    
    # Öffne Test-Positionen
    portfolio.open_position({
        'symbol': 'EURUSD',
        'type': 'BUY',
        'entry_price': 1.10000,
        'volume': 0.01,
        'sl_pips': 20,
        'tp_pips': 40
    })
    
    portfolio.open_position({
        'symbol': 'EURUSD',
        'type': 'SELL',
        'entry_price': 1.10100,
        'volume': 0.01,
        'sl_pips': 20,
        'tp_pips': 40
    })
    
    print(f"📊 Portfolio nach Eröffnung: {portfolio.get_summary()}")
    
    # Simuliere Preis-Updates und prüfe SL/TP
    test_prices = {
        'EURUSD': 1.09980  # Unter SL für BUY Position
    }
    
    closed = portfolio.check_sltp_levels(test_prices)
    print(f"🔔 Geschlossene Positionen: {len(closed)}")
    
    if closed:
        for trade in closed:
            print(f"   Trade: {trade['position_id']} - {trade['close_reason']} - Profit: ${trade['profit']:.2f}")
            
    print(f"📊 Finales Portfolio: {portfolio.get_summary()}")
    
    return True


if __name__ == "__main__":
    test_portfolio()