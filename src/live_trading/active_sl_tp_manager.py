"""
Aktiver SL/TP Manager für MT5 Demo Accounts
Ersetzt nicht-funktionierende MT5 SL/TP mit aktivem Monitoring
"""

import MetaTrader5 as mt5
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import json
import os

class ActiveSLTPManager:
    """Manager für aktives SL/TP Monitoring."""
    
    def __init__(self, mt5_client):
        self.mt5_client = mt5_client
        self.monitored_positions = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval = 1.0  # Sekunden zwischen Checks
        self.sl_tp_history = []
        
        print("✅ Aktiver SL/TP Manager initialisiert")
    
    def start_monitoring(self):
        """Startet den Monitoring-Thread."""
        if self.monitoring_active:
            print("⚠️  Monitoring bereits aktiv")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("🚀 Aktives SL/TP Monitoring gestartet")
    
    def stop_monitoring(self):
        """Stoppt das Monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("⏹️  Aktives SL/TP Monitoring gestoppt")
    
    def add_position_to_monitor(self, ticket: int, symbol: str, position_type: str,
                               entry_price: float, sl_price: float, tp_price: float,
                               volume: float):
        """Fügt eine Position zum Monitoring hinzu."""
        self.monitored_positions[ticket] = {
            'ticket': ticket,
            'symbol': symbol,
            'type': position_type,  # 'BUY' or 'SELL'
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'volume': volume,
            'added_time': datetime.now().isoformat(),
            'status': 'MONITORING',
            'last_check': datetime.now()
        }
        
        print(f"🔔 Position {ticket} zum SL/TP Monitoring hinzugefügt")
        print(f"   SL: {sl_price:.5f}, TP: {tp_price:.5f}")
        
        # Starte Monitoring wenn nicht aktiv
        if not self.monitoring_active:
            self.start_monitoring()
    
    def _monitoring_loop(self):
        """Haupt-Monitoring Loop."""
        print(f"🔍 Monitoring Loop gestartet (Check alle {self.check_interval}s)")
        
        while self.monitoring_active:
            try:
                self._check_all_positions()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"⚠️  Fehler im Monitoring Loop: {e}")
                time.sleep(5.0)
    
    def _check_all_positions(self):
        """Prüft alle überwachten Positionen."""
        if not self.monitored_positions or not self.mt5_client.connected:
            return
        
        current_time = datetime.now()
        
        for ticket, position in list(self.monitored_positions.items()):
            if position['status'] != 'MONITORING':
                continue
            
            # Prüfe ob Position noch existiert
            mt5_position = self._get_mt5_position(ticket)
            if not mt5_position:
                position['status'] = 'CLOSED_EXTERNALLY'
                position['closed_time'] = current_time.isoformat()
                print(f"   ℹ️  Position {ticket} extern geschlossen")
                continue
            
            # Hole aktuellen Preis
            tick = mt5.symbol_info_tick(position['symbol'])
            if not tick:
                continue
            
            if position['type'] == 'BUY':
                current_price = tick.bid
                sl_condition = current_price <= position['sl_price'] if position['sl_price'] > 0 else False
                tp_condition = current_price >= position['tp_price'] if position['tp_price'] > 0 else False
            else:  # SELL
                current_price = tick.ask
                sl_condition = current_price >= position['sl_price'] if position['sl_price'] > 0 else False
                tp_condition = current_price <= position['tp_price'] if position['tp_price'] > 0 else False
            
            # Debug Info (nur alle 10 Checks oder bei nah dran)
            position['last_check'] = current_time
            check_count = position.get('check_count', 0) + 1
            position['check_count'] = check_count
            
            if check_count % 10 == 0 or abs(current_price - position['sl_price']) * 10000 < 5:
                sl_distance = abs(current_price - position['sl_price']) * 10000 if position['sl_price'] > 0 else 0
                tp_distance = abs(current_price - position['tp_price']) * 10000 if position['tp_price'] > 0 else 0
                print(f"   📊 Pos {ticket}: {current_price:.5f} | "
                      f"SL: {sl_distance:.1f}p | TP: {tp_distance:.1f}p")
            
            # Prüfe SL/TP Bedingungen
            if sl_condition:
                self._close_for_sl_tp(ticket, position, current_price, 'SL')
            elif tp_condition:
                self._close_for_sl_tp(ticket, position, current_price, 'TP')
    
    def _close_for_sl_tp(self, ticket: int, position: Dict, close_price: float, reason: str):
        """Schließt eine Position wegen SL/TP."""
        print(f"   🎯 {reason} erreicht für Position {ticket}!")
        print(f"      Aktuell: {close_price:.5f}, Target: {position[f'{reason.lower()}_price']:.5f}")
        
        try:
            # Schließe Position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position['symbol'],
                "volume": position['volume'],
                "type": mt5.ORDER_TYPE_BUY if position['type'] == 'SELL' else mt5.ORDER_TYPE_SELL,
                "position": ticket,
                "price": close_price,
                "deviation": 10,
                "magic": 234000,
                "comment": f"AI-Bot {reason} Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                position['status'] = f'CLOSED_{reason}'
                position['closed_time'] = datetime.now().isoformat()
                position['close_price'] = close_price
                position['close_reason'] = reason
                
                # P&L berechnen
                if position['type'] == 'BUY':
                    pnl = (close_price - position['entry_price']) * position['volume'] * 100000
                else:
                    pnl = (position['entry_price'] - close_price) * position['volume'] * 100000
                
                position['pnl'] = pnl
                
                self.sl_tp_history.append({
                    'ticket': ticket,
                    'symbol': position['symbol'],
                    'reason': reason,
                    'entry_price': position['entry_price'],
                    'close_price': close_price,
                    'pnl': pnl,
                    'time': datetime.now().isoformat()
                })
                
                print(f"      ✅ Auto-geschlossen durch {reason}, P&L: ${pnl:.2f}")
                
                # Speichere History
                self._save_history()
                
            else:
                print(f"      ❌ Auto-Close fehlgeschlagen: {result.comment}")
                
        except Exception as e:
            print(f"      ❌ Fehler beim Schließen: {e}")
    
    def _get_mt5_position(self, ticket: int):
        """Holt eine Position von MT5."""
        positions = mt5.positions_get(ticket=ticket)
        return positions[0] if positions else None
    
    def remove_position(self, ticket: int):
        """Entfernt eine Position vom Monitoring."""
        if ticket in self.monitored_positions:
            del self.monitored_positions[ticket]
            print(f"   ℹ️  Position {ticket} vom Monitoring entfernt")
    
    def get_monitored_positions(self) -> List[Dict]:
        """Gibt alle überwachten Positionen zurück."""
        return list(self.monitored_positions.values())
    
    def get_active_positions(self) -> List[Dict]:
        """Gibt alle aktiven Positionen zurück."""
        return [p for p in self.monitored_positions.values() if p['status'] == 'MONITORING']
    
    def _save_history(self):
        """Speichert SL/TP History."""
        os.makedirs('data/sl_tp_history', exist_ok=True)
        filename = f"data/sl_tp_history/history_{datetime.now().strftime('%Y%m%d')}.json"
        
        data = {
            'history': self.sl_tp_history,
            'last_updated': datetime.now().isoformat(),
            'total_hits': len(self.sl_tp_history),
            'sl_hits': len([h for h in self.sl_tp_history if h['reason'] == 'SL']),
            'tp_hits': len([h for h in self.sl_tp_history if h['reason'] == 'TP'])
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def print_stats(self):
        """Gibt SL/TP Statistiken aus."""
        total = len(self.sl_tp_history)
        sl_hits = len([h for h in self.sl_tp_history if h['reason'] == 'SL'])
        tp_hits = len([h for h in self.sl_tp_history if h['reason'] == 'TP'])
        
        print("\n" + "="*60)
        print("📊 AKTIVES SL/TP MONITORING STATISTIKEN")
        print("="*60)
        print(f"Total SL/TP Hits: {total}")
        print(f"SL Hits: {sl_hits}")
        print(f"TP Hits: {tp_hits}")
        print(f"Aktuell überwachte Positionen: {len(self.get_active_positions())}")
        
        if total > 0:
            total_pnl = sum(h['pnl'] for h in self.sl_tp_history)
            avg_pnl = total_pnl / total
            print(f"Total P&L durch SL/TP: ${total_pnl:.2f}")
            print(f"Avg P&L pro Hit: ${avg_pnl:.2f}")
        
        print("="*60)

def test_active_sl_tp():
    """Testet den aktiven SL/TP Manager."""
    print("🧪 TEST AKTIVER SL/TP MANAGER")
    print("-" * 50)
    
    try:
        from mt5_client import MT5LiveClient
        
        # MT5 Client
        client = MT5LiveClient()
        if not client.connect():
            print("❌ MT5 Verbindung fehlgeschlagen")
            return
        
        # SL/TP Manager
        sltp_manager = ActiveSLTPManager(client)
        
        # Test BUY Order mit engem SL/TP
        print("\n1️⃣ Erstelle BUY Order mit engem SL/TP...")
        
        # Hole aktuellen Preis
        tick = mt5.symbol_info_tick("EURUSD")
        if not tick:
            print("❌ Kein Preis verfügbar")
            return
        
        entry_price = tick.ask
        sl_price = entry_price - (3 / 10000)  # 3 pips SL
        tp_price = entry_price + (5 / 10000)  # 5 pips TP
        
        # Order Request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": "EURUSD",
            "volume": 0.01,
            "type": mt5.ORDER_TYPE_BUY,
            "price": entry_price,
            "sl": 0.0,  # WICHTIG: Kein MT5 SL setzen - wir machen aktives Monitoring!
            "tp": 0.0,  # WICHTIG: Kein MT5 TP setzen
            "deviation": 10,
            "magic": 234001,
            "comment": "SL/TP Test Order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"❌ Order fehlgeschlagen: {result.comment}")
            return
        
        ticket = result.order
        print(f"✅ BUY Order erstellt: {ticket}")
        print(f"   Entry: {entry_price:.5f}")
        print(f"   Unser SL: {sl_price:.5f} (3 pips)")
        print(f"   Unser TP: {tp_price:.5f} (5 pips)")
        
        # Zum Monitoring hinzufügen
        sltp_manager.add_position_to_monitor(
            ticket=ticket,
            symbol="EURUSD",
            position_type="BUY",
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            volume=0.01
        )
        
        # Monitoring für 30 Sekunden laufen lassen
        print("\n2️⃣ Aktives Monitoring für 30 Sekunden...")
        print("   Der Kurs muss um 5 pips steigen (TP) oder 3 pips fallen (SL)")
        
        for i in range(30):
            time.sleep(1)
            positions = sltp_manager.get_active_positions()
            if not positions:
                print(f"   ✅ Position wurde durch unseren SL/TP Manager geschlossen!")
                break
            
            if i % 5 == 0:
                print(f"   Check {i+1}/30 - Noch {len(positions)} Position(en) aktiv")
        else:
            print("   ⏱️  Monitoring Zeit abgelaufen")
        
        # Position schließen falls noch offen
        positions = mt5.positions_get()
        if positions:
            print("\n3️⃣ Manuelles Schließen...")
            for pos in positions:
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": pos.symbol,
                    "volume": pos.volume,
                    "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": pos.ticket,
                    "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == 0 else mt5.symbol_info_tick(pos.symbol).ask,
                    "deviation": 10,
                    "magic": 234001,
                    "comment": "Test Ende",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                mt5.order_send(close_request)
        
        # Statistiken anzeigen
        sltp_manager.print_stats()
        sltp_manager.stop_monitoring()
        
        print("\n✅ Test abgeschlossen!")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.shutdown()

if __name__ == "__main__":
    test_active_sl_tp()