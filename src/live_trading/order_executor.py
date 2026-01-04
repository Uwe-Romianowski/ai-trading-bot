# src/live_trading/order_executor.py
"""
MT5 Order Executor - Phase E Woche 2
Wandelt virtuelle Paper-Trading Orders in echte MT5 Orders um.
"""
import MetaTrader5 as mt5
from datetime import datetime
from typing import Dict, Optional, Tuple
import json
import time

class MT5OrderExecutor:
    """Klasse für die Ausführung von MT5 Orders."""
    
    def __init__(self, mt5_client):
        """
        Initialisiert den Order Executor.
        
        Args:
            mt5_client: Verbundener MT5LiveClient
        """
        self.mt5_client = mt5_client
        self.magic_number = 234000  # Magic number für unsere Orders
        self.order_counter = 1
        
    def _prepare_order_request(self, symbol: str, order_type: str, volume: float,
                              sl_price: Optional[float] = None,
                              tp_price: Optional[float] = None) -> Optional[Dict]:
        """Bereitet Order Request vor."""
        # Aktuellen Preis holen
        price_info = self.mt5_client.get_live_price(symbol)
        if not price_info:
            print(f"❌ Kein Live-Preis für {symbol} verfügbar")
            return None
        
        # Order Typ bestimmen
        if order_type.upper() == "BUY":
            mt5_order_type = mt5.ORDER_TYPE_BUY
            price = price_info['ask']  # Buy bei Ask-Preis
        elif order_type.upper() == "SELL":
            mt5_order_type = mt5.ORDER_TYPE_SELL
            price = price_info['bid']  # Sell bei Bid-Preis
        else:
            print(f"❌ Ungültiger Order Typ: {order_type}")
            return None
        
        # Order Request
        order_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": mt5_order_type,
            "price": price,
            "sl": sl_price if sl_price else 0.0,
            "tp": tp_price if tp_price else 0.0,
            "deviation": 20,  # Maximaler Preisabweichung in Punkten
            "magic": self.magic_number,
            "comment": f"AI-Bot {order_type} {volume}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        return order_request
    
    def execute_order(self, symbol: str, order_type: str, volume: float,
                     sl_pips: Optional[float] = None,
                     tp_pips: Optional[float] = None) -> Dict:
        """
        Führt eine Order aus.
        
        Args:
            symbol: Trading Symbol (z.B. 'EURUSD')
            order_type: 'BUY' oder 'SELL'
            volume: Lot Größe (z.B. 0.1)
            sl_pips: Stop-Loss in Pips
            tp_pips: Take-Profit in Pips
            
        Returns:
            Dict mit Order-Ergebnissen
        """
        print(f"\n📤 Versuche {order_type} Order für {symbol} {volume} Lots...")
        
        # Prüfe ob verbunden
        if not self.mt5_client.connected:
            print("❌ Nicht mit MT5 verbunden")
            return {"success": False, "error": "Nicht verbunden"}
        
        # Aktuellen Preis für SL/TP Berechnung
        price_info = self.mt5_client.get_live_price(symbol)
        if not price_info:
            return {"success": False, "error": "Kein Live-Preis"}
        
        # SL/TP Preise berechnen
        sl_price = None
        tp_price = None
        
        if sl_pips:
            if order_type.upper() == "BUY":
                sl_price = price_info['bid'] - (sl_pips / 10000)
            else:  # SELL
                sl_price = price_info['ask'] + (sl_pips / 10000)
        
        if tp_pips:
            if order_type.upper() == "BUY":
                tp_price = price_info['bid'] + (tp_pips / 10000)
            else:  # SELL
                tp_price = price_info['ask'] - (tp_pips / 10000)
        
        # Order Request vorbereiten
        order_request = self._prepare_order_request(
            symbol=symbol,
            order_type=order_type,
            volume=volume,
            sl_price=sl_price,
            tp_price=tp_price
        )
        
        if not order_request:
            return {"success": False, "error": "Order Request fehlgeschlagen"}
        
        # Order senden
        print(f"   Preis: {order_request['price']:.5f}")
        if sl_price:
            print(f"   Stop-Loss: {sl_price:.5f} ({sl_pips} pips)")
        if tp_price:
            print(f"   Take-Profit: {tp_price:.5f} ({tp_pips} pips)")
        
        try:
            result = mt5.order_send(order_request)
            
            print(f"\n📊 Order Ergebnis:")
            print(f"   Retcode: {result.retcode}")
            print(f"   Deal: {result.deal}")
            print(f"   Order ID: {result.order}")
            print(f"   Volume: {result.volume}")
            print(f"   Price: {result.price}")
            print(f"   Bid/Ask: {result.bid}/{result.ask}")
            print(f"   Comment: {result.comment}")
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Order erfolgreich ausgeführt!")
                
                # Position Details abrufen
                time.sleep(0.5)  # Kurz warten für Positionsupdate
                positions = mt5.positions_get(ticket=result.order)
                
                return {
                    "success": True,
                    "order_id": result.order,
                    "deal_id": result.deal,
                    "ticket": result.order,
                    "price": result.price,
                    "volume": result.volume,
                    "symbol": symbol,
                    "type": order_type,
                    "sl": sl_price,
                    "tp": tp_price,
                    "comment": result.comment,
                    "timestamp": datetime.now().isoformat(),
                    "position_found": len(positions) > 0 if positions else False
                }
            else:
                error_msg = self._get_error_message(result.retcode)
                print(f"❌ Order fehlgeschlagen: {error_msg}")
                
                return {
                    "success": False,
                    "error": f"Order fehlgeschlagen: {result.retcode}",
                    "error_message": error_msg,
                    "comment": result.comment
                }
                
        except Exception as e:
            print(f"❌ Exception bei Order Send: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_error_message(self, retcode: int) -> str:
        """Gibt eine menschenlesbare Fehlermeldung zurück."""
        error_messages = {
            10004: "Requote",
            10006: "Request abgelehnt",
            10007: "Request abgebrochen",
            10008: "Order Platzierung fehlgeschlagen",
            10009: "Order geändert fehlgeschlagen",
            10010: "Order storniert fehlgeschlagen",
            10011: "Order abgeschlossen fehlgeschlagen",
            10012: "Order suspendiert",
            10013: "Order geändert",
            10014: "Order storniert",
            10015: "Order ausstehend",
            10016: "Order abgeschlossen",
            10017: "Order teilweise erfüllt",
            10018: "Markt geschlossen",
            10019: "Nicht genug Geld",
            10020: "Preis geändert",
            10021: "Keine Preise",
            10022: "Broker nicht verfügbar",
            10023: "Handel deaktiviert",
            10024: "Account gesperrt",
            10025: "Ungültiges Konto",
            10026: "Timeout",
            10027: "Ungültiges Preis",
            10028: "Ungültiges Stop-Loss",
            10029: "Ungültiges Take-Profit",
            10030: "Ungültiges Volumen",
        }
        
        return error_messages.get(retcode, f"Unbekannter Fehler: {retcode}")
    
    def close_position(self, ticket: int) -> Dict:
        """Schließt eine offene Position."""
        print(f"\n🔒 Versuche Position {ticket} zu schließen...")
        
        if not self.mt5_client.connected:
            return {"success": False, "error": "Nicht verbunden"}
        
        # Position finden
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            print(f"❌ Position {ticket} nicht gefunden")
            return {"success": False, "error": "Position nicht gefunden"}
        
        position = positions[0]
        print(f"   Gefundene Position: {position.symbol} {position.volume} Lots")
        
        # Aktuellen Preis für Closing
        price_info = self.mt5_client.get_live_price(position.symbol)
        if not price_info:
            return {"success": False, "error": "Kein Live-Preis"}
        
        # Gegenteilige Order zum Schließen
        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            close_price = price_info['bid']
            print(f"   Closing BUY Position @ {close_price:.5f}")
        else:  # SELL Position
            close_type = mt5.ORDER_TYPE_BUY
            close_price = price_info['ask']
            print(f"   Closing SELL Position @ {close_price:.5f}")
        
        # Close Request
        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": self.magic_number,
            "comment": f"AI-Bot CLOSE {position.ticket}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        try:
            result = mt5.order_send(close_request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Position {ticket} erfolgreich geschlossen!")
                print(f"   Closing Price: {result.price:.5f}")
                print(f"   P&L berechnen...")
                
                # P&L berechnen
                if position.type == mt5.POSITION_TYPE_BUY:
                    pnl = (close_price - position.price_open) * position.volume * 100000
                else:
                    pnl = (position.price_open - close_price) * position.volume * 100000
                
                return {
                    "success": True,
                    "closed_ticket": ticket,
                    "close_price": result.price,
                    "open_price": position.price_open,
                    "volume": position.volume,
                    "pnl": pnl,
                    "pnl_pips": pnl / (position.volume * 10),
                    "symbol": position.symbol,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_msg = self._get_error_message(result.retcode)
                print(f"❌ Closing fehlgeschlagen: {error_msg}")
                return {
                    "success": False,
                    "error": f"Closing fehlgeschlagen: {result.retcode}",
                    "error_message": error_msg
                }
                
        except Exception as e:
            print(f"❌ Exception beim Closing: {e}")
            return {"success": False, "error": str(e)}
    
    def get_open_positions(self) -> list:
        """Gibt alle offenen Positionen zurück."""
        if not self.mt5_client.connected:
            return []
        
        positions = mt5.positions_get()
        if positions is None:
            return []
        
        result = []
        for position in positions:
            pos_dict = position._asdict()
            
            # Aktuellen P&L berechnen
            price_info = self.mt5_client.get_live_price(position.symbol)
            if price_info:
                if position.type == mt5.POSITION_TYPE_BUY:
                    current_profit = (price_info['bid'] - position.price_open) * position.volume * 100000
                else:
                    current_profit = (position.price_open - price_info['ask']) * position.volume * 100000
                
                pos_dict['current_profit'] = current_profit
                pos_dict['current_profit_pips'] = current_profit / (position.volume * 10)
            
            result.append(pos_dict)
        
        return result
    
    def test_order_execution(self) -> bool:
        """Testet die Order Execution mit einer Mini-Order."""
        print("\n" + "="*60)
        print("🧪 TEST ORDER EXECUTION - PHASE E WOCHE 2")
        print("="*60)
        
        # Sicherstellen, dass wir verbunden sind
        if not self.mt5_client.connected:
            if not self.mt5_client.connect():
                print("❌ Konnte keine Verbindung herstellen")
                return False
        
        try:
            # 1. Test: Kleine BUY Order (0.01 Lots - Minimum)
            print("\n1️⃣ Teste BUY Order (0.01 Lots EURUSD)...")
            result = self.execute_order(
                symbol="EURUSD",
                order_type="BUY",
                volume=0.01,
                sl_pips=20,
                tp_pips=40
            )
            
            if result.get("success"):
                print("✅ BUY Order Test erfolgreich!")
                ticket = result.get("order_id")
                
                # Kurz warten
                print("⏱️  Warte 3 Sekunden...")
                time.sleep(3)
                
                # 2. Test: Position schließen
                print("\n2️⃣ Teste Position Closing...")
                close_result = self.close_position(ticket)
                
                if close_result.get("success"):
                    print("✅ Closing Test erfolgreich!")
                    pnl = close_result.get("pnl", 0)
                    print(f"   P&L: ${pnl:.2f}")
                    
                    # 3. Test: Positions-Liste
                    print("\n3️⃣ Teste Positions-Liste...")
                    positions = self.get_open_positions()
                    print(f"   Offene Positionen: {len(positions)}")
                    
                    return True
                else:
                    print(f"❌ Closing Test fehlgeschlagen: {close_result.get('error')}")
                    return False
            else:
                print(f"❌ BUY Order Test fehlgeschlagen: {result.get('error')}")
                
                # Fallback: Vielleicht ist Market geschlossen oder andere Limits
                print("\n💡 Mögliche Gründe:")
                print("   - Market ist geschlossen (Wochenende)")
                print("   - Nicht genügend Margin")
                print("   - Broker Limits (Minimum Trade Size)")
                print("   - Account nicht für Live-Trading freigeschaltet")
                
                return False
                
        except Exception as e:
            print(f"❌ Unerwarteter Fehler: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            print("\n" + "="*60)
            print("🧪 ORDER EXECUTION TEST ABGESCHLOSSEN")