# src/live_trading/live_bridge.py
"""
Live Trading Bridge - Phase E Woche 2
Vereinfachte Version ohne Paper-Trading Importprobleme.
Führt Live-Orders im MT5 Demo Account aus.
"""
from .mt5_client import MT5LiveClient
from .order_executor import MT5OrderExecutor
import time
import random
from datetime import datetime

class LiveTradingBridge:
    """Vereinfachte Live Trading Bridge ohne Paper-Trading Abhängigkeiten."""
    
    def __init__(self):
        self.mt5_client = None
        self.order_executor = None
        self.initialized = False
        self.trade_history = []
        
    def initialize(self) -> bool:
        """Initialisiert Live-Trading Komponenten."""
        print("🔧 Initialisiere Live Trading Bridge...")
        
        try:
            # 1. MT5 Client
            self.mt5_client = MT5LiveClient()
            if not self.mt5_client.connect():
                print("❌ MT5 Verbindung fehlgeschlagen")
                return False
            
            # 2. Order Executor
            self.order_executor = MT5OrderExecutor(self.mt5_client)
            
            self.initialized = True
            print("✅ Live Trading Bridge initialisiert")
            return True
            
        except Exception as e:
            print(f"❌ Initialisierung fehlgeschlagen: {e}")
            return False
    
    def simulate_ml_signal(self):
        """Simuliert ein ML Signal für Testing."""
        # Zufällige Signal-Generierung
        signals = ["BUY", "SELL", "HOLD"]
        weights = [0.4, 0.4, 0.2]  # 40% BUY, 40% SELL, 20% HOLD
        
        signal = random.choices(signals, weights=weights)[0]
        
        # Confidence basierend auf Signal
        if signal == "HOLD":
            confidence = random.uniform(40, 60)
        else:
            confidence = random.uniform(65, 85)
        
        return signal, confidence
    
    def test_order_execution(self) -> bool:
        """Testet die Order Execution mit einer Mini-Order."""
        print("\n" + "="*60)
        print("🧪 TEST ORDER EXECUTION - PHASE E WOCHE 2")
        print("="*60)
        
        if not self.initialized:
            if not self.initialize():
                return False
        
        try:
            # 1. Aktuelle Marktinfo anzeigen
            price_info = self.mt5_client.get_live_price("EURUSD")
            if price_info:
                print(f"📊 Aktueller EURUSD Preis:")
                print(f"   Bid: {price_info['bid']:.5f}")
                print(f"   Ask: {price_info['ask']:.5f}")
                print(f"   Spread: {price_info['spread_pips']/10:.1f} pips")
            
            # 2. Test: Kleine BUY Order (0.01 Lots - Minimum)
            print("\n1️⃣ Teste BUY Order (0.01 Lots EURUSD)...")
            result = self.order_executor.execute_order(
                symbol="EURUSD",
                order_type="BUY",
                volume=0.01,
                sl_pips=20,
                tp_pips=40
            )
            
            if result.get("success"):
                print("✅ BUY Order Test erfolgreich!")
                ticket = result.get("order_id")
                
                # Zur Trade-History hinzufügen
                self.trade_history.append({
                    "type": "BUY",
                    "ticket": ticket,
                    "price": result.get("price"),
                    "volume": result.get("volume"),
                    "time": datetime.now().isoformat(),
                    "test": True
                })
                
                # Kurz warten
                print("⏱️  Warte 3 Sekunden...")
                time.sleep(3)
                
                # 3. Offene Positionen prüfen
                print("\n2️⃣ Prüfe offene Positionen...")
                positions = self.order_executor.get_open_positions()
                if positions:
                    print(f"   Gefundene Positionen: {len(positions)}")
                    for pos in positions:
                        print(f"   Ticket {pos['ticket']}: {pos['symbol']} {pos['type']} {pos['volume']} Lots")
                
                # 4. Test: Position schließen
                print("\n3️⃣ Teste Position Closing...")
                close_result = self.order_executor.close_position(ticket)
                
                if close_result.get("success"):
                    print("✅ Closing Test erfolgreich!")
                    pnl = close_result.get("pnl", 0)
                    print(f"   P&L: ${pnl:.2f}")
                    print(f"   P&L in Pips: {close_result.get('pnl_pips', 0):.1f}")
                    
                    # Update Trade History
                    for trade in self.trade_history:
                        if trade.get("ticket") == ticket:
                            trade["closed"] = True
                            trade["close_price"] = close_result.get("close_price")
                            trade["pnl"] = pnl
                            trade["close_time"] = datetime.now().isoformat()
                    
                    # 5. Finale Positions-Liste
                    print("\n4️⃣ Finale Positions-Übersicht...")
                    positions = self.order_executor.get_open_positions()
                    print(f"   Offene Positionen: {len(positions)}")
                    
                    return True
                else:
                    print(f"❌ Closing Test fehlgeschlagen: {close_result.get('error')}")
                    
                    # Versuche Position anders zu finden und schließen
                    print("🔄 Versuche alternative Closing-Methode...")
                    positions = mt5.positions_get()
                    if positions:
                        print(f"   Manuell gefundene Positionen: {len(positions)}")
                        for pos in positions:
                            print(f"   Versuche Position {pos.ticket} zu schließen...")
                            alt_result = self.order_executor.close_position(pos.ticket)
                            if alt_result.get("success"):
                                print(f"   ✅ Position {pos.ticket} geschlossen")
                                return True
                    
                    return False
            else:
                print(f"❌ BUY Order Test fehlgeschlagen: {result.get('error')}")
                
                # Detaillierte Fehleranalyse
                print("\n🔍 Detaillierte Fehleranalyse:")
                print("   1. Prüfe ob MT5 Terminal geöffnet ist")
                print("   2. Prüfe Internetverbindung")
                print("   3. Prüfe ob Market geöffnet ist (Forex: 24/5)")
                print("   4. Prüfe Account-Balance und Margin")
                print("   5. Prüfe ob Symbol handelbar ist")
                
                # Alternative: Versuche ohne SL/TP
                print("\n🔄 Versuche Order ohne SL/TP...")
                simple_result = self.order_executor.execute_order(
                    symbol="EURUSD",
                    order_type="BUY",
                    volume=0.01,
                    sl_pips=None,
                    tp_pips=None
                )
                
                if simple_result.get("success"):
                    print("✅ Order ohne SL/TP erfolgreich!")
                    ticket = simple_result.get("order_id")
                    
                    # Sofort schließen
                    time.sleep(2)
                    close_result = self.order_executor.close_position(ticket)
                    if close_result.get("success"):
                        print("✅ Position erfolgreich geschlossen")
                        return True
                
                return False
                
        except Exception as e:
            print(f"❌ Unerwarteter Fehler: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            print("\n" + "="*60)
            print("🧪 ORDER EXECUTION TEST ABGESCHLOSSEN")
            self.print_trade_summary()
    
    def run_simple_live_trading(self, iterations: int = 3):
        """Führt einfaches Live-Trading durch (mit simulierten Signalen)."""
        if not self.initialized:
            if not self.initialize():
                return
        
        print("\n" + "="*60)
        print("🚀 EINFACHES LIVE TRADING - PHASE E WOCHE 2")
        print("="*60)
        print(f"🔄 Starte {iterations} Live-Trading Iterationen...")
        print("⚠️  Verwendet 0.01 Lots pro Trade (Minimum)")
        print("="*60)
        
        try:
            for i in range(iterations):
                print(f"\n{'='*50}")
                print(f"🎯 LIVE ITERATION {i+1}/{iterations}")
                print(f"{'='*50}")
                
                # 1. Aktuelle Marktinfo
                price_info = self.mt5_client.get_live_price("EURUSD")
                if price_info:
                    print(f"📊 Live Preis EURUSD:")
                    print(f"   Bid: {price_info['bid']:.5f}")
                    print(f"   Ask: {price_info['ask']:.5f}")
                    print(f"   Spread: {price_info['spread_pips']/10:.1f} pips")
                
                # 2. Simuliertes ML Signal
                signal, confidence = self.simulate_ml_signal()
                print(f"🤖 Simuliertes Signal: {signal} ({confidence:.1f}% confidence)")
                
                # 3. Trading Entscheidung
                if confidence > 65.0 and signal in ["BUY", "SELL"]:
                    print(f"🚀 EXECUTE LIVE {signal} ORDER")
                    
                    result = self.order_executor.execute_order(
                        symbol="EURUSD",
                        order_type=signal,
                        volume=0.01,
                        sl_pips=20,
                        tp_pips=40
                    )
                    
                    if result.get("success"):
                        print(f"✅ Live {signal} Order erfolgreich!")
                        print(f"   Ticket: {result.get('order_id')}")
                        print(f"   Price: {result.get('price'):.5f}")
                        
                        # Zur History hinzufügen
                        self.trade_history.append({
                            "type": signal,
                            "ticket": result.get("order_id"),
                            "price": result.get("price"),
                            "volume": result.get("volume"),
                            "sl": result.get("sl"),
                            "tp": result.get("tp"),
                            "time": datetime.now().isoformat(),
                            "iteration": i+1,
                            "confidence": confidence
                        })
                    else:
                        print(f"❌ Live {signal} Order fehlgeschlagen: {result.get('error')}")
                else:
                    if signal == "HOLD":
                        print(f"⏸️  HOLD Signal - keine Aktion")
                    else:
                        print(f"⏸️  Confidence zu niedrig ({confidence:.1f}% < 65%)")
                
                # 4. Aktuelle Positionen anzeigen
                positions = self.order_executor.get_open_positions()
                if positions:
                    print(f"\n📦 Aktuelle offene Positionen: {len(positions)}")
                    total_pnl = 0
                    for pos in positions:
                        profit = pos.get('current_profit', 0)
                        profit_pips = pos.get('current_profit_pips', 0)
                        total_pnl += profit
                        print(f"   {pos['symbol']} {pos['type']} {pos['volume']}:")
                        print(f"     P&L: ${profit:.2f} ({profit_pips:.1f} pips)")
                        print(f"     Open: {pos['price_open']:.5f}, SL: {pos['sl']:.5f}, TP: {pos['tp']:.5f}")
                    
                    if len(positions) > 0:
                        print(f"   📈 Total P&L: ${total_pnl:.2f}")
                else:
                    print(f"\n📦 Keine offenen Positionen")
                
                # 5. Account Status
                account_info = self.mt5_client.get_account_info()
                if account_info:
                    print(f"\n💰 Account Status:")
                    print(f"   Balance: ${account_info.get('balance', 0):.2f}")
                    print(f"   Equity: ${account_info.get('equity', 0):.2f}")
                    print(f"   Free Margin: ${account_info.get('margin_free', 0):.2f}")
                
                # 6. Warten zwischen Iterationen (außer letzte)
                if i < iterations - 1:
                    wait_time = random.randint(3, 8)
                    print(f"\n⏱️  Warte {wait_time} Sekunden...")
                    time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            print("\n⚠️  Live Trading durch Benutzer abgebrochen")
        except Exception as e:
            print(f"\n❌ Fehler beim Live Trading: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "="*60)
            print("🏁 LIVE TRADING SESSION BEENDET")
            self.print_trade_summary()
            
            # Alle offenen Positionen schließen (optional)
            print("\n🔒 Schließe alle offenen Positionen...")
            self.close_all_positions()
            
            # Verbindung schließen
            if self.mt5_client:
                self.mt5_client.shutdown()
    
    def close_all_positions(self):
        """Schließt alle offenen Positionen."""
        if not self.initialized or not self.mt5_client.connected:
            return
        
        positions = self.order_executor.get_open_positions()
        if not positions:
            print("   Keine offenen Positionen zum Schließen")
            return
        
        print(f"   Gefundene Positionen: {len(positions)}")
        closed_count = 0
        total_pnl = 0
        
        for pos in positions:
            print(f"   Schließe Position {pos['ticket']}...")
            result = self.order_executor.close_position(pos['ticket'])
            
            if result.get("success"):
                closed_count += 1
                pnl = result.get("pnl", 0)
                total_pnl += pnl
                print(f"     ✅ Geschlossen, P&L: ${pnl:.2f}")
                
                # Update Trade History
                for trade in self.trade_history:
                    if trade.get("ticket") == pos['ticket'] and not trade.get("closed", False):
                        trade["closed"] = True
                        trade["close_price"] = result.get("close_price")
                        trade["pnl"] = pnl
                        trade["close_time"] = datetime.now().isoformat()
                        break
            else:
                print(f"     ❌ Fehler: {result.get('error')}")
        
        print(f"\n   📊 Zusammenfassung:")
        print(f"     Geschlossene Positionen: {closed_count}/{len(positions)}")
        print(f"     Total P&L: ${total_pnl:.2f}")
    
    def print_trade_summary(self):
        """Zeigt eine Zusammenfassung aller Trades an."""
        if not self.trade_history:
            print("\n📊 Trade Zusammenfassung: Keine Trades")
            return
        
        print("\n" + "="*60)
        print("📊 TRADE ZUSAMMENFASSUNG")
        print("="*60)
        
        total_trades = len(self.trade_history)
        closed_trades = sum(1 for t in self.trade_history if t.get("closed", False))
        open_trades = total_trades - closed_trades
        
        print(f"   Gesamt Trades: {total_trades}")
        print(f"   Geschlossene Trades: {closed_trades}")
        print(f"   Offene Trades: {open_trades}")
        
        if closed_trades > 0:
            total_pnl = sum(t.get("pnl", 0) for t in self.trade_history if t.get("closed", False))
            winning_trades = sum(1 for t in self.trade_history if t.get("closed", False) and t.get("pnl", 0) > 0)
            losing_trades = closed_trades - winning_trades
            
            print(f"\n   📈 Performance:")
            print(f"     Total P&L: ${total_pnl:.2f}")
            print(f"     Gewinner: {winning_trades}")
            print(f"     Verlierer: {losing_trades}")
            
            if closed_trades > 0:
                win_rate = (winning_trades / closed_trades) * 100
                print(f"     Win Rate: {win_rate:.1f}%")
        
        # Detailierte Trade-Liste
        print(f"\n   📋 Detailierte Trade-Liste:")
        for i, trade in enumerate(self.trade_history, 1):
            status = "✅ GESCHLOSSEN" if trade.get("closed", False) else "🟡 OFFEN"
            pnl_str = f"${trade.get('pnl', 0):.2f}" if trade.get("pnl") is not None else "N/A"
            
            print(f"\n     {i}. {trade['type']} {trade.get('volume', 0.01)} Lots")
            print(f"        Ticket: {trade.get('ticket', 'N/A')}")
            print(f"        Preis: {trade.get('price', 'N/A'):.5f}")
            print(f"        Zeit: {trade.get('time', 'N/A')}")
            print(f"        Status: {status}")
            print(f"        P&L: {pnl_str}")
            
            if trade.get("confidence"):
                print(f"        Confidence: {trade['confidence']:.1f}%")

def simple_test():
    """Einfacher Test der Live Trading Bridge."""
    print("🚀 Einfacher Live Trading Bridge Test")
    print("-" * 50)
    
    bridge = LiveTradingBridge()
    
    try:
        # Nur Order Execution Test
        print("🧪 Führe Order Execution Test durch...")
        success = bridge.test_order_execution()
        
        if success:
            print("\n✅ Live Trading Bridge funktioniert!")
            return True
        else:
            print("\n❌ Live Trading Bridge Test fehlgeschlagen")
            return False
            
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Wenn direkt ausgeführt, führe einfachen Test durch
    simple_test()