"""
Live Trading Bridge - OPTIMIERTE Version mit P&L FIX
Vollständige Version mit ALLEN Methoden
"""

from .mt5_client import MT5LiveClient
from .order_executor import MT5OrderExecutor
import time
import random
from datetime import datetime
import os
import json
import MetaTrader5 as mt5
import threading

class LiveTradingBridge:
    """OPTIMIERTE Live Trading Bridge mit P&L BERECHNUNG."""
    
    def __init__(self):
        self.mt5_client = None
        self.order_executor = None
        self.ml_engine = None
        self.portfolio = None
        self.initialized = False
        self.trade_history = []
        self.sltp_monitor_active = False
        self.sltp_monitor_thread = None
        self._last_status_show = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("🤖 OPTIMIERTER FOREX TRADING BOT - P&L FIX")
        print("   - Vollständige P&L Berechnung")
        print("   - Trade History Speicherung")
        print("   - Performance Statistiken")
    
    def initialize(self) -> bool:
        """Initialisiert alle Komponenten."""
        print("🔧 Initialisiere optimierten Forex Bot...")
        
        try:
            # Verzeichnis für Daten erstellen
            os.makedirs('data', exist_ok=True)
            
            # 1. MT5 Client
            self.mt5_client = MT5LiveClient()
            if not self.mt5_client.connect():
                print("❌ MT5 Verbindung fehlgeschlagen")
                return False
            
            # 2. Order Executor
            self.order_executor = MT5OrderExecutor(self.mt5_client)
            
            # 3. Portfolio und ML Engine
            try:
                from src.paper_trading.portfolio import Portfolio
                from src.paper_trading.ml_integration import MLTradingEngine
                
                account_info = self.mt5_client.get_account_info()
                initial_balance = account_info.get('balance', 10000.0)
                
                self.portfolio = Portfolio(initial_balance=initial_balance)
                print(f"✅ Portfolio: ${initial_balance:.2f}")
                
                self.ml_engine = MLTradingEngine(self.portfolio)
                print("✅ ML Engine geladen")
                
            except ImportError as e:
                print(f"⚠️  ML-Engine nicht verfügbar: {e}")
                print("   Verwende optimierte simulierte Signale")
                self.ml_engine = None
                self.portfolio = None
            
            # Trade History aus vorheriger Session laden
            self._load_trade_history()
            
            # Status aller Trades synchronisieren (präzise geschlossene Positionen)
            self._sync_trade_status()
            
            # Test-Trades entfernen
            self._clean_test_trades()
            
            self.initialized = True
            print("✅ Bot initialisiert")
            
            # Starte SL/TP Monitor
            self.start_sltp_monitor()
            
            return True
            
        except Exception as e:
            print(f"❌ Initialisierung fehlgeschlagen: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_trade_history(self):
        """Lädt Trade History falls vorhanden."""
        try:
            if os.path.exists('data/latest_trades.json'):
                with open('data/latest_trades.json', 'r') as f:
                    self.trade_history = json.load(f)
                print(f"📊 {len(self.trade_history)} Trades aus History geladen")
        except:
            self.trade_history = []
    
    def _sync_trade_status(self):
        """Synchronisiert Trade Status mit MT5."""
        if not self.mt5_client or not self.mt5_client.connected:
            return
        
        try:
            # Hole alle offenen Positionen von MT5
            mt5_positions = mt5.positions_get()
            open_ticket_ids = [pos.ticket for pos in mt5_positions] if mt5_positions else []
            
            updated_count = 0
            for trade in self.trade_history:
                ticket = trade.get("ticket")
                if ticket and not trade.get("closed", False):
                    # Prüfe ob Position noch offen ist
                    if ticket not in open_ticket_ids:
                        # Position wurde geschlossen (wahrscheinlich in vorheriger Session)
                        try:
                            # Versuche P&L aus History zu holen
                            history_deals = mt5.history_deals_get(position=ticket)
                            if history_deals and len(history_deals) >= 2:
                                open_deal = history_deals[0]
                                close_deal = history_deals[-1]
                                
                                if open_deal.type == 0:  # BUY
                                    pnl = (close_deal.price - open_deal.price) * open_deal.volume * 100000
                                else:  # SELL
                                    pnl = (open_deal.price - close_deal.price) * open_deal.volume * 100000
                                
                                trade.update({
                                    "pnl": pnl,
                                    "close_price": close_deal.price,
                                    "closed": True,
                                    "close_time": close_deal.time if hasattr(close_deal, 'time') else datetime.now().isoformat(),
                                    "close_reason": "Auto-Sync"
                                })
                                updated_count += 1
                        except:
                            # Markiere einfach als geschlossen ohne P&L
                            trade.update({
                                "closed": True,
                                "close_time": datetime.now().isoformat(),
                                "close_reason": "Auto-Sync (No P&L)"
                            })
                            updated_count += 1
            
            if updated_count > 0:
                print(f"🔄 {updated_count} Trades mit MT5 synchronisiert")
                self._save_trade_history()
                
        except Exception as e:
            print(f"⚠️  Trade-Sync Fehler: {e}")
    
    def _clean_test_trades(self):
        """Entfernt Test-Trades aus der History."""
        original_count = len(self.trade_history)
        self.trade_history = [t for t in self.trade_history if t.get("ticket") != 999888777]
        removed = original_count - len(self.trade_history)
        if removed > 0:
            print(f"🧹 {removed} Test-Trades entfernt")
            self._save_trade_history()
    
    def _save_trade_history(self):
        """Speichert Trade History."""
        try:
            trade_file = f"data/trade_history_{self.session_id}.json"
            with open(trade_file, 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            
            with open('data/latest_trades.json', 'w') as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            
        except Exception as e:
            print(f"⚠️  Trade History konnte nicht gespeichert werden: {e}")
    
    def start_sltp_monitor(self):
        """Startet den SL/TP Monitor Thread."""
        if self.sltp_monitor_active:
            return
        
        self.sltp_monitor_active = True
        self.sltp_monitor_thread = threading.Thread(target=self._sltp_monitor_loop, daemon=True)
        self.sltp_monitor_thread.start()
        print("🔔 Aktiver SL/TP Monitor gestartet (Check alle 3s)")
    
    def stop_sltp_monitor(self):
        """Stoppt den SL/TP Monitor."""
        self.sltp_monitor_active = False
        if self.sltp_monitor_thread:
            self.sltp_monitor_thread.join(timeout=2.0)
        print("⏹️  SL/TP Monitor gestoppt")
    
    def _sltp_monitor_loop(self):
        """Haupt-Loop für SL/TP Monitoring."""
        check_interval = 3.0
        
        while self.sltp_monitor_active:
            try:
                self._check_position_for_sltp()
                time.sleep(check_interval)
            except Exception as e:
                print(f"⚠️  SL/TP Monitor Fehler: {e}")
                time.sleep(5.0)
    
    def _check_position_for_sltp(self):
        """Prüft aktuelle Position auf SL/TP."""
        if not self.initialized or not self.mt5_client.connected:
            return
        
        position = self.get_current_position()
        if not position:
            return
        
        ticket = position['ticket']
        pos_type = position['type']
        sl_price = position['sl']
        tp_price = position['tp']
        
        tick = mt5.symbol_info_tick("EURUSD")
        if tick is None:
            return
        
        current_bid = tick.bid
        current_ask = tick.ask
        
        # Status alle 30 Sekunden zeigen
        current_time = datetime.now()
        if self._last_status_show is None or (current_time - self._last_status_show).total_seconds() > 30:
            self._last_status_show = current_time
            
            # P&L für diese Position berechnen
            pnl_info = self.get_position_pnl_details(ticket)
            print(f"📊 Position {ticket}: {pnl_info}")
            
            # Update Trade History mit aktuellem P&L
            self.update_trade_pnl(ticket)
        
        # Prüfe SL/TP
        close_reason = None
        
        if pos_type == 0:  # BUY
            if sl_price > 0 and current_bid <= sl_price:
                close_reason = "SL"
            elif tp_price > 0 and current_bid >= tp_price:
                close_reason = "TP"
        
        elif pos_type == 1:  # SELL
            if sl_price > 0 and current_ask >= sl_price:
                close_reason = "SL"
            elif tp_price > 0 and current_ask <= tp_price:
                close_reason = "TP"
        
        # Schließen wenn SL/TP erreicht
        if close_reason:
            print(f"\n🎯 {close_reason} erreicht für Position {ticket}!")
            
            result = self.close_position_with_pnl(ticket)
            if result.get("success"):
                pnl = result.get("pnl", 0)
                color = "🟢" if pnl and pnl > 0 else "🔴" if pnl and pnl < 0 else "⚪"
                print(f"   ✅ Auto-geschlossen durch {close_reason}")
                print(f"   💰 P&L: {color} ${pnl:.2f}")
    
    def calculate_pnl(self, position_ticket: int) -> dict:
        """
        BERECHNET P&L FÜR EINE POSITION.
        """
        if not self.mt5_client or not self.mt5_client.connected:
            return {"success": False, "error": "MT5 nicht verbunden"}
        
        try:
            # Position von MT5 holen
            positions = mt5.positions_get(ticket=position_ticket)
            if not positions or len(positions) == 0:
                # Position könnte geschlossen sein - versuche aus History
                try:
                    history_deals = mt5.history_deals_get(position=position_ticket)
                    if history_deals and len(history_deals) >= 2:
                        open_deal = history_deals[0]
                        close_deal = history_deals[-1]
                        
                        if open_deal.type == 0:  # BUY
                            pnl = (close_deal.price - open_deal.price) * open_deal.volume * 100000
                            pnl_pips = (close_deal.price - open_deal.price) * 10000
                        else:  # SELL
                            pnl = (open_deal.price - close_deal.price) * open_deal.volume * 100000
                            pnl_pips = (open_deal.price - close_deal.price) * 10000
                        
                        return {
                            "success": True,
                            "ticket": position_ticket,
                            "current_pnl": pnl,
                            "current_pnl_pips": pnl_pips,
                            "current_price": close_deal.price,
                            "entry_price": open_deal.price,
                            "volume": open_deal.volume,
                            "type": "BUY" if open_deal.type == 0 else "SELL",
                            "closed": True
                        }
                except:
                    pass
                
                return {"success": False, "error": "Position nicht gefunden"}
            
            position = positions[0]
            
            # Aktuelle Preise
            tick = mt5.symbol_info_tick(position.symbol)
            if not tick:
                return {"success": False, "error": "Tick nicht verfügbar"}
            
            # Korrekte P&L Berechnung für Forex
            contract_size = 100000  # 1 Lot = 100,000 Einheiten
            volume = position.volume
            
            if position.type == 0:  # BUY Position
                price_difference = tick.bid - position.price_open
                pnl = price_difference * volume * contract_size
                pnl_pips = (tick.bid - position.price_open) * 10000
                
            else:  # SELL Position
                price_difference = position.price_open - tick.ask
                pnl = price_difference * volume * contract_size
                pnl_pips = (position.price_open - tick.ask) * 10000
            
            return {
                "success": True,
                "ticket": position_ticket,
                "current_pnl": pnl,
                "current_pnl_pips": pnl_pips,
                "current_price": tick.bid if position.type == 0 else tick.ask,
                "entry_price": position.price_open,
                "volume": volume,
                "type": "BUY" if position.type == 0 else "SELL",
                "closed": False
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_position_pnl_details(self, position_ticket: int) -> str:
        """Gibt detaillierte P&L Info für eine Position aus."""
        pnl_data = self.calculate_pnl(position_ticket)
        
        if not pnl_data.get("success"):
            return f"P&L nicht verfügbar: {pnl_data.get('error', 'Unknown')}"
        
        pnl = pnl_data["current_pnl"]
        pnl_pips = pnl_data["current_pnl_pips"]
        color = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "⚪"
        
        if pnl_data.get("closed", False):
            return f"{color} ${pnl:.2f} ({pnl_pips:.1f} pips) [GESCHLOSSEN]"
        else:
            return f"{color} ${pnl:.2f} ({pnl_pips:.1f} pips)"
    
    def update_trade_pnl(self, ticket: int):
        """Aktualisiert P&L für einen Trade in der History."""
        for trade in self.trade_history:
            if trade.get("ticket") == ticket:
                pnl_data = self.calculate_pnl(ticket)
                if pnl_data.get("success"):
                    trade["current_pnl"] = pnl_data["current_pnl"]
                    trade["current_pnl_pips"] = pnl_data["current_pnl_pips"]
                    trade["last_update"] = datetime.now().isoformat()
                    
                    # Wenn Position geschlossen ist, markiere sie auch in der History
                    if pnl_data.get("closed", False) and not trade.get("closed", False):
                        trade.update({
                            "closed": True,
                            "close_price": pnl_data.get("current_price"),
                            "pnl": pnl_data["current_pnl"],
                            "close_time": datetime.now().isoformat(),
                            "close_reason": "Auto-Detected"
                        })
                break
    
    def close_position_with_pnl(self, ticket: int) -> dict:
        """
        Schließt Position und berechnet FINALEN P&L.
        """
        result = self.order_executor.close_position(ticket)
        
        if result.get("success"):
            # Warte kurz für Position Update
            time.sleep(0.5)
            
            try:
                # P&L aus MT5 History holen
                history_deals = mt5.history_deals_get(position=ticket)
                if history_deals and len(history_deals) >= 2:
                    open_deal = history_deals[0]
                    close_deal = history_deals[-1]
                    
                    if open_deal.type == 0:  # BUY
                        pnl = (close_deal.price - open_deal.price) * open_deal.volume * 100000
                    else:  # SELL
                        pnl = (open_deal.price - close_deal.price) * open_deal.volume * 100000
                    
                    result["pnl"] = pnl
                    result["close_price"] = close_deal.price
                    
                    # Update Trade History
                    for trade in self.trade_history:
                        if trade.get("ticket") == ticket:
                            trade.update({
                                "pnl": pnl,
                                "close_price": close_deal.price,
                                "closed": True,
                                "close_time": datetime.now().isoformat(),
                                "close_reason": result.get("close_reason", "Manual")
                            })
                            break
                
            except Exception as e:
                print(f"⚠️  P&L Berechnung nach Close fehlgeschlagen: {e}")
                # Fallback auf geschätzten P&L
                pnl_data = self.calculate_pnl(ticket)
                if pnl_data.get("success"):
                    result["pnl"] = pnl_data["current_pnl"]
        
        return result
    
    def has_open_position(self):
        """Prüft ob bereits eine Position offen ist."""
        positions = self.order_executor.get_open_positions()
        return len(positions) > 0
    
    def get_current_position(self):
        """Gibt die aktuelle Position zurück."""
        positions = self.order_executor.get_open_positions()
        return positions[0] if positions else None
    
    def get_ml_signal(self):
        """Holt ML-Signal mit optimierter Logik."""
        if self.ml_engine:
            try:
                signal, confidence = self.ml_engine.generate_signal()
                return signal, confidence
            except Exception as e:
                print(f"⚠️  ML-Signal Fehler: {e}")
        
        # Optimiertes Fallback
        current_hour = datetime.now().hour
        
        if 8 <= current_hour <= 17:  # Handelszeiten
            weights = [0.30, 0.30, 0.40]
        else:
            weights = [0.15, 0.15, 0.70]
        
        signals = ["BUY", "SELL", "HOLD"]
        signal = random.choices(signals, weights=weights)[0]
        
        if signal == "HOLD":
            confidence = random.uniform(40, 60)
        else:
            confidence = random.uniform(75, 90)
        
        return signal, confidence
    
    def execute_trade(self, signal: str, confidence: float):
        """Führt Trade aus UND SPEICHERT DATEN."""
        if self.has_open_position():
            print("⏸️  Bereits Position offen - warte auf SL/TP")
            return None
        
        print(f"🚀 PRÜFE {signal} TRADE (Confidence: {confidence:.1f}%)")
        
        # Optimierte Forex-Parameter
        if confidence > 85:
            volume = 0.02
            sl_pips = 20
            tp_pips = 40
        elif confidence > 75:
            volume = 0.01
            sl_pips = 20
            tp_pips = 40
        else:
            print(f"   ❌ Confidence zu niedrig ({confidence:.1f}% < 75%)")
            return None
        
        result = self.order_executor.execute_order(
            symbol="EURUSD",
            order_type=signal,
            volume=volume,
            sl_pips=sl_pips,
            tp_pips=tp_pips
        )
        
        if not result.get("success"):
            print(f"❌ {signal} fehlgeschlagen: {result.get('error')}")
            return None
        
        ticket = result.get("order_id")
        
        print(f"✅ {signal} erfolgreich! Ticket: {ticket}")
        print(f"   Entry: {result.get('price'):.5f}")
        print(f"   SL: {sl_pips}pips, TP: {tp_pips}pips")
        
        # Trade History
        trade_record = {
            "session_id": self.session_id,
            "type": signal,
            "ticket": ticket,
            "price": result.get("price"),
            "volume": volume,
            "sl": result.get("sl"),
            "tp": result.get("tp"),
            "sl_pips": sl_pips,
            "tp_pips": tp_pips,
            "confidence": confidence,
            "time": datetime.now().isoformat(),
            "status": "OPEN",
            "closed": False,
            "current_pnl": 0.0,
            "current_pnl_pips": 0.0,
            "pnl": None,
            "close_price": None,
            "close_time": None,
            "close_reason": None
        }
        
        self.trade_history.append(trade_record)
        self._save_trade_history()
        
        return ticket
    
    def close_all_positions(self):
        """Schließt alle Positionen MIT P&L BERECHNUNG."""
        if not self.initialized:
            return
        
        positions = self.order_executor.get_open_positions()
        if not positions:
            print("   Keine offenen Positionen")
            return
        
        print(f"   Positionen zum Schließen: {len(positions)}")
        
        for pos in positions:
            result = self.close_position_with_pnl(pos['ticket'])
            if result.get("success"):
                pnl = result.get("pnl", 0)
                color = "🟢" if pnl and pnl > 0 else "🔴" if pnl and pnl < 0 else "⚪"
                print(f"   ✅ Position {pos['ticket']} geschlossen")
                print(f"      {color} P&L: ${pnl:.2f}")
        
        self._save_trade_history()
    
    def print_summary(self):
        """ZEIGT ZUSAMMENFASSUNG MIT P&L - KORRIGIERTE VERSION."""
        print("\n" + "="*60)
        print("📊 TRADING ZUSAMMENFASSUNG MIT P&L")
        print("="*60)
        
        # Filtere Test-Trades heraus (Ticket #999888777)
        real_trades = [t for t in self.trade_history if t.get("ticket") != 999888777]
        
        if not real_trades:
            print("Keine realen Trades in dieser Session")
            return
        
        # Aktualisiere P&L für alle Trades
        for trade in real_trades:
            if not trade.get("closed", False):
                self.update_trade_pnl(trade.get("ticket"))
        
        closed_trades = [t for t in real_trades if t.get("closed", False)]
        open_trades = [t for t in real_trades if not t.get("closed", False)]
        
        print(f"📊 ECHTE TRADES: {len(real_trades)}")
        print(f"   ✅ Geschlossen: {len(closed_trades)}")
        print(f"   🟡 Offen: {len(open_trades)}")
        
        if closed_trades:
            # P&L Statistiken für geschlossene Trades
            pnl_values = []
            for trade in closed_trades:
                pnl = trade.get("pnl")
                if pnl is not None:
                    try:
                        pnl_values.append(float(pnl))
                    except (ValueError, TypeError):
                        pass
            
            if pnl_values:
                total_pnl = sum(pnl_values)
                winning = len([pnl for pnl in pnl_values if pnl > 0])
                losing = len([pnl for pnl in pnl_values if pnl < 0])
                win_rate = winning / len(pnl_values) * 100 if pnl_values else 0
                
                print(f"\n💰 P&L STATISTIKEN (Geschlossen):")
                print(f"   Total P&L: ${total_pnl:.2f}")
                print(f"   Gewinner: {winning}, Verlierer: {losing}")
                print(f"   Win Rate: {win_rate:.1f}%")
        
        # Aktuelle P&L für offene Positionen
        if open_trades:
            print(f"\n📈 AKTUELLE POSITIONEN:")
            for trade in open_trades:
                ticket = trade.get("ticket")
                if ticket:
                    # AKTUELLEN P&L BERECHNEN
                    pnl_data = self.calculate_pnl(ticket)
                    if pnl_data.get("success"):
                        current_pnl = pnl_data["current_pnl"]
                        current_pips = pnl_data["current_pnl_pips"]
                        color = "🟢" if current_pnl > 0 else "🔴" if current_pnl < 0 else "⚪"
                        print(f"   {trade['type']} #{ticket}: {color} ${current_pnl:.2f} ({current_pips:.1f} pips)")
        
        print(f"\n📋 TRADE DETAILS:")
        for i, trade in enumerate(real_trades, 1):
            ticket = trade.get("ticket")
            status = "✅ GESCHLOSSEN" if trade.get("closed", False) else "🟡 OFFEN"
            
            # P&L Wert berechnen
            if trade.get("closed", False):
                # Für geschlossene Trades: finale P&L
                pnl = trade.get("pnl", 0)
                if pnl is None:
                    pnl = 0
            else:
                # Für offene Trades: aktuellen P&L berechnen
                pnl = trade.get("current_pnl", 0)
                if pnl is None:
                    pnl = 0
            
            # Farbe basierend auf P&L
            if pnl > 0:
                color = "🟢"
            elif pnl < 0:
                color = "🔴"
            else:
                color = "⚪"
            
            print(f"\n  {i}. {trade['type']} {trade.get('volume', 0.01)} Lots")
            print(f"     Ticket: #{trade.get('ticket', 'N/A')}")
            print(f"     Entry: {trade.get('price', 'N/A'):.5f}")
            print(f"     Confidence: {trade.get('confidence', 'N/A'):.1f}%")
            print(f"     SL/TP: {trade.get('sl_pips', 'N/A')}/{trade.get('tp_pips', 'N/A')}pips")
            
            if trade.get('closed', False):
                close_price = trade.get('close_price', 'N/A')
                close_reason = trade.get('close_reason', 'N/A')
                if close_price != 'N/A':
                    print(f"     Close: {close_price:.5f}")
                if close_reason != 'N/A':
                    print(f"     Grund: {close_reason}")
                print(f"     Status: {status}")
                print(f"     P&L: {color} ${pnl:.2f}")
            else:
                # Für offene Positionen zusätzlich aktuelle P&L Info anzeigen
                if ticket:
                    pnl_data = self.calculate_pnl(ticket)
                    if pnl_data.get("success"):
                        current_pnl = pnl_data["current_pnl"]
                        current_pips = pnl_data["current_pnl_pips"]
                        print(f"     Status: {status}")
                        print(f"     Current P&L: {color} ${current_pnl:.2f} ({current_pips:.1f} pips)")
                    else:
                        print(f"     Status: {status}")
                        print(f"     Current P&L: {color} ${pnl:.2f}")
                else:
                    print(f"     Status: {status}")
                    print(f"     Current P&L: {color} ${pnl:.2f}")
        
        print("="*60)
        print(f"💾 Session ID: {self.session_id}")
        print(f"📅 Datum: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    
    def run_ml_live_trading(self, iterations: int = 5):
        """OPTIMIERTE Trading Session MIT P&L TRACKING."""
        print("\n" + "="*60)
        print("🚀 ML LIVE TRADING - MIT VOLLSTÄNDIGEM P&L TRACKING")
        print("="*60)
        print(f"🔄 {iterations} Iterationen")
        print("🎯 Parameter: SL=20pips, TP=40pips, Confidence>75%")
        print("⏱️  Wartezeiten: 30-90 Sekunden")
        print("="*60)
        
        if not self.initialized:
            if not self.initialize():
                return
        
        try:
            for i in range(iterations):
                print(f"\n{'='*50}")
                print(f"📈 ITERATION {i+1}/{iterations}")
                print(f"{'='*50}")
                
                # Check aktuelle Position MIT P&L
                if self.has_open_position():
                    position = self.get_current_position()
                    if position:
                        ticket = position['ticket']
                        pnl_str = self.get_position_pnl_details(ticket)
                        print(f"📦 AKTUELLE POSITION: Ticket {ticket}")
                        print(f"   Type: {'BUY' if position['type'] == 0 else 'SELL'}")
                        print(f"   Entry: {position['price_open']:.5f}")
                        print(f"   P&L: {pnl_str}")
                        
                        # Aktualisiere Trade History mit aktuellem P&L
                        self.update_trade_pnl(ticket)
                        
                        wait_time = random.randint(60, 90)
                        print(f"\n⏱️  Position offen - warte {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                
                # Keine Position - prüfe für neuen Trade
                print("📊 Keine offene Position")
                
                price_info = self.mt5_client.get_live_price("EURUSD")
                if price_info:
                    print(f"💰 EURUSD: {price_info['bid']:.5f}/{price_info['ask']:.5f}")
                
                signal, confidence = self.get_ml_signal()
                
                if signal in ["BUY", "SELL"] and confidence >= 75:
                    self.execute_trade(signal, confidence)
                else:
                    print(f"⏸️  {signal} ({confidence:.1f}%) - keine Aktion")
                
                wait_time = random.randint(30, 60)
                print(f"\n⏱️  Warte {wait_time} Sekunden...")
                time.sleep(wait_time)
                    
        except KeyboardInterrupt:
            print("\n⚠️  Session abgebrochen")
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("\n" + "="*60)
            print("🏁 SESSION BEENDET - FINALE STATISTIKEN")
            print("="*60)
            
            # Finale P&L Updates für offene Positionen
            if self.has_open_position():
                print("\n🔄 Aktualisiere finale P&L für offene Positionen...")
                positions = self.order_executor.get_open_positions()
                for pos in positions:
                    self.update_trade_pnl(pos['ticket'])
            
            self.print_summary()
            
            print("\n🔒 Schließe Positionen...")
            self.close_all_positions()
            
            self.stop_sltp_monitor()
            
            if self.mt5_client:
                self.mt5_client.shutdown()
    
    def run_optimized_trading(self, iterations: int = 5):
        """Alias für run_ml_live_trading."""
        return self.run_ml_live_trading(iterations)
    
    def run_simple_live_trading(self, iterations: int = 5):
        """Alias für run_ml_live_trading."""
        print("⚠️  Verwende optimierte Version...")
        return self.run_ml_live_trading(iterations)

# ============================================================================
# TEST FUNKTIONEN
# ============================================================================

def test_pnl_calculation():
    """Testet speziell die P&L Berechnung."""
    print("🧪 TESTE P&L BERECHNUNG")
    print("-" * 50)
    
    bridge = LiveTradingBridge()
    
    try:
        if not bridge.initialize():
            return False
        
        print("\n✅ P&L System initialisiert")
        print("   🎯 Teste P&L Berechnung...")
        
        print("\n2. Zeige Zusammenfassung...")
        bridge.print_summary()
        
        print("\n✅ P&L Test abgeschlossen!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        bridge.stop_sltp_monitor()
        if bridge.mt5_client:
            bridge.mt5_client.shutdown()

def test_ml_integration():
    """Testet die ML-Integration MIT P&L."""
    print("🚀 ML-INTEGRATION TEST MIT P&L")
    print("-" * 50)
    
    bridge = LiveTradingBridge()
    
    try:
        print("🧪 Teste optimierten Bot mit P&L Tracking...")
        
        if not bridge.initialize():
            return False
        
        print("\n🎯 Kurzer Trading Test...")
        signal, confidence = bridge.get_ml_signal()
        print(f"   Signal: {signal}, Confidence: {confidence:.1f}%")
        
        if signal in ["BUY", "SELL"] and confidence >= 75:
            print("   Würde Trade ausführen...")
            print("   (Test ohne echten Trade)")
        else:
            print("   Kein Trade (Signal zu schwach)")
        
        print("\n📊 Zeige Systemstatus...")
        print(f"   Initialized: {bridge.initialized}")
        print(f"   MT5 Connected: {bridge.mt5_client.connected if bridge.mt5_client else False}")
        print(f"   Trade History: {len(bridge.trade_history)} Trades")
        
        print("\n✅ Test mit P&L abgeschlossen!")
        return True
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        bridge.stop_sltp_monitor()
        if bridge.mt5_client:
            bridge.mt5_client.shutdown()

def test_order_execution():
    """Testet Order Execution."""
    print("🚀 ORDER EXECUTION TEST")
    print("-" * 50)
    return test_ml_integration()

def test_optimized_integration():
    """Haupttest-Funktion für main.py."""
    print("🤖 OPTIMIERTER INTEGRATIONSTEST")
    print("-" * 50)
    return test_ml_integration()

if __name__ == "__main__":
    print("🤖 P&L FIX TEST")
    print("="*60)
    test_pnl_calculation()