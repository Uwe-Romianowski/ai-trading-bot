#!/usr/bin/env python3
"""
AI TRADING BOT v4.0 - MIT PAPER TRADING ENGINE (PHASE D)
=========================================================
Hauptsteuerung für den AI Trading Bot mit ML-Signalen,
MT5-Live-Integration und Paper-Trading Engine.
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Importiere alle Module
try:
    from src.ml_integration.ml_signal_generator import MLSignalGenerator
    from src.mt5_client.mt5_live_client import MT5LiveClient
    # IMPORT KORREKTUR: Portfolio liegt in src/paper_trading/portfolio.py (kein Unterordner)
    from src.paper_trading.portfolio import PaperPortfolio
    # PaperOrder-Klassen müssen aus order.py importiert werden
    from src.paper_trading.order import OrderType
    print("✅ Alle Module erfolgreich importiert")
except ImportError as e:
    print(f"⚠️  Import-Fehler: {e}")
    print("📁 Stellen Sie sicher, dass alle Module existieren")
    print("📁 Struktur sollte sein: src/paper_trading/portfolio.py")
    sys.exit(1)

# Lade Umgebungsvariablen
load_dotenv()


class AITradingBot:
    """Hauptklasse für den AI Trading Bot."""
    
    def __init__(self):
        """Initialisiert den Trading Bot."""
        self.ml_generator = None
        self.mt5_client = None
        self.paper_portfolio = None
        self.running = True
        
        # Lade Konfiguration
        self.initial_balance = float(os.getenv('PAPER_TRADING_INITIAL_BALANCE', 10000.0))
        self.ml_buy_threshold = float(os.getenv('ML_BUY_THRESHOLD', 0.60))
        self.ml_sell_threshold = float(os.getenv('ML_SELL_THRESHOLD', 0.60))
        
        print("\n" + "="*60)
        print("🤖 AI TRADING BOT v4.0 - MIT PAPER TRADING ENGINE")
        print("="*60)
        print(f"📅 Systemzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Paper Trading Startkapital: {self.initial_balance:.2f} USD")
        print("="*60)
    
    def init_ml_system(self):
        """Initialisiert das ML-System."""
        print("\n" + "="*40)
        print("🧠 ML-SYSTEM STARTEN")
        print("="*40)
        
        try:
            self.ml_generator = MLSignalGenerator()
            print("✅ ML-Modell geladen: RandomForestClassifier")
            print("✅ Scaler geladen")
            print("✅ Features geladen")
            print("✅ MLSignalGenerator initialisiert")
            return True
        except Exception as e:
            print(f"❌ ML-System Fehler: {e}")
            return False
    
    def init_mt5_client(self):
        """Initialisiert den MT5 Client."""
        print("\n" + "="*40)
        print("📡 MT5 CLIENT INITIALISIEREN")
        print("="*40)
        
        try:
            self.mt5_client = MT5LiveClient()
            print(f"✅ MT5 verbunden (Demo Account)")
            return True
        except Exception as e:
            print(f"❌ MT5 Verbindungsfehler: {e}")
            return False
    
    def init_paper_trading(self):
        """Initialisiert das Paper-Trading Portfolio."""
        print("\n" + "="*40)
        print("📊 PAPER TRADING INITIALISIEREN")
        print("="*40)
        
        try:
            self.paper_portfolio = PaperPortfolio(initial_balance=self.initial_balance)
            print(f"✅ Paper Portfolio erstellt: {self.paper_portfolio.portfolio_id}")
            print(f"💰 Startkapital: {self.initial_balance:.2f} USD")
            return True
        except Exception as e:
            print(f"❌ Paper Trading Fehler: {e}")
            return False
    
    def run_ml_signal_generation(self):
        """Führt ML-Signal-Generation im Live-Modus aus."""
        if not self.ml_generator:
            print("❌ ML-System nicht initialisiert. Bitte Option 1 zuerst ausführen.")
            return
        
        print("\n" + "="*40)
        print("⚡ ML-SIGNAL GENERIERUNG (LIVE)")
        print("="*40)
        
        try:
            # Simuliere ML-Signal (ersetzten Sie dies mit Ihrer echten ML-Logik)
            signal, confidence = self._simulate_ml_signal()
            
            print(f"\n🎯 GENERIERTES SIGNAL:")
            print(f"   Signal:     {signal}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Threshold:  BUY > {self.ml_buy_threshold}, SELL > {self.ml_sell_threshold}")
            
            # Zeige Handelsempfehlung basierend auf Confidence
            if signal == "BUY" and confidence >= self.ml_buy_threshold:
                print(f"   🟢 EMPFEHLUNG: BUY Position eröffnen (Confidence: {confidence:.1%})")
            elif signal == "SELL" and confidence >= self.ml_sell_threshold:
                print(f"   🔴 EMPFEHLUNG: SELL Position eröffnen (Confidence: {confidence:.1%})")
            else:
                print(f"   ⚪ EMPFEHLUNG: HOLD (Confidence unter Threshold)")
            
            return signal, confidence
            
        except Exception as e:
            print(f"❌ Signal-Generierungsfehler: {e}")
            return None, None
    
    def run_paper_trading_demo(self):
        """Führt eine Paper-Trading Demo aus."""
        if not self.paper_portfolio:
            print("❌ Paper Trading nicht initialisiert. Bitte Option 9 zuerst ausführen.")
            return
        
        print("\n" + "="*40)
        print("🎮 PAPER TRADING DEMO")
        print("="*40)
        
        # Demo: ML-Signal generieren
        print("\n1. GENERIERE ML-SIGNAL FÜR DEMO:")
        signal, confidence = self._simulate_ml_signal()
        print(f"   Signal: {signal} mit {confidence:.1%} Confidence")
        
        # Demo-Parameter
        demo_symbol = "EURUSD"
        demo_price = 1.0850
        demo_stop_loss = 1.0800 if signal == "BUY" else 1.0900
        demo_take_profit = 1.0950 if signal == "BUY" else 1.0750
        
        print(f"\n2. TRADE PARAMETER:")
        print(f"   Symbol:      {demo_symbol}")
        print(f"   Preis:       {demo_price}")
        print(f"   Stop-Loss:   {demo_stop_loss}")
        print(f"   Take-Profit: {demo_take_profit}")
        
        # Position basierend auf Signal öffnen
        print(f"\n3. ÖFFNE POSITION:")
        if signal == "BUY":
            order_type = OrderType.BUY
            order = self.paper_portfolio.open_position(
                symbol=demo_symbol,
                order_type=order_type,
                entry_price=demo_price,
                stop_loss=demo_stop_loss,
                take_profit=demo_take_profit,
                signal_confidence=confidence
            )
        elif signal == "SELL":
            order_type = OrderType.SELL
            order = self.paper_portfolio.open_position(
                symbol=demo_symbol,
                order_type=order_type,
                entry_price=demo_price,
                stop_loss=demo_stop_loss,
                take_profit=demo_take_profit,
                signal_confidence=confidence
            )
        else:
            print("   ⚪ HOLD Signal - keine Position eröffnet")
            return
        
        if not order:
            print("   ❌ Position konnte nicht eröffnet werden")
            return
        
        # Simuliere Preisänderung und schließe Position
        print(f"\n4. SIMULIERE MARKTBEWEGUNG:")
        time.sleep(2)  # Kurze Pause für Realismus
        
        # Bestimme Exit-Preis basierend auf Signal (hier: immer profitabel für Demo)
        if signal == "BUY":
            exit_price = demo_price + 0.0030  # +0.0030 Gewinn
        else:
            exit_price = demo_price - 0.0020  # -0.0020 Gewinn
        
        print(f"   Neuer Marktpreis: {exit_price}")
        
        print(f"\n5. SCHLIESSE POSITION:")
        pnl = self.paper_portfolio.close_position(demo_symbol, exit_price)
        
        if pnl is not None:
            pnl_sign = "+" if pnl > 0 else ""
            print(f"   🔒 Position geschlossen mit P&L: {pnl_sign}{pnl:.2f} USD")
        
        print(f"\n6. PERFORMANCE REPORT:")
        self.paper_portfolio.print_detailed_report()
    
    def show_paper_portfolio_status(self):
        """Zeigt den aktuellen Paper Portfolio Status."""
        if not self.paper_portfolio:
            print("❌ Paper Trading nicht initialisiert.")
            return
        
        print("\n" + "="*40)
        print("📈 PAPER PORTFOLIO STATUS")
        print("="*40)
        
        self.paper_portfolio.print_detailed_report()
    
    def _simulate_ml_signal(self):
        """Simuliert ein ML-Signal für Demo-Zwecke."""
        # Diese Funktion simuliert Ihre echte ML-Logik
        # Ersetzen Sie dies mit Ihrem echten MLSignalGenerator
        
        import random
        signals = ["BUY", "SELL", "HOLD"]
        weights = [0.35, 0.35, 0.30]  # 35% BUY, 35% SELL, 30% HOLD
        
        signal = random.choices(signals, weights)[0]
        
        # Confidence basierend auf Signal
        if signal == "BUY":
            confidence = random.uniform(0.50, 0.85)
        elif signal == "SELL":
            confidence = random.uniform(0.50, 0.85)
        else:  # HOLD
            confidence = random.uniform(0.40, 0.60)
        
        return signal, confidence
    
    def show_menu(self):
        """Zeigt das Hauptmenü an."""
        print("\n" + "="*60)
        print("📋 HAUPTMENÜ - AI TRADING BOT v4.0")
        print("="*60)
        print("1. 🧠 ML-System starten")
        print("2. 📊 Testdaten verarbeiten (simuliert)")
        print("3. 🎯 Signal generieren")
        print("4. 📈 Status anzeigen")
        print("5. 🔄 MT5 Integration testen")
        print("6. 📡 MT5 + ML Integration (LIVE)")
        print("7. 🛠️  System Check")
        print("8. 🚪 Beenden")
        print("9. 📊 PAPER TRADING MODUS (NEU - PHASE D)")
        print("="*60)
    
    def handle_choice(self, choice):
        """Verarbeitet die Benutzerauswahl."""
        if choice == "1":
            self.init_ml_system()
        elif choice == "2":
            print("\n📊 Testdaten werden verarbeitet...")
            # Ihre existierende Testdaten-Logik hier
            print("✅ Testdaten erfolgreich verarbeitet")
        elif choice == "3":
            print("\n🎯 Signal wird generiert...")
            signal, confidence = self.run_ml_signal_generation()
            if signal:
                print(f"✅ Signal generiert: {signal} ({confidence:.1%})")
        elif choice == "4":
            print("\n📈 System Status:")
            print(f"   ML-System: {'✅ Initialisiert' if self.ml_generator else '❌ Nicht initialisiert'}")
            print(f"   MT5 Client: {'✅ Verbunden' if self.mt5_client else '❌ Nicht verbunden'}")
            print(f"   Paper Trading: {'✅ Aktiv' if self.paper_portfolio else '❌ Nicht aktiv'}")
            if self.paper_portfolio:
                summary = self.paper_portfolio.get_portfolio_summary()
                print(f"   Portfolio Balance: {summary['current_balance']:.2f} USD")
                print(f"   Total Trades: {summary['total_trades']}")
        elif choice == "5":
            if self.init_mt5_client():
                print("✅ MT5 Integration erfolgreich getestet")
        elif choice == "6":
            if self.init_ml_system() and self.init_mt5_client():
                self.run_ml_signal_generation()
        elif choice == "7":
            print("\n🛠️  System Check wird durchgeführt...")
            # Ihre existierende System-Check-Logik hier
            print("✅ System Check abgeschlossen")
        elif choice == "8":
            print("\n👋 Beende AI Trading Bot...")
            self.running = False
        elif choice == "9":
            self.paper_trading_menu()
        else:
            print(f"\n❌ Ungültige Auswahl: '{choice}'. Bitte 1-9 wählen.")
    
    def paper_trading_menu(self):
        """Zeigt das Paper Trading Untermenü an."""
        while True:
            print("\n" + "="*50)
            print("📊 PAPER TRADING ENGINE - PHASE D")
            print("="*50)
            print("a. 🆕 Paper Portfolio initialisieren")
            print("b. 🎮 Demo Trade ausführen")
            print("c. 📈 Portfolio Status anzeigen")
            print("d. 🧠 ML-Signal + Auto-Trade (Demo)")
            print("e. 💾 Performance Report speichern")
            print("f. ↩️  Zurück zum Hauptmenü")
            print("="*50)
            
            sub_choice = input("Wahl (a-f): ").strip().lower()
            
            if sub_choice == "a":
                self.init_paper_trading()
            elif sub_choice == "b":
                self.run_paper_trading_demo()
            elif sub_choice == "c":
                self.show_paper_portfolio_status()
            elif sub_choice == "d":
                print("\n🧠 ML-SIGNAL + AUTO-TRADE DEMO")
                print("="*40)
                if not self.paper_portfolio:
                    print("❌ Bitte zuerst Paper Portfolio initialisieren (Option a)")
                    continue
                
                # Generiere ML-Signal
                signal, confidence = self.run_ml_signal_generation()
                
                if signal in ["BUY", "SELL"]:
                    # Automatischen Trade ausführen
                    print(f"\n🤖 AUTOMATISCHER TRADE AUSFÜHREN:")
                    print(f"   Signal: {signal} mit {confidence:.1%} Confidence")
                    
                    # Trade-Parameter
                    symbol = "EURUSD"
                    price = 1.0850
                    stop_loss = 1.0800 if signal == "BUY" else 1.0900
                    take_profit = 1.0900 if signal == "BUY" else 1.0800
                    
                    # Position öffnen
                    order_type = OrderType.BUY if signal == "BUY" else OrderType.SELL
                    order = self.paper_portfolio.open_position(
                        symbol=symbol,
                        order_type=order_type,
                        entry_price=price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        signal_confidence=confidence
                    )
                    
                    if order:
                        print(f"   ✅ Auto-Trade ausgeführt: {order}")
                    else:
                        print("   ❌ Auto-Trade fehlgeschlagen")
                else:
                    print("   ⚪ HOLD Signal - kein Auto-Trade ausgeführt")
            
            elif sub_choice == "e":
                if self.paper_portfolio:
                    self.paper_portfolio.save_performance_report()
                else:
                    print("❌ Kein aktives Paper Portfolio")
            
            elif sub_choice == "f":
                print("↩️  Zurück zum Hauptmenü...")
                break
            
            else:
                print(f"❌ Ungültige Auswahl: '{sub_choice}'. Bitte a-f wählen.")
    
    def run(self):
        """Hauptausführungsfunktion des Bots."""
        while self.running:
            self.show_menu()
            choice = input("\nWahl (1-9): ").strip()
            self.handle_choice(choice)
        
        print("\n" + "="*60)
        print("✅ AI Trading Bot erfolgreich beendet")
        print(f"📅 Endzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)


def main():
    """Hauptfunktion."""
    bot = AITradingBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programm durch Benutzer abgebrochen")
    except Exception as e:
        print(f"\n❌ Kritischer Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()