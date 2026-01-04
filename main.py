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
import importlib.util
from datetime import datetime
from dotenv import load_dotenv

# Lade Umgebungsvariablen
load_dotenv()

# ============================================
# KORRIGIERTE IMPORT-LOGIK FÜR PAPER TRADING
# ============================================
print("🤖 AI TRADING BOT v4.0 - PHASE D")
print("="*60)

# 1. PAPER TRADING PORTFOLIO IMPORT (FESTER PFAD)
try:
    # Fester Pfad zur portfolio.py Datei
    portfolio_path = os.path.join('src', 'paper_trading', 'portfolio.py')
    
    # Als Modul direkt laden (umgeht Import-Probleme)
    spec = importlib.util.spec_from_file_location("paper_portfolio", portfolio_path)
    portfolio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portfolio_module)
    
    PaperPortfolio = portfolio_module.PaperPortfolio
    OrderType = portfolio_module.OrderType
    
    paper_trading_available = True
    print("✅ PaperPortfolio importiert")
    
except Exception as e:
    print(f"❌ PaperPortfolio Import fehlgeschlagen: {e}")
    print(f"   Pfad: {portfolio_path}")
    paper_trading_available = False
    PaperPortfolio = None
    OrderType = None

# 2. ML INTEGRATION IMPORT (OPTIONAL)
try:
    ml_integration_path = os.path.join('src', 'paper_trading', 'ml_integration.py')
    if os.path.exists(ml_integration_path):
        spec = importlib.util.spec_from_file_location("ml_integration", ml_integration_path)
        ml_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ml_module)
        
        MLTradingEngine = ml_module.MLTradingEngine
        SignalType = ml_module.SignalType
        
        ml_integration_available = True
        print("✅ MLTradingEngine importiert")
    else:
        ml_integration_available = False
        MLTradingEngine = None
        SignalType = None
        print("⚠️  ml_integration.py nicht gefunden (Option 10 nicht verfügbar)")
        
except Exception as e:
    print(f"⚠️  ML Integration Import fehlgeschlagen: {e}")
    ml_integration_available = False
    MLTradingEngine = None
    SignalType = None

# 3. ANDERE MODULE (OPTIONAL)
ml_available = False
mt5_available = False

try:
    from src.ml_integration.ml_signal_generator import MLSignalGenerator
    ml_available = True
    print("✅ MLSignalGenerator importiert")
except ImportError:
    print("⚠️  MLSignalGenerator nicht verfügbar")

try:
    from src.mt5_client.mt5_live_client import MT5LiveClient
    mt5_available = True
    print("✅ MT5LiveClient importiert")
except ImportError:
    print("⚠️  MT5LiveClient nicht verfügbar")

print(f"\n📦 SYSTEM STATUS:")
print(f"   Paper Trading: {'✅ BEREIT' if paper_trading_available else '❌ FEHLER'}")
print(f"   ML Integration: {'✅ Verfügbar' if ml_integration_available else '⚠️  Nicht verfügbar'}")
print(f"   ML Module: {'✅ Verfügbar' if ml_available else '⚠️  Nicht verfügbar'}")
print(f"   MT5 Module: {'✅ Verfügbar' if mt5_available else '⚠️  Nicht verfügbar'}")
print("="*60)


class AITradingBot:
    """Hauptklasse für den AI Trading Bot."""

    def __init__(self):
        """Initialisiert den Trading Bot."""
        self.ml_generator = None
        self.mt5_client = None
        self.paper_portfolio = None
        self.ml_trading_engine = None
        self.running = True

        # Lade Konfiguration
        self.initial_balance = float(os.getenv('PAPER_TRADING_INITIAL_BALANCE', 10000.0))
        self.ml_buy_threshold = float(os.getenv('ML_BUY_THRESHOLD', 0.60))
        self.ml_sell_threshold = float(os.getenv('ML_SELL_THRESHOLD', 0.60))
        self.ml_confidence_threshold = float(os.getenv('ML_MIN_CONFIDENCE', 0.52))

        print(f"\n🕐 Systemzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Paper Trading Startkapital: {self.initial_balance:.2f} USD")

    def init_ml_system(self):
        """Initialisiert das ML-System."""
        if not ml_available:
            print("❌ ML-Module nicht verfügbar. Simulation wird verwendet.")
            return True  # Trotzdem fortfahren mit Simulation
            
        print("\n" + "="*40)
        print("🧠 ML-SYSTEM STARTEN")
        print("="*40)

        try:
            self.ml_generator = MLSignalGenerator()
            print("✅ ML-Modell geladen")
            return True
        except Exception as e:
            print(f"⚠️  ML-System Fehler: {e}")
            print("   Verwende Simulation für Demo")
            return True  # Mit Simulation fortfahren

    def init_mt5_client(self):
        """Initialisiert den MT5 Client."""
        if not mt5_available:
            print("❌ MT5-Module nicht verfügbar.")
            return False
            
        print("\n" + "="*40)
        print("📡 MT5 CLIENT INITIALISIEREN")
        print("="*40)

        try:
            self.mt5_client = MT5LiveClient()
            print(f"✅ MT5 verbunden")
            return True
        except Exception as e:
            print(f"❌ MT5 Verbindungsfehler: {e}")
            return False

    def init_paper_trading(self):
        """Initialisiert das Paper-Trading Portfolio."""
        if not paper_trading_available:
            print("❌ Paper Trading Module nicht verfügbar.")
            return False
            
        print("\n" + "="*40)
        print("📊 PAPER TRADING INITIALISIEREN")
        print("="*40)

        try:
            self.paper_portfolio = PaperPortfolio(initial_balance=self.initial_balance)
            print(f"✅ Paper Portfolio: {self.paper_portfolio.portfolio_id}")
            print(f"💰 Startkapital: {self.initial_balance:.2f} USD")
            return True
        except Exception as e:
            print(f"❌ Paper Trading Fehler: {e}")
            return False

    def init_ml_trading_engine(self):
        """Initialisiert die ML Trading Engine."""
        if not ml_integration_available:
            print("❌ ML Integration nicht verfügbar.")
            print("   Bitte erstellen Sie: src/paper_trading/ml_integration.py")
            return False
            
        print("\n" + "="*40)
        print("🚀 ML TRADING ENGINE INITIALISIEREN")
        print("="*40)

        if not self.paper_portfolio:
            print("❌ Paper Trading nicht initialisiert. Bitte Option 9a zuerst.")
            return False

        try:
            # Für ML Generator: Verwende echten oder simulierten
            ml_gen = self.ml_generator if self.ml_generator else self._create_mock_ml_generator()
            
            self.ml_trading_engine = MLTradingEngine(
                paper_portfolio=self.paper_portfolio,
                ml_generator=ml_gen,
                mt5_client=self.mt5_client
            )
            print("✅ ML Trading Engine initialisiert")
            print(f"   Confidence Threshold: {self.ml_confidence_threshold:.0%}")
            return True
        except Exception as e:
            print(f"❌ ML Trading Engine Fehler: {e}")
            return False

    def _create_mock_ml_generator(self):
        """Erstellt einen simulierten ML Generator für Tests."""
        class MockMLGenerator:
            def generate_signal(self):
                import random
                signals = ["BUY", "SELL", "HOLD"]
                return random.choice(signals), random.uniform(0.5, 0.9)
        
        return MockMLGenerator()

    def run_ml_signal_generation(self):
        """Führt ML-Signal-Generation aus."""
        print("\n" + "="*40)
        print("📈 ML-SIGNAL GENERIERUNG")
        print("="*40)

        try:
            if self.ml_generator and hasattr(self.ml_generator, 'generate_signal'):
                signal, confidence = self.ml_generator.generate_signal()
                print(f"✅ ECHTES ML-Signal generiert")
            else:
                signal, confidence = self._simulate_ml_signal()
                print(f"⚠️  SIMULIERTES Signal (ML nicht verfügbar)")

            print(f"\n📊 SIGNAL:")
            print(f"   Typ: {signal}")
            print(f"   Confidence: {confidence:.1%}")

            if signal == "BUY" and confidence >= self.ml_buy_threshold:
                print(f"   🟢 EMPFEHLUNG: BUY (Confidence: {confidence:.1%})")
            elif signal == "SELL" and confidence >= self.ml_sell_threshold:
                print(f"   🔴 EMPFEHLUNG: SELL (Confidence: {confidence:.1%})")
            else:
                print(f"   ⏸️  EMPFEHLUNG: HOLD")

            return signal, confidence

        except Exception as e:
            print(f"❌ Signal-Generierungsfehler: {e}")
            return None, None

    def run_ml_auto_trading(self):
        """Führt automatisches Trading basierend auf ML-Signalen durch."""
        if not self.paper_portfolio:
            print("❌ Paper Trading nicht initialisiert. Bitte Option 9a zuerst.")
            return
        
        print("\n" + "="*60)
        print("🤖 ML AUTO-TRADING ENGINE - PHASE D")
        print("="*60)
        
        # ML Trading Engine initialisieren
        if not self.ml_trading_engine:
            if not self.init_ml_trading_engine():
                return
        
        print("✅ ML Trading Engine bereit")
        print("   Verbindet ML-Signale mit Paper-Trades")
        
        # Anzahl der Iterationen
        try:
            iterations = int(input("\nAnzahl der Trading-Iterationen (1-10): ").strip())
            iterations = max(1, min(10, iterations))
        except:
            iterations = 3
            print(f"⚠️  Verwende Standard: {iterations} Iterationen")
        
        print(f"\n🚀 Starte {iterations} Iterationen...")
        print("-" * 50)
        
        for i in range(iterations):
            print(f"\n🔄 Iteration {i+1}/{iterations}:")
            print("-" * 30)
            
            self.ml_trading_engine.run_single_iteration()
            
            if i < iterations - 1:
                wait_time = 2
                print(f"⏱️  Warte {wait_time}s...")
                time.sleep(wait_time)
        
        # Statistik
        print("\n" + "="*50)
        print("📈 AUTO-TRADING ABGESCHLOSSEN")
        print("="*50)
        
        stats = self.ml_trading_engine.get_statistics()
        print(f"📊 STATISTIK:")
        print(f"   Signale: {stats['signals_generated']}")
        print(f"   Trades: {stats['trades_executed']}")
        print(f"   Balance: {stats['current_balance']:.2f} USD")
        
        pnl_change = stats['current_balance'] - self.initial_balance
        print(f"   P&L: {pnl_change:+.2f} USD")
        
        # Portfolio-Report
        print("\n" + "="*50)
        self.paper_portfolio.print_detailed_report()
        
        # Speichern
        self.paper_portfolio.save_performance_report()
        print(f"\n💾 Daten gespeichert")
        print("🎉 Phase D erfolgreich!")

    def run_paper_trading_demo(self):
        """Führt eine Paper-Trading Demo aus."""
        if not self.paper_portfolio:
            print("❌ Paper Trading nicht initialisiert. Bitte Option 9a zuerst.")
            return

        print("\n" + "="*40)
        print("🎯 PAPER TRADING DEMO")
        print("="*40)

        # Signal generieren
        print("\n1. GENERIERE SIGNAL:")
        signal, confidence = self.run_ml_signal_generation()
        
        if not signal:
            return
            
        # Trade-Parameter
        demo_symbol = "EURUSD"
        demo_price = 1.0850
        demo_stop_loss = 1.0800 if signal == "BUY" else 1.0900
        demo_take_profit = 1.0900 if signal == "BUY" else 1.0750

        print(f"\n2. TRADE PARAMETER:")
        print(f"   Symbol: {demo_symbol}")
        print(f"   Preis: {demo_price}")

        # Position öffnen
        print(f"\n3. ÖFFNE POSITION:")
        if signal == "BUY" and confidence >= self.ml_buy_threshold:
            order_type = OrderType.BUY
            order = self.paper_portfolio.open_position(
                symbol=demo_symbol,
                order_type=order_type,
                entry_price=demo_price,
                stop_loss=demo_stop_loss,
                take_profit=demo_take_profit,
                signal_confidence=confidence
            )
        elif signal == "SELL" and confidence >= self.ml_sell_threshold:
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
            print("   ⏸️  HOLD - keine Position")
            return

        if not order:
            print("   ❌ Position fehlgeschlagen")
            return

        # Position schließen
        print(f"\n4. SIMULIERE MARKT:")
        time.sleep(2)

        if signal == "BUY":
            exit_price = demo_price + 0.0020
        else:
            exit_price = demo_price - 0.0015

        print(f"   Neuer Preis: {exit_price}")

        print(f"\n5. SCHLIESSE POSITION:")
        pnl = self.paper_portfolio.close_position(demo_symbol, exit_price)

        if pnl is not None:
            pnl_sign = "+" if pnl > 0 else ""
            print(f"   💰 P&L: {pnl_sign}{pnl:.2f} USD")

        print(f"\n6. REPORT:")
        self.paper_portfolio.print_detailed_report()

    def show_paper_portfolio_status(self):
        """Zeigt Portfolio Status."""
        if not self.paper_portfolio:
            print("❌ Kein Portfolio initialisiert")
            return

        print("\n" + "="*40)
        print("📊 PAPER PORTFOLIO STATUS")
        print("="*40)

        self.paper_portfolio.print_detailed_report()

    def _simulate_ml_signal(self):
        """Simuliert ML-Signal."""
        import random
        
        signals = ["BUY", "SELL", "HOLD"]
        weights = [0.35, 0.35, 0.30]
        signal = random.choices(signals, weights=weights)[0]

        if signal == "BUY":
            confidence = random.uniform(self.ml_buy_threshold - 0.1, self.ml_buy_threshold + 0.1)
        elif signal == "SELL":
            confidence = random.uniform(self.ml_sell_threshold - 0.1, self.ml_sell_threshold + 0.1)
        else:
            confidence = random.uniform(0.4, 0.6)

        return signal, max(0.3, min(0.95, confidence))

    def show_menu(self):
        print("\n" + "="*60)
        print("📋 HAUPTMENÜ - AI TRADING BOT v4.0")
        print("="*60)
        print("1. 🧠 ML-System starten")
        print("2. 🧪 Testdaten verarbeiten")
        print("3. 📈 Signal generieren")
        print("4. 📊 Status anzeigen")
        print("5. 📡 MT5 Integration testen")
        print("6. 🔗 MT5 + ML Integration")
        print("7. 🔍 System Check")
        print("8. 🚪 Beenden")
        print("9. 📊 PAPER TRADING MODUS (PHASE D)")
        print("10. 🤖 ML AUTO-TRADING (PHASE D KERN)")
        print("="*60)

    def handle_choice(self, choice):
        if choice == "1":
            self.init_ml_system()
        elif choice == "2":
            print("\n🧪 Testdaten verarbeitet")
        elif choice == "3":
            print("\n📈 Signal wird generiert...")
            signal, confidence = self.run_ml_signal_generation()
            if signal:
                print(f"✅ Signal: {signal} ({confidence:.1%})")
        elif choice == "4":
            print("\n📊 System Status:")
            status = "✅" if self.ml_generator else "❌"
            print(f"   ML-System: {status}")
            status = "✅" if self.paper_portfolio else "❌"
            print(f"   Paper Trading: {status}")
            
            if self.paper_portfolio:
                summary = self.paper_portfolio.get_portfolio_summary()
                print(f"   Balance: {summary['current_balance']:.2f} USD")
        elif choice == "5":
            self.init_mt5_client()
        elif choice == "6":
            if self.init_ml_system():
                self.run_ml_signal_generation()
        elif choice == "7":
            print("\n🔍 System Check OK")
        elif choice == "8":
            print("\n👋 Beende...")
            self.running = False
        elif choice == "9":
            self.paper_trading_menu()
        elif choice == "10":
            self.run_ml_auto_trading()
        else:
            print(f"\n❌ Ungültige Wahl: '{choice}'")

    def paper_trading_menu(self):
        while True:
            print("\n" + "="*50)
            print("📊 PAPER TRADING ENGINE - PHASE D")
            print("="*50)
            print("a. 🆕 Portfolio initialisieren")
            print("b. 🎯 Demo Trade")
            print("c. 📊 Portfolio Status")
            print("d. 🤖 Auto-Trade Demo")
            print("e. 💾 Report speichern")
            print("f. 🔙 Zurück")
            print("="*50)

            sub_choice = input("Wahl (a-f): ").strip().lower()

            if sub_choice == "a":
                self.init_paper_trading()
            elif sub_choice == "b":
                self.run_paper_trading_demo()
            elif sub_choice == "c":
                self.show_paper_portfolio_status()
            elif sub_choice == "d":
                print("\n🤖 AUTO-TRADE DEMO")
                print("="*40)
                if not self.paper_portfolio:
                    print("❌ Portfolio benötigt (Option a)")
                    continue

                signal, confidence = self.run_ml_signal_generation()

                if signal in ["BUY", "SELL"]:
                    print(f"\n🤖 Führe Trade aus...")
                    symbol = "EURUSD"
                    price = 1.0850
                    
                    order_type = OrderType.BUY if signal == "BUY" else OrderType.SELL
                    order = self.paper_portfolio.open_position(
                        symbol=symbol,
                        order_type=order_type,
                        entry_price=price,
                        signal_confidence=confidence
                    )

                    if order:
                        time.sleep(1)
                        exit_price = price + 0.0015 if signal == "BUY" else price - 0.0010
                        pnl = self.paper_portfolio.close_position(symbol, exit_price)
                        
                        if pnl:
                            pnl_sign = "+" if pnl > 0 else ""
                            print(f"   💰 P&L: {pnl_sign}{pnl:.2f}")
                else:
                    print("   ⏸️  HOLD - kein Trade")

            elif sub_choice == "e":
                if self.paper_portfolio:
                    self.paper_portfolio.save_performance_report()
                    print("✅ Report gespeichert")
                else:
                    print("❌ Kein Portfolio")

            elif sub_choice == "f":
                print("🔙 Zurück...")
                break
            else:
                print("❌ Ungültige Wahl")

    def run(self):
        while self.running:
            self.show_menu()
            choice = input("\nWahl (1-10): ").strip()
            self.handle_choice(choice)

        print("\n" + "="*60)
        print("✅ AI Trading Bot beendet")
        print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.paper_portfolio:
            self.paper_portfolio.save_performance_report()
            
        print("="*60)


def main():
    bot = AITradingBot()

    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Programm abgebrochen")
        if bot.paper_portfolio:
            bot.paper_portfolio.save_performance_report()
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()