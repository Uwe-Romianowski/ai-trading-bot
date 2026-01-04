#!/usr/bin/env python3
"""
🤖 AI TRADING BOT v4.1 - MAIN CONTROLLER
=========================================
Hauptsteuerung für den AI Trading Bot mit ML-Signalen,
MT5 Integration und Paper-Trading Engine.

Phasenübersicht:
- Phase A-C: ML Research & MT5 Integration ✅
- Phase D: Paper-Trading Engine ✅
- Phase E: MT5 Live-Demo Integration 🚧 (Woche 2)
"""

import sys
import os
import time
import json
from datetime import datetime

# Pfade für Importe hinzufügen
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def clear_screen():
    """Löscht den Bildschirm (plattformunabhängig)."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Gibt den Header des Trading Bots aus."""
    clear_screen()
    print("=" * 60)
    print("🤖 AI TRADING BOT v4.1 - PRODUKTIONSBEREIT")
    print("=" * 60)
    print("📅 Datum:", datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    print("📍 Phase D: ✅ Paper-Trading Engine (ML Auto-Trading)")
    print("📍 Phase E: 🚧 MT5 Live-Demo Integration (Woche 2)")
    print("=" * 60)

def show_main_menu():
    """Zeigt das Hauptmenü an."""
    print("\n📋 HAUPTMENÜ - WÄHLEN SIE EINE OPTION:")
    print("-" * 40)
    print("1.  🧠 ML-System starten")
    print("2.  🧪 Testdaten verarbeiten")
    print("3.  📈 Signal generieren")
    print("4.  📊 Status anzeigen")
    print("5.  📡 MT5 Integration testen")
    print("6.  🔗 MT5 + ML Integration")
    print("7.  🔍 System Check")
    print("8.  🚪 Beenden")
    print("9.  📊 PAPER TRADING MODUS (PHASE D)")
    print("10. 🤖 ML AUTO-TRADING (PHASE D KERN)")
    print("11. 🔗 MT5 LIVE TEST (PHASE E WOCHE 1)")
    print("12. 🌉 LIVE TRADING BRIDGE (PHASE E WOCHE 2)")
    print("-" * 40)

def run_ml_system():
    """Option 1: Startet das ML-System."""
    print("\n🧠 ML-SYSTEM STARTEN")
    print("-" * 30)
    try:
        from src.ml_model import TradingModel
        model = TradingModel()
        model.load_model()
        print("✅ ML-Modell geladen und bereit.")
    except ImportError:
        print("❌ ML-Modell nicht gefunden. Bitte zuerst trainieren.")
    except Exception as e:
        print(f"❌ Fehler: {e}")
    input("\nDrücke Enter zum Fortfahren...")

def process_test_data():
    """Option 2: Verarbeitet Testdaten."""
    print("\n🧪 TESTDATEN VERARBEITEN")
    print("-" * 30)
    try:
        from src.data_processor import DataProcessor
        processor = DataProcessor()
        processor.load_data('data/raw/eurusd_2024.csv')
        print("✅ Daten erfolgreich geladen und verarbeitet.")
    except Exception as e:
        print(f"❌ Fehler: {e}")
    input("\nDrücke Enter zum Fortfahren...")

def generate_signal():
    """Option 3: Generiert ein Trading-Signal."""
    print("\n📈 SIGNAL GENERIEREN")
    print("-" * 30)
    try:
        from src.signal_generator import generate_trading_signal
        signal, confidence = generate_trading_signal()
        print(f"✅ Signal generiert: {signal}")
        print(f"   Confidence: {confidence:.1f}%")
    except Exception as e:
        print(f"❌ Fehler: {e}")
    input("\nDrücke Enter zum Fortfahren...")

def show_status():
    """Option 4: Zeigt System-Status an."""
    print("\n📊 SYSTEM STATUS")
    print("-" * 30)
    
    # Prüfe wichtige Module
    modules = {
        'ML Model': 'src.ml_model',
        'Data Processor': 'src.data_processor',
        'MT5 Integration': 'src.mt5_integration',
        'Paper Trading': 'src.paper_trading.portfolio'
    }
    
    for name, module in modules.items():
        try:
            __import__(module.replace('/', '.'))
            print(f"✅ {name}: Verfügbar")
        except ImportError:
            print(f"❌ {name}: Nicht verfügbar")
    
    # Dateisystem prüfen
    print("\n📁 DATEIEN:")
    important_files = [
        'requirements.txt',
        'data/raw/eurusd_2024.csv',
        'src/paper_trading/portfolio.py'
    ]
    
    for file in important_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"⚠️  {file} (fehlt)")
    
    input("\nDrücke Enter zum Fortfahren...")

def test_mt5_integration():
    """Option 5: Testet die MT5 Integration."""
    print("\n📡 MT5 INTEGRATION TESTEN")
    print("-" * 30)
    try:
        from src.mt5_integration import test_mt5_connection
        test_mt5_connection()
    except ImportError:
        print("❌ MT5 Integration nicht verfügbar.")
    except Exception as e:
        print(f"❌ Fehler: {e}")
    input("\nDrücke Enter zum Fortfahren...")

def run_mt5_ml_integration():
    """Option 6: MT5 + ML Integration."""
    print("\n🔗 MT5 + ML INTEGRATION")
    print("-" * 30)
    try:
        from src.mt5_integration import get_live_data
        from src.signal_generator import generate_trading_signal
        
        # Live-Daten holen
        data = get_live_data('EURUSD')
        print(f"✅ Live-Daten: {data}")
        
        # Signal generieren
        signal, confidence = generate_trading_signal()
        print(f"✅ Signal: {signal} ({confidence:.1f}%)")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
    input("\nDrücke Enter zum Fortfahren...")

def system_check():
    """Option 7: Führt einen System-Check durch."""
    print("\n🔍 SYSTEM CHECK")
    print("-" * 30)
    
    # Python Version
    print(f"🐍 Python Version: {sys.version}")
    
    # Wichtige Pakete prüfen
    packages = ['pandas', 'numpy', 'sklearn', 'tensorflow', 'MetaTrader5', 'python-dotenv']
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installiert")
        except ImportError:
            print(f"❌ {package}: Nicht installiert")
    
    # Verzeichnisse prüfen
    print("\n📁 VERZEICHNISSE:")
    dirs = ['data', 'data/raw', 'data/paper_trading', 'src', 'src/paper_trading', 'src/live_trading']
    
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"⚠️  {dir_path}/ (fehlt)")
    
    input("\nDrücke Enter zum Fortfahren...")

def run_paper_trading():
    """Option 9: Startet den Paper-Trading Modus."""
    print("\n📊 PAPER TRADING MODUS - PHASE D")
    print("=" * 40)
    
    try:
        from src.paper_trading.portfolio import Portfolio
        from src.paper_trading.ml_integration import MLTradingEngine
        
        # Portfolio initialisieren
        portfolio = Portfolio()
        print(f"✅ Portfolio initialisiert: {portfolio.portfolio_id}")
        print(f"   Startkapital: ${portfolio.balance:.2f}")
        
        # ML Engine initialisieren
        engine = MLTradingEngine(portfolio)
        print("✅ ML Trading Engine initialisiert")
        
        # User Input für Iterationen
        while True:
            try:
                iterations = int(input("\nAnzahl der Trading-Iterationen (1-10): "))
                if 1 <= iterations <= 10:
                    break
                print("❌ Bitte eine Zahl zwischen 1 und 10 eingeben.")
            except ValueError:
                print("❌ Ungültige Eingabe.")
        
        # Trading starten
        print(f"\n🚀 Starte {iterations} Iterationen...")
        engine.run_auto_trading(iterations)
        
    except ImportError as e:
        print(f"❌ Paper-Trading Module nicht gefunden: {e}")
        print("   Stellen Sie sicher, dass Phase D korrekt implementiert ist.")
    except Exception as e:
        print(f"❌ Fehler: {e}")
    
    input("\nDrücke Enter zum Fortfahren...")

def run_ml_auto_trading():
    """Option 10: ML Auto-Trading Engine (Phase D Kern)."""
    print("\n" + "=" * 60)
    print("🤖 ML AUTO-TRADING ENGINE - PHASE D")
    print("=" * 60)
    
    try:
        # Import aus ml_integration.py (direkter Import)
        import sys
        import os
        
        # Pfad für Import hinzufügen
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'paper_trading'))
        
        from ml_integration import run_ml_trading
        
        # Auto-Trading starten
        run_ml_trading()
        
    except ImportError as e:
        print(f"❌ ML Integration nicht gefunden: {e}")
        print("   Bitte sicherstellen, dass 'src/paper_trading/ml_integration.py' existiert.")
    except Exception as e:
        print(f"❌ Fehler beim ML Auto-Trading: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nDrücke Enter zum Fortfahren...")

def test_mt5_live_connection():
    """Option 11: Testet die MT5 Live-Verbindung (Phase E Woche 1)."""
    print("\n" + "=" * 60)
    print("🔗 MT5 LIVE CONNECTION TEST - PHASE E WOCHE 1")
    print("=" * 60)
    
    try:
        from src.live_trading.mt5_client import quick_test
        
        print("🚀 Starte MT5 Live-Connection Test...")
        print("-" * 50)
        
        # Schnelltest durchführen
        success = quick_test()
        
        if success:
            print("\n✅ Phase E - Woche 1: MT5 Live Client funktioniert!")
            print("   Nächste Schritte:")
            print("   1. Live-Daten in Paper-Trading integrieren")
            print("   2. Order Executor entwickeln (Woche 2)")
            print("   3. Dashboard implementieren (Woche 4)")
        else:
            print("\n❌ MT5 Live Connection fehlgeschlagen.")
            print("   Mögliche Ursachen:")
            print("   - MT5 Terminal nicht geöffnet")
            print("   - Falsche Login-Daten")
            print("   - Keine Internetverbindung")
            print("   - MetaTrader5 Package nicht installiert")
            print("\n💡 Lösung: 'pip install MetaTrader5' und MT5 Terminal öffnen")
    
    except ImportError as e:
        print(f"❌ Live-Trading Module nicht gefunden: {e}")
        print("\n📋 Bitte folgende Schritte ausführen:")
        print("   1. 'pip install MetaTrader5' ausführen")
        print("   2. Ordner 'src/live_trading/' erstellen")
        print("   3. 'mt5_client.py' im Ordner ablegen")
        print("\n🔧 Schnellfix:")
        print("   mkdir src\\live_trading")
        print("   type nul > src\\live_trading\\__init__.py")
        print("   notepad src\\live_trading\\mt5_client.py")
        
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nDrücke Enter zum Fortfahren...")

def run_live_trading_bridge():
    """Option 12: Startet die Live-Trading Bridge (Phase E Woche 2)."""
    print("\n" + "=" * 60)
    print("🌉 LIVE TRADING BRIDGE - PHASE E WOCHE 2")
    print("=" * 60)
    print("⚠️  WICHTIG: Dies führt echte Orders im Demo-Account aus!")
    print("   Verwendet 0.01 Lots (Minimum) für Testing.")
    print("=" * 60)
    
    try:
        from src.live_trading.live_bridge import LiveTradingBridge
        bridge = LiveTradingBridge()
        
        # User Menu für Live Trading
        print("\n📋 LIVE TRADING OPTIONEN:")
        print("   1. Order Execution Test (eine Mini-Order)")
        print("   2. Live Trading Session (mit ML-Signalen)")
        print("   3. Zurück zum Hauptmenü")
        
        sub_choice = input("\n📝 Wahl (1-3): ").strip()
        
        if sub_choice == "1":
            # Order Execution Test
            print("\n🧪 Starte Order Execution Test...")
            success = bridge.test_order_execution()
            if success:
                print("\n✅ Order Execution Test erfolgreich!")
            else:
                print("\n❌ Order Execution Test fehlgeschlagen")
                
        elif sub_choice == "2":
            # Live Trading Session
            try:
                iterations = int(input("\nAnzahl der Iterationen (1-5): "))
                if 1 <= iterations <= 5:
                    print(f"\n🚀 Starte Live Trading mit {iterations} Iterationen...")
                    print("⚠️  Achtung: Echte Demo-Orders werden ausgeführt!")
                    confirm = input("   Bestätigen? (j/n): ").strip().lower()
                    
                    if confirm == 'j' or confirm == 'y':
                        bridge.run_live_trading(iterations)
                    else:
                        print("❌ Abgebrochen")
                else:
                    print("❌ Bitte eine Zahl zwischen 1 und 5 eingeben.")
            except ValueError:
                print("❌ Ungültige Eingabe.")
        
        elif sub_choice == "3":
            print("↩️  Zurück zum Hauptmenü")
        else:
            print("❌ Ungültige Auswahl")
            
    except ImportError as e:
        print(f"❌ Live-Trading Module nicht gefunden: {e}")
        print("💡 Bitte erstelle die Dateien:")
        print("   src/live_trading/order_executor.py")
        print("   src/live_trading/live_bridge.py")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
    
    input("\nDrücke Enter zum Fortfahren...")

def main():
    """Hauptfunktion des Trading Bots."""
    while True:
        try:
            print_header()
            show_main_menu()
            
            choice = input("\n📝 Wahl (1-12): ").strip()
            
            if choice == "1":
                run_ml_system()
            elif choice == "2":
                process_test_data()
            elif choice == "3":
                generate_signal()
            elif choice == "4":
                show_status()
            elif choice == "5":
                test_mt5_integration()
            elif choice == "6":
                run_mt5_ml_integration()
            elif choice == "7":
                system_check()
            elif choice == "8":
                print("\n👋 Auf Wiedersehen! Trading Bot wird beendet.")
                sys.exit(0)
            elif choice == "9":
                run_paper_trading()
            elif choice == "10":
                run_ml_auto_trading()
            elif choice == "11":
                test_mt5_live_connection()
            elif choice == "12":
                run_live_trading_bridge()
            else:
                print("❌ Ungültige Auswahl. Bitte 1-12 wählen.")
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Programm durch Benutzer abgebrochen.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Kritischer Fehler: {e}")
            import traceback
            traceback.print_exc()
            input("\nDrücke Enter zum Fortfahren...")

if __name__ == "__main__":
    main()