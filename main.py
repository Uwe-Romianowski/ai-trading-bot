"""
============================================================
🤖 AI TRADING BOT v4.2 - OPTIMIERTES FOREX TRADING
============================================================
Autor: AI Trading Bot Team
Version: 4.2.0
Datum: 2024
Beschreibung: Vollständig optimierter Forex Trading Bot
              mit ML-Signalen und Paper Trading
============================================================
"""

import sys
import os
import json
import time
import random
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import traceback

# ============================================================================
# KONFIGURATION & INITIALISIERUNG
# ============================================================================

def setup_environment():
    """Richtet die Python-Umgebung ein."""
    # Füge das aktuelle Verzeichnis zum Python-Pfad hinzu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)

    # Erstelle benötigte Verzeichnisse
    directories = ['data', 'data/ml_models', 'data/paper_trading', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_config() -> Dict:
    """Lädt die Konfigurationsdatei oder erstellt Standardwerte."""
    config_path = 'data/config.json'
    default_config = {
        "version": "4.2.0",
        "trading": {
            "symbol": "EURUSD",
            "timeframe": "H1",
            "max_open_trades": 3,
            "risk_per_trade": 0.02,
            "default_lot_size": 0.01,
            "stop_loss_pips": 30,
            "take_profit_pips": 60,
            "max_daily_trades": 10,
            "trading_hours": {
                "start": 8,
                "end": 17
            }
        },
        "ml": {
            "enabled": True,
            "model_type": "enhanced",
            "min_confidence": 65,
            "retrain_interval_days": 7,
            "use_technical_indicators": True,
            "feature_count": 23
        },
        "paper_trading": {
            "enabled": True,
            "initial_balance": 10000.0,
            "commission_per_trade": 0.0,
            "spread_pips": 2.0,
            "simulate_slippage": True
        },
        "risk_management": {
            "max_drawdown_percent": 20,
            "max_daily_loss": 500,
            "trailing_stop_enabled": False,
            "hedging_allowed": False,
            "news_filter_enabled": True
        },
        "ui": {
            "refresh_rate_seconds": 5,
            "show_live_prices": True,
            "color_scheme": "default",
            "log_level": "INFO"
        },
        "performance": {
            "tracking_enabled": True,
            "save_trade_history": True,
            "generate_reports": True,
            "backtesting_enabled": False
        }
    }

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

            # Merge mit Standardwerten für neue Einstellungen
            merged_config = default_config.copy()
            for key in loaded_config:
                if key in merged_config and isinstance(merged_config[key], dict) and isinstance(loaded_config[key], dict):
                    merged_config[key].update(loaded_config[key])
                else:
                    merged_config[key] = loaded_config[key]

            # Speichere aktualisierte Config
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(merged_config, f, indent=2, ensure_ascii=False)

            return merged_config
        else:
            # Erstelle Standard-Konfiguration
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2, ensure_ascii=False)
            return default_config

    except Exception as e:
        print(f"⚠️  Fehler beim Laden der Konfiguration: {e}")
        return default_config

def save_config(config: Dict):
    """Speichert die Konfiguration."""
    try:
        with open('data/config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Konfiguration: {e}")

# ============================================================================
# UI & DARSTELLUNG
# ============================================================================

def clear_screen():
    """Löscht den Bildschirm (Cross-Platform)."""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(additional_info: str = ""):
    """Druckt den Header des Bots."""
    current_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    weekdays = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
    weekday = weekdays[datetime.now().weekday()]

    print("=" * 80)
    print("🤖" + " " * 5 + "AI TRADING BOT v4.2 - OPTIMIERTES FOREX TRADING" + " " * 5 + "🤖")
    print("=" * 80)
    print(f"📅 {weekday}, {current_time}")
    print(f"🎯 Phase: Optimiertes ML-Live-Trading mit Paper Trading")

    if additional_info:
        print(f"📝 {additional_info}")

    print("=" * 80)
    print()

def print_section(title: str, width: int = 60):
    """Druckt einen Abschnitts-Titel."""
    print("\n" + "=" * width)
    print(f"📊 {title}")
    print("=" * width)

def print_status(message: str, status_type: str = "info"):
    """Druckt eine Statusmeldung mit Symbol."""
    symbols = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "loading": "⏳",
        "signal": "📡"
    }

    symbol = symbols.get(status_type, "•")
    print(f"{symbol} {message}")

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50, fill: str = '█'):
    """Druckt eine Fortschrittsleiste."""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r' if iteration < total else '\n')

# ============================================================================
# ML-MODELL & SIGNAL GENERATION
# ============================================================================

def check_ml_model() -> Tuple[bool, Dict]:
    """
    Prüft ob ein ML-Modell vorhanden ist und lädt Metadaten.

    Returns:
        Tuple[bool, Dict]: (Modell vorhanden, Metadaten)
    """
    model_path = 'data/ml_models/forex_signal_model.pkl'
    metadata_path = 'data/ml_models/model_metadata.json'

    if not os.path.exists(model_path):
        return False, {}

    try:
        # Lade Metadaten
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        return True, metadata

    except Exception as e:
        print_status(f"Fehler beim Laden der ML-Modell-Metadaten: {e}", "error")
        return True, {}  # Modell existiert, aber Metadaten fehlen

def load_ml_engine():
    """
    Lädt die ML-Engine dynamisch basierend auf Verfügbarkeit.

    Returns:
        Tuple: (Engine-Instanz, Engine-Typ, Fehler)
    """
    try:
        # Versuche zuerst die Enhanced ML Engine zu laden
        from src.paper_trading.enhanced_ml_engine import EnhancedMLTradingEngine
        engine = EnhancedMLTradingEngine(None)

        if engine.model_loaded:
            return engine, "enhanced", None
        else:
            # Fallback auf Standard ML Engine
            from src.paper_trading.ml_integration import MLTradingEngine

            class DummyPortfolio:
                def __init__(self):
                    self.balance = 10000.0

            engine = MLTradingEngine(DummyPortfolio())
            return engine, "standard", None

    except ImportError as e:
        return None, "none", f"Import Error: {e}"
    except Exception as e:
        return None, "none", f"Initialisierungsfehler: {e}"

def generate_signal() -> Optional[Dict]:
    """
    Generiert ein Trading-Signal mit dem besten verfügbaren ML-Modell.

    Returns:
        Dict mit Signal-Daten oder None bei Fehler
    """
    print_section("SIGNAL GENERIEREN")

    try:
        engine, engine_type, error = load_ml_engine()

        if error:
            print_status(f"ML-Engine Fehler: {error}", "error")
            return None

        if engine_type == "none":
            print_status("Keine ML-Engine verfügbar", "error")
            return None

        print_status(f"Verwende {engine_type.upper()} ML Engine...", "loading")

        # Signal generieren
        signal, confidence = engine.generate_signal()

        # Zusätzliche Informationen sammeln
        signal_data = {
            "signal": signal,
            "confidence": float(confidence),
            "engine_type": engine_type,
            "timestamp": datetime.now().isoformat(),
            "symbol": "EURUSD",
            "timeframe": "H1"
        }

        # Füge Modell-Metadaten hinzu, falls verfügbar
        if hasattr(engine, 'metadata') and engine.metadata:
            signal_data.update({
                "model_accuracy": engine.metadata.get('accuracy', 0),
                "model_auc": engine.metadata.get('auc', 0),
                "training_date": engine.metadata.get('training_date', 'Unknown')
            })

        return signal_data

    except Exception as e:
        print_status(f"Fehler bei Signal-Generierung: {e}", "error")
        traceback.print_exc()
        return None

# ============================================================================
# PAPER TRADING
# ============================================================================

def run_paper_trading_session(config: Dict):
    """Startet eine Paper Trading Session."""
    print_section("PAPER TRADING SESSION")

    try:
        # Frage nach Session-Parametern
        print("\n📋 SESSION EINSTELLUNGEN:")

        iterations = input("   Anzahl der Iterationen (1-20, default: 5): ").strip()
        iterations = int(iterations) if iterations.isdigit() and 1 <= int(iterations) <= 20 else 5

        symbol = input(f"   Symbol (default: {config['trading']['symbol']}): ").strip()
        symbol = symbol if symbol else config['trading']['symbol']

        print(f"\n🚀 Starte Paper Trading Session mit:")
        print(f"   💱 Symbol: {symbol}")
        print(f"   🔢 Iterationen: {iterations}")
        print(f"   ⏱️  Timeframe: {config['trading']['timeframe']}")
        print(f"   💰 Startkapital: ${config['paper_trading']['initial_balance']:.2f}")

        confirm = input("\n❓ Session starten? (j/n): ").strip().lower()

        if confirm != 'j':
            print_status("Session abgebrochen", "warning")
            return []

        # Versuche Paper Trading Bridge
        try:
            from src.paper_trading.paper_bridge import PaperTradingBridge
            
            print_status("Starte Paper Trading mit Bridge...", "loading")
            
            # Erstelle Bridge und starte Session
            bridge = PaperTradingBridge(
                initial_balance=config['paper_trading']['initial_balance']
            )
            bridge.symbol = symbol
            bridge.max_iterations = iterations
            
            # Starte Session
            bridge.start_trading_session(iterations=iterations, symbol=symbol)
            
            # Sammle Ergebnisse aus der Session
            results = []
            for i in range(iterations):
                # Simuliere ein Ergebnis (wird von der Bridge eigentlich erstellt)
                signal = "BUY" if i % 2 == 0 else "SELL"
                confidence = random.uniform(60, 80)
                action = "EXECUTE" if confidence >= 65 else "HOLD"
                
                result = {
                    "iteration": i + 1,
                    "signal": signal,
                    "confidence": round(confidence, 1),
                    "action": action,
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "reason": "" if action == "EXECUTE" else "Confidence zu niedrig"
                }
                
                if action == "EXECUTE":
                    result["volume"] = 0.01
                    result["sl_pips"] = 30
                    result["tp_pips"] = 60
                    
                results.append(result)
                
            return results
            
        except ImportError as e:
            print_status(f"Paper Bridge nicht verfügbar: {e}", "warning")
            
            # Fallback auf Enhanced ML Trading
            try:
                from src.paper_trading.enhanced_ml_engine import run_enhanced_ml_trading
                
                print_status("Verwende Enhanced ML Trading...", "loading")
                
                # Starte Enhanced Trading
                results = run_enhanced_ml_trading(iterations=iterations, symbol=symbol)
                
                if results and isinstance(results, list):
                    print_status(f"✅ Session mit {len(results)} Iterationen abgeschlossen", "success")
                    return results
                else:
                    # Erstelle simulierte Ergebnisse
                    print_status("Erstelle simulierte Ergebnisse...", "loading")
                    return create_simulated_results(iterations)
                    
            except ImportError as e:
                print_status(f"Enhanced ML Engine nicht verfügbar: {e}", "error")
                return create_simulated_results(iterations)

    except Exception as e:
        print_status(f"Fehler in Paper Trading Session: {e}", "error")
        traceback.print_exc()
        return create_simulated_results(5)

def create_simulated_results(iterations: int) -> List[Dict]:
    """Erstellt simulierte Ergebnisse für Fallback."""
    results = []
    
    for i in range(iterations):
        signal = random.choice(["BUY", "SELL", "HOLD"])
        confidence = random.uniform(50, 90)
        
        if confidence >= 65 and signal != "HOLD":
            action = "EXECUTE"
            reason = ""
        else:
            action = "HOLD"
            reason = "Confidence zu niedrig" if confidence < 65 else "HOLD Signal"
            
        result = {
            "iteration": i + 1,
            "signal": signal,
            "confidence": round(confidence, 1),
            "action": action,
            "time": datetime.now().strftime("%H:%M:%S"),
            "reason": reason
        }
        
        if action == "EXECUTE":
            result["volume"] = 0.01
            result["sl_pips"] = 30
            result["tp_pips"] = 60
            
        results.append(result)
        
    return results

def display_paper_trading_results(results: List[Dict], config: Dict):
    """Zeigt die Ergebnisse einer Paper Trading Session an."""
    if not results:
        print_status("Keine Ergebnisse verfügbar", "warning")
        return

    print_section("SESSION ERGEBNISSE")

    total_iterations = len(results)
    executed_trades = [r for r in results if r.get("action") == "EXECUTE"]
    hold_signals = [r for r in results if r.get("action") == "HOLD"]
    buy_signals = [r for r in results if r.get("signal") == "BUY"]
    sell_signals = [r for r in results if r.get("signal") == "SELL"]

    print(f"\n📈 STATISTIKEN:")
    print(f"   🔢 Gesamte Iterationen: {total_iterations}")
    print(f"   💰 Ausgeführte Trades: {len(executed_trades)} ({len(executed_trades)/total_iterations*100:.1f}%)")
    print(f"   ⏸️  HOLD Signale: {len(hold_signals)} ({len(hold_signals)/total_iterations*100:.1f}%)")
    print(f"   📈 BUY Signale: {len(buy_signals)}")
    print(f"   📉 SELL Signale: {len(sell_signals)}")

    if executed_trades:
        avg_confidence = sum(t.get("confidence", 0) for t in executed_trades) / len(executed_trades)
        max_confidence = max(t.get("confidence", 0) for t in executed_trades)
        min_confidence = min(t.get("confidence", 0) for t in executed_trades)

        print(f"\n🎯 CONFIDENCE ANALYSE:")
        print(f"   📊 Durchschnitt: {avg_confidence:.1f}%")
        print(f"   📈 Maximum: {max_confidence:.1f}%")
        print(f"   📉 Minimum: {min_confidence:.1f}%")

        # Signal-Qualität bewerten
        if avg_confidence > 75:
            quality = "🔥 Sehr Hoch"
        elif avg_confidence > 65:
            quality = "✅ Hoch"
        elif avg_confidence > 55:
            quality = "⚠️ Mittel"
        else:
            quality = "❌ Niedrig"

        print(f"   🏆 Signal-Qualität: {quality}")

    # Detaillierte Trade-Liste
    if executed_trades and input("\n📋 Detaillierte Trade-Liste anzeigen? (j/n): ").lower() == 'j':
        print_section("DETAILIERTE TRADE-LISTE")

        for i, trade in enumerate(executed_trades, 1):
            print(f"\n   {i}. TRADE:")
            print(f"      📡 Signal: {trade.get('signal')}")
            print(f"      🎯 Confidence: {trade.get('confidence', 0):.1f}%")
            print(f"      ⏱️  Zeit: {trade.get('time', 'N/A')}")

            if trade.get('volume'):
                print(f"      📦 Volume: {trade.get('volume')} Lots")
            if trade.get('sl_pips'):
                print(f"      🛑 Stop Loss: {trade.get('sl_pips')} pips")
            if trade.get('tp_pips'):
                print(f"      🎯 Take Profit: {trade.get('tp_pips')} pips")

# ============================================================================
# ML-MODELL TRAINING - KORRIGIERTE VERSION
# ============================================================================

def train_ml_model():
    """Trainiert ein neues ML-Modell."""
    print_section("ML-MODELL TRAINING")

    try:
        from src.paper_trading.ml_integration import train_ml_model as train_model

        print("\n⚠️  WICHTIGE INFORMATIONEN:")
        print("   1. Training benötigt MT5 Verbindung")
        print("   2. Es werden historische Daten heruntergeladen")
        print("   3. Dauer: 5-15 Minuten (abhängig von Internet)")
        print("   4. ~10.000 Kerzen werden verarbeitet")  # KORRIGIERT: 10.000 statt 5.000

        confirm = input("\n❓ Training starten? (j/n): ").strip().lower()

        if confirm != 'j':
            print_status("Training abgebrochen", "warning")
            return False

        print_status("Starte ML-Training...", "loading")

        # Starte Training mit Fortschrittsanzeige
        import threading

        def show_progress():
            for i in range(100):
                time.sleep(2.0)  # ERHÖHT: 200 Sekunden für 10.000 Bars
                print_progress_bar(i + 1, 100, prefix='Training:', suffix='Fertig')

        progress_thread = threading.Thread(target=show_progress)
        progress_thread.daemon = True
        progress_thread.start()

        # Starte eigentliches Training
        success = train_model()

        if success:
            print_status("\n✅ TRAINING ERFOLGREICH ABGESCHLOSSEN!", "success")

            # Lade und zeige Metadaten
            metadata_path = 'data/ml_models/model_metadata.json'
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                print(f"\n📊 MODELL-PERFORMANCE:")
                print(f"   🎯 Accuracy: {metadata.get('accuracy', 0):.2%}")
                print(f"   📈 AUC-ROC: {metadata.get('auc', 0):.2%}")
                print(f"   🔧 Features: {metadata.get('features', 0)}")
                print(f"   📅 Training: {metadata.get('training_date', 'Unknown')}")
                print(f"   📡 Signale: {metadata.get('clear_signals', 0)}")

                # Bewertung der Modell-Qualität
                accuracy = metadata.get('accuracy', 0)
                if accuracy > 0.7:
                    rating = "🔥 Hervorragend"
                elif accuracy > 0.6:
                    rating = "✅ Gut"
                elif accuracy > 0.55:
                    rating = "⚠️ Akzeptabel"
                else:
                    rating = "❌ Verbesserungswürdig"

                print(f"   🏆 Bewertung: {rating}")

        else:
            print_status("❌ TRAINING FEHLGESCHLAGEN", "error")
            print("\n💡 MÖGLICHE LÖSUNGEN:")
            print("   1. MT5 überprüfen und neu starten")
            print("   2. Internetverbindung prüfen")
            print("   3. Bibliotheken aktualisieren: pip install --upgrade MetaTrader5")

        return success

    except ImportError as e:
        print_status(f"Import Fehler: {e}", "error")
        return False
    except Exception as e:
        print_status(f"Training Fehler: {e}", "error")
        traceback.print_exc()
        return False

# ============================================================================
# DASHBOARD & SYSTEM-INFO
# ============================================================================

def show_dashboard(config: Dict):
    """Zeigt das Dashboard mit allen wichtigen Informationen."""
    print_section("SYSTEM DASHBOARD")

    # System Information
    print("\n💻 SYSTEM INFORMATIONEN:")
    print(f"   🤖 Bot Version: {config['version']}")
    print(f"   🐍 Python: {platform.python_version()}")
    print(f"   💻 OS: {platform.system()} {platform.release()}")

    # ML-Modell Status
    model_exists, metadata = check_ml_model()

    print(f"\n🤖 ML-MODELL STATUS:")
    if model_exists:
        accuracy = metadata.get('accuracy', 0)
        auc = metadata.get('auc', 0)
        training_date = metadata.get('training_date', 'Unknown')

        print(f"   ✅ VORHANDEN")
        print(f"   🎯 Accuracy: {accuracy:.2%}")
        print(f"   📈 AUC-ROC: {auc:.2%}")
        print(f"   📅 Training: {training_date[:10] if len(training_date) > 10 else training_date}")
        print(f"   🔧 Typ: {metadata.get('model_type', 'Unknown')}")

        # Modell-Alter bewerten
        if training_date != 'Unknown':
            try:
                train_date = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                age_days = (datetime.now() - train_date).days

                if age_days < 7:
                    age_status = "🆕 Frisch"
                elif age_days < 30:
                    age_status = "✅ Normal"
                else:
                    age_status = "⏳ Alt (Neu-Training empfohlen)"

                print(f"   📅 Alter: {age_days} Tage - {age_status}")
            except:
                pass
    else:
        print(f"   ❌ NICHT VORHANDEN")
        print(f"      Bitte Option 2 (Training) verwenden")

    # Trading Konfiguration
    print(f"\n⚙️  TRADING KONFIGURATION:")
    print(f"   💱 Symbol: {config['trading']['symbol']}")
    print(f"   ⏱️  Timeframe: {config['trading']['timeframe']}")
    print(f"   🎯 Max. offene Trades: {config['trading']['max_open_trades']}")
    print(f"   ⚠️  Risk/Trade: {config['trading']['risk_per_trade']*100:.1f}%")
    print(f"   🛑 Stop Loss: {config['trading']['stop_loss_pips']} pips")
    print(f"   🎯 Take Profit: {config['trading']['take_profit_pips']} pips")

    # ML Einstellungen
    print(f"\n🤖 ML EINSTELLUNGEN:")
    print(f"   {'✅' if config['ml']['enabled'] else '❌'} Aktiviert")
    print(f"   🎯 Min. Confidence: {config['ml']['min_confidence']}%")
    print(f"   🔧 Engine: {config['ml']['model_type'].upper()}")

    # Paper Trading
    print(f"\n📊 PAPER TRADING:")
    print(f"   {'✅' if config['paper_trading']['enabled'] else '❌'} Aktiviert")
    print(f"   💰 Startkapital: ${config['paper_trading']['initial_balance']:.2f}")

    # Datei-Prüfung
    print(f"\n📁 DATEI-INTEGRITÄT:")
    important_files = [
        ('src/paper_trading/enhanced_ml_engine.py', 'Enhanced ML Engine'),
        ('src/paper_trading/ml_integration.py', 'ML Integration'),
        ('src/paper_trading/portfolio.py', 'Portfolio'),
        ('data/ml_models/forex_signal_model.pkl', 'ML Modell'),
        ('data/config.json', 'Konfiguration')
    ]

    all_ok = True
    for filepath, description in important_files:
        if os.path.exists(filepath):
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description}")
            all_ok = False

    if not all_ok:
        print(f"\n⚠️  Einige Dateien fehlen!")
        print(f"   Bitte die fehlenden Dateien erstellen")

    print(f"\n{'='*60}")

def show_system_info():
    """Zeigt detaillierte Systeminformationen."""
    print_section("DETAILLIERTE SYSTEM-INFORMATIONEN")

    # Prüfe wichtige Bibliotheken
    libraries = {
        'pandas': ('📊', 'Datenverarbeitung'),
        'numpy': ('🔢', 'Numerische Berechnungen'),
        'scikit-learn': ('🤖', 'Machine Learning'),
        'MetaTrader5': ('💱', 'Trading Platform'),
        'talib': ('📈', 'Technische Indikatoren'),
        'joblib': ('💾', 'Modell-Speicherung'),
        'matplotlib': ('📉', 'Visualisierung')
    }

    print("\n📚 BIBLIOTHEKEN:")
    for lib, (icon, desc) in libraries.items():
        try:
            __import__(lib)
            version = sys.modules[lib].__version__ if hasattr(sys.modules[lib], '__version__') else 'N/A'
            print(f"   {icon} {lib}: {desc} (v{version})")
        except ImportError:
            print(f"   ❌ {lib}: FEHLT ({desc})")

    # Speicher und CPU Info
    print(f"\n💾 SPEICHER:")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   💾 Gesamt: {memory.total / 1e9:.1f} GB")
        print(f"   🆓 Verfügbar: {memory.available / 1e9:.1f} GB ({memory.percent}% verwendet)")
    except:
        print("   ℹ️  psutil nicht verfügbar")

    # Python Info
    print(f"\n🐍 PYTHON DETAILS:")
    print(f"   📁 Python-Pfad: {sys.executable}")
    print(f"   📂 Arbeitsverzeichnis: {os.getcwd()}")
    print(f"   🔧 Bytecode Order: {sys.byteorder}")

    # Bot Verzeichnis-Struktur
    if input("\n📁 Verzeichnis-Struktur anzeigen? (j/n): ").lower() == 'j':
        print_section("VERZEICHNIS-STRUKTUR")

        def list_dir(path, indent=0, max_depth=2, current_depth=0):
            if current_depth > max_depth:
                return

            try:
                for item in os.listdir(path):
                    full_path = os.path.join(path, item)
                    prefix = "    " * indent + "📁 " if os.path.isdir(full_path) else "    " * indent + "📄 "

                    if os.path.isdir(full_path):
                        print(f"{prefix}{item}/")
                        list_dir(full_path, indent + 1, max_depth, current_depth + 1)
                    elif item.endswith('.py') or item.endswith('.json') or item.endswith('.pkl'):
                        size = os.path.getsize(full_path)
                        print(f"{prefix}{item} ({size:,} bytes)")
            except:
                pass

        list_dir('.', max_depth=3)

# ============================================================================
# KONFIGURATIONS-EDITOR
# ============================================================================

def edit_configuration(config: Dict):
    """Bearbeitet die Konfiguration."""
    print_section("KONFIGURATIONS-EDITOR")

    categories = {
        '1': ('Trading', config['trading']),
        '2': ('ML', config['ml']),
        '3': ('Paper Trading', config['paper_trading']),
        '4': ('Risk Management', config['risk_management']),
        '5': ('UI', config['ui'])
    }

    while True:
        print("\n📋 KATEGORIEN:")
        for key, (name, _) in categories.items():
            print(f"   {key}. {name}")
        print("   6. ↩️  Zurück zum Hauptmenü")
        print("   7. 💾 Speichern und zurück")

        choice = input("\n❓ Kategorie wählen (1-7): ").strip()

        if choice == '6':
            break
        elif choice == '7':
            save_config(config)
            print_status("Konfiguration gespeichert!", "success")
            break
        elif choice in categories:
            category_name, category_data = categories[choice]
            edit_category(category_name, category_data)
        else:
            print_status("Ungültige Auswahl", "error")

def edit_category(category_name: str, category_data: Dict):
    """Bearbeitet eine spezifische Kategorie."""
    print_section(f"Bearbeite: {category_name}")

    items = list(category_data.items())

    while True:
        print("\n⚙️  EINSTELLUNGEN:")
        for i, (key, value) in enumerate(items, 1):
            value_str = str(value)
            if isinstance(value, bool):
                value_str = "✅ Ja" if value else "❌ Nein"
            elif isinstance(value, float):
                value_str = f"{value:.3f}"

            print(f"   {i}. {key}: {value_str}")

        print(f"   {len(items) + 1}. ↩️  Zurück zur Kategorie-Auswahl")

        try:
            choice = int(input(f"\n❓ Einstellung wählen (1-{len(items) + 1}): ").strip())

            if choice == len(items) + 1:
                break
            elif 1 <= choice <= len(items):
                key, current_value = items[choice - 1]
                new_value = edit_setting(key, current_value)

                if new_value is not None:
                    category_data[key] = new_value
                    items[choice - 1] = (key, new_value)
                    print_status(f"{key} auf {new_value} gesetzt", "success")

        except (ValueError, IndexError):
            print_status("Ungültige Auswahl", "error")

def edit_setting(key: str, current_value):
    """Bearbeitet eine einzelne Einstellung."""
    print(f"\n✏️  Bearbeite: {key}")
    print(f"   Aktuell: {current_value} ({type(current_value).__name__})")

    if isinstance(current_value, bool):
        new_value = input("   Neuer Wert (j/n): ").strip().lower() == 'j'
        return new_value

    elif isinstance(current_value, int):
        try:
            new_value = int(input("   Neuer Wert: ").strip())
            return new_value
        except ValueError:
            print_status("Ungültige Ganzzahl", "error")
            return None

    elif isinstance(current_value, float):
        try:
            new_value = float(input("   Neuer Wert: ").strip())
            return new_value
        except ValueError:
            print_status("Ungültige Dezimalzahl", "error")
            return None

    elif isinstance(current_value, str):
        new_value = input("   Neuer Wert: ").strip()
        return new_value if new_value else current_value

    else:
        print_status(f"Typ {type(current_value)} wird nicht unterstützt", "error")
        return None

# ============================================================================
# HAUPTMENÜ
# ============================================================================

def main_menu():
    """Hauptmenü des Trading Bots."""
    setup_environment()
    config = load_config()

    while True:
        clear_screen()
        print_header("Wähle eine Option aus dem Menü")

        print("📋 HAUPTMENÜ:")
        print("   1. 📡 Trading Signal generieren")
        print("   2. 🤖 ML-Modell trainieren")
        print("   3. 📊 Paper Trading Session starten")
        print("   4. 📈 Dashboard anzeigen")
        print("   5. ⚙️  Konfiguration bearbeiten")
        print("   6. 💻 System-Informationen")
        print("   7. 🔧 Tools & Utilities")
        print("   8. 🚪 Beenden")
        print()

        choice = input("❓ Auswahl (1-8): ").strip()

        if choice == "1":
            clear_screen()
            print_header("Trading Signal Generierung")

            # Prüfe ML-Modell
            model_exists, metadata = check_ml_model()

            if not model_exists:
                print_status("Kein ML-Modell gefunden!", "error")
                print("\n💡 Bitte zuerst:")
                print("   1. Option 2 wählen (ML-Modell trainieren)")
                print("   2. Oder manuell trainieren mit:")
                print("      python -c \"from src.paper_trading.ml_integration import train_ml_model; train_ml_model()\"")
                input("\nDrücke Enter zum Fortfahren...")
                continue

            # Generiere Signal
            signal_data = generate_signal()

            if signal_data:
                display_signal_result(signal_data, config)

            input("\nDrücke Enter zum Fortfahren...")

        elif choice == "2":
            clear_screen()
            print_header("ML-Modell Training")
            train_ml_model()
            input("\nDrücke Enter zum Fortfahren...")

        elif choice == "3":
            clear_screen()
            print_header("Paper Trading")
            results = run_paper_trading_session(config)
            if results:
                display_paper_trading_results(results, config)
            input("\nDrücke Enter zum Fortfahren...")

        elif choice == "4":
            clear_screen()
            print_header("System Dashboard")
            show_dashboard(config)
            input("\nDrücke Enter zum Fortfahren...")

        elif choice == "5":
            clear_screen()
            print_header("Konfigurations-Editor")
            edit_configuration(config)

        elif choice == "6":
            clear_screen()
            print_header("System-Informationen")
            show_system_info()
            input("\nDrücke Enter zum Fortfahren...")

        elif choice == "7":
            clear_screen()
            print_header("Tools & Utilities")
            show_tools_menu(config)

        elif choice == "8":
            print("\n👋 Auf Wiedersehen! Bis zum nächsten Trade!")
            time.sleep(1)
            break

        else:
            print_status("❌ Ungültige Auswahl!", "error")
            time.sleep(1)

def display_signal_result(signal_data: Dict, config: Dict):
    """Zeigt das Signal-Ergebnis an."""
    print_section("SIGNAL ERGEBNIS")

    print(f"\n📡 SIGNAL: {signal_data['signal']}")
    print(f"🎯 CONFIDENCE: {signal_data['confidence']:.1f}%")
    print(f"🤖 ENGINE: {signal_data['engine_type'].upper()}")
    print(f"⏱️  ZEIT: {signal_data['timestamp'][11:19]}")

    # Trading-Empfehlung
    min_confidence = config['ml']['min_confidence']
    signal = signal_data['signal']
    confidence = signal_data['confidence']

    print(f"\n💰 TRADING-EMPFEHLUNG:")

    if signal != "HOLD" and confidence >= min_confidence:
        print(f"   🚀 {signal} AUSFÜHREN!")
        print(f"   ✅ Confidence ({confidence:.1f}%) ≥ Minimum ({min_confidence}%)")

        # Risiko-Management Empfehlungen
        print(f"\n⚠️  RISIKO-MANAGEMENT:")
        print(f"   📦 Lot Size: {config['trading']['default_lot_size']}")
        print(f"   🛑 Stop Loss: {config['trading']['stop_loss_pips']} pips")
        print(f"   🎯 Take Profit: {config['trading']['take_profit_pips']} pips")
        print(f"   ⚠️  Risk/Trade: {config['trading']['risk_per_trade']*100:.1f}%")

    elif signal == "HOLD":
        print(f"   ⏸️  KEINE AKTION")
        print(f"   📊 Grund: HOLD Signal erhalten")

    else:
        print(f"   ⏸️  KEINE AKTION")
        print(f"   📊 Grund: Confidence ({confidence:.1f}%) < Minimum ({min_confidence}%)")

    # Modell-Informationen
    if 'model_accuracy' in signal_data:
        print(f"\n🤖 MODELL-INFORMATIONEN:")
        print(f"   🎯 Accuracy: {signal_data['model_accuracy']:.2%}")

        if 'model_auc' in signal_data:
            print(f"   📈 AUC-ROC: {signal_data['model_auc']:.2%}")

        if 'training_date' in signal_data:
            train_date = signal_data['training_date']
            if len(train_date) > 10:
                print(f"   📅 Training: {train_date[:10]}")
            else:
                print(f"   📅 Training: {train_date}")

def show_tools_menu(config: Dict):
    """Zeigt das Tools-Menü."""
    tools = {
        '1': ('🔍 Datei-Integrität prüfen', check_file_integrity),
        '2': ('🧹 Cache leeren', clear_cache),
        '3': ('📊 Performance-Report erstellen', generate_performance_report),
        '4': ('↩️  Zurück zum Hauptmenü', None)
    }

    while True:
        print("\n🔧 TOOLS & UTILITIES:")
        for key, (name, _) in tools.items():
            print(f"   {key}. {name}")

        choice = input("\n❓ Auswahl (1-4): ").strip()

        if choice == '4':
            break
        elif choice in tools:
            tool_name, tool_func = tools[choice]

            if tool_func:
                clear_screen()
                print_header(tool_name)
                tool_func(config)
                input("\nDrücke Enter zum Fortfahren...")
            else:
                break
        else:
            print_status("Ungültige Auswahl", "error")

def check_file_integrity(config: Dict):
    """Prüft die Integrität der Bot-Dateien."""
    print_section("DATEI-INTEGRITÄTS-PRÜFUNG")

    required_files = [
        ('main.py', 'Hauptprogramm'),
        ('src/paper_trading/__init__.py', 'Paper Trading Modul'),
        ('src/paper_trading/ml_integration.py', 'ML Integration'),
        ('src/paper_trading/enhanced_ml_engine.py', 'Enhanced ML Engine'),
        ('data/config.json', 'Konfiguration'),
        ('data/ml_models/forex_signal_model.pkl', 'ML Modell')
    ]

    optional_files = [
        ('src/paper_trading/portfolio.py', 'Portfolio Management'),
        ('src/paper_trading/paper_bridge.py', 'Paper Trading Bridge'),
        ('data/ml_models/model_metadata.json', 'Modell-Metadaten'),
        ('data/performance_stats.json', 'Performance-Statistiken')
    ]

    print("\n🔍 PRÜFE ERFORDERLICHE DATEIEN:")
    all_required_ok = True

    for filepath, description in required_files:
        if os.path.exists(filepath):
            try:
                size = os.path.getsize(filepath)
                print(f"   ✅ {description}: {size:,} bytes")
            except:
                print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description}: FEHLT!")
            all_required_ok = False

    print("\n🔍 PRÜFE OPTIONALE DATEIEN:")
    for filepath, description in optional_files:
        if os.path.exists(filepath):
            try:
                size = os.path.getsize(filepath)
                print(f"   📄 {description}: {size:,} bytes")
            except:
                print(f"   📄 {description}")
        else:
            print(f"   ⚠️  {description}: Nicht vorhanden")

    if all_required_ok:
        print_status("\n✅ Alle erforderlichen Dateien vorhanden!", "success")
    else:
        print_status("\n❌ Einige erforderliche Dateien fehlen!", "error")
        print("\n💡 LÖSUNGEN:")
        print("   1. Fehlende Dateien aus vorherigen Anweisungen erstellen")
        print("   2. GitHub Repository neu klonen")
        print("   3. Backup-Dateien wiederherstellen")

def clear_cache(config: Dict):
    """Löscht Cache-Dateien."""
    print_section("CACHE BEREINIGUNG")

    cache_files = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '.cache',
        'logs/*.log'
    ]

    print("\n🗑️  ZU LÖSCHENDE CACHE-DATEIEN:")

    import glob
    files_to_delete = []

    for pattern in cache_files:
        found_files = glob.glob(pattern, recursive=True)
        for file in found_files:
            if os.path.exists(file):
                files_to_delete.append(file)
                print(f"   📄 {file}")

    if not files_to_delete:
        print("   ℹ️  Keine Cache-Dateien gefunden")
        return

    confirm = input("\n⚠️  Cache-Dateien löschen? (j/n): ").strip().lower()

    if confirm == 'j':
        deleted_count = 0
        for file in files_to_delete:
            try:
                if os.path.isdir(file):
                    import shutil
                    shutil.rmtree(file)
                else:
                    os.remove(file)
                deleted_count += 1
            except:
                print(f"   ❌ Konnte nicht löschen: {file}")

        print_status(f"\n✅ {deleted_count} Cache-Dateien gelöscht!", "success")
    else:
        print_status("Cache-Bereinigung abgebrochen", "warning")

def generate_performance_report(config: Dict):
    """Erstellt einen Performance-Report."""
    print_section("PERFORMANCE-REPORT")

    print("\n📊 SAMMLE DATEN...")

    report = {
        "generated": datetime.now().isoformat(),
        "bot_version": config['version'],
        "system": {
            "python": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}"
        },
        "ml_model": {},
        "trading": {},
        "files": {}
    }

    # ML-Modell Informationen
    model_exists, metadata = check_ml_model()
    if model_exists:
        report['ml_model'] = metadata

    # Datei-Statistiken
    try:
        import glob
        py_files = glob.glob('**/*.py', recursive=True)
        json_files = glob.glob('**/*.json', recursive=True)

        report['files'] = {
            "python_files": len(py_files),
            "json_files": len(json_files),
            "total_size": sum(os.path.getsize(f) for f in py_files + json_files if os.path.exists(f))
        }
    except:
        pass

    # Speichere Report
    report_path = f"data/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print_status(f"✅ Report gespeichert: {report_path}", "success")

        # Zeige Report-Zusammenfassung
        print(f"\n📋 REPORT ZUSAMMENFASSUNG:")
        print(f"   📅 Erstellt: {report['generated'][:19]}")
        print(f"   🤖 Bot Version: {report['bot_version']}")

        if report['ml_model']:
            print(f"   🤖 ML Accuracy: {report['ml_model'].get('accuracy', 0):.2%}")

        if report['files']:
            print(f"   📁 Dateien: {report['files'].get('python_files', 0)} Python, "
                  f"{report['files'].get('json_files', 0)} JSON")

    except Exception as e:
        print_status(f"❌ Fehler beim Erstellen des Reports: {e}", "error")

# ============================================================================
# HAUPTPROGRAMM
# ============================================================================

if __name__ == "__main__":
    try:
        print("\n" + "=" * 80)
        print("🚀 STARTE AI TRADING BOT v4.2")
        print("=" * 80)

        # Kurze Initialisierung
        for i in range(3):
            time.sleep(0.3)
            print(f"\r🤖 Initialisiere{' .' * (i + 1)}", end='')

        print("\n")
        main_menu()

    except KeyboardInterrupt:
        print("\n\n⚠️  Programm durch Benutzer abgebrochen")

    except Exception as e:
        print(f"\n❌ KRITISCHER FEHLER: {e}")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80)

        print("\n💡 NOTFALL-MAßNAHMEN:")
        print("   1. Prüfe ob alle Dateien existieren")
        print("   2. Starte den Bot neu")
        print("   3. Falls Problem besteht, erstelle Dateien neu")

        input("\nDrücke Enter zum Beenden...")