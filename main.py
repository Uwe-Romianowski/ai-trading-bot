# main.py - Erweiterte Version mit MT5 Integration
#!/usr/bin/env python3
"""
AI TRADING BOT v4.0 - MIT MT5 INTEGRATION
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Import ML-Modul
try:
    from src.ml_integration.ml_signal_generator import MLSignalGenerator
    print("✅ ML-Module importiert")
except ImportError as e:
    print(f"❌ Import Fehler: {e}")
    print("💡 Stelle sicher dass 'src/ml_integration/ml_signal_generator.py' existiert")
    sys.exit(1)

# Konfiguration
def load_config():
    try:
        with open('config/bot_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except:
        print("⚠️  Keine Konfig gefunden - verwende Defaults")
        return {'ml': {'lookback': 100, 'confidence_threshold': 0.65}}

# Testdaten Generator
def get_test_data(count=150):
    """Generiert Testdaten falls keine MT5 Verbindung möglich"""
    dates = pd.date_range(end=datetime.now(), periods=count, freq='5min')
    price = 1.0800
    data = []

    for i in range(count):
        returns = np.random.normal(0.0001, 0.0005)
        open_price = price
        close_price = price * (1 + returns)

        data.append({
            'time': dates[i],
            'open': open_price,
            'high': max(open_price, close_price) + abs(np.random.normal(0, 0.0002)),
            'low': min(open_price, close_price) - abs(np.random.normal(0, 0.0002)),
            'close': close_price,
            'volume': np.random.randint(100, 1000)
        })
        price = close_price

    return pd.DataFrame(data)

# MT5 Integration Funktionen
def test_mt5_integration():
    """Testet MT5 Integration"""
    print("\n" + "="*60)
    print("📡 MT5 LIVE-INTEGRATION TEST")
    print("="*60)
    
    try:
        # Prüfe ob MT5 Client existiert
        mt5_client_path = 'src/mt5_client/mt5_live_client.py'
        if not os.path.exists(mt5_client_path):
            print(f"❌ MT5 Client nicht gefunden: {mt5_client_path}")
            print("💡 Erstelle das Verzeichnis und die Datei:")
            print("   mkdir src\\mt5_client")
            print("   Erstelle: src/mt5_client/mt5_live_client.py")
            return False
        
        from src.mt5_client.mt5_live_client import MT5LiveClient
        
        client = MT5LiveClient()
        success = client.test_connection()
        
        if success:
            print("\n✅ MT5 Integration erfolgreich!")
            print("\n📌 Nächste Schritte:")
            print("1. MT5 Daten in ML integrieren")
            print("2. Live-Feature-Berechnung testen")
            print("3. Paper-Trading mit echten Daten starten")
        else:
            print("\n❌ MT5 Integration fehlgeschlagen")
            print("\n💡 TROUBLESHOOTING:")
            print("   • MT5 Terminal muss LAUFEN und sichtbar sein")
            print("   • EURUSD muss in Market Watch sein (Rechtsklick -> Show)")
            print("   • Demo Account muss aktiv sein")
            print("   • Firewall-Einstellungen prüfen (Port 443)")
            print("   • Python mit Admin-Rechten starten")
            print("   • MT5 Version prüfen: pip show MetaTrader5")
        
        return success
    
    except ImportError as e:
        print(f"❌ MT5 Client Import Fehler: {e}")
        print("\n💡 So erstellst du den MT5 Client:")
        print("   1. Erstelle Verzeichnis: mkdir src\\mt5_client")
        print("   2. Erstelle Datei: src/mt5_client/mt5_live_client.py")
        print("   3. Kopiere den Code von oben in die Datei")
        return False
    except Exception as e:
        print(f"❌ Unerwarteter Fehler: {e}")
        import traceback
        print("\n🔍 Debug Info:")
        traceback.print_exc()
        return False

def integrate_mt5_with_ml():
    """Integriert MT5 Daten mit ML"""
    print("\n" + "="*60)
    print("🤖 MT5 + ML LIVE INTEGRATION")
    print("="*60)
    
    # Prüfe ob MT5 Client verfügbar
    mt5_client_path = 'src/mt5_client/mt5_live_client.py'
    if not os.path.exists(mt5_client_path):
        print(f"❌ MT5 Client nicht gefunden: {mt5_client_path}")
        print("💡 Bitte erstelle zuerst: src/mt5_client/mt5_live_client.py")
        return
    
    try:
        # ML initialisieren
        config = load_config()
        ml = MLSignalGenerator(config)
        print("✅ ML System initialisiert")
        
        # MT5 Client
        from src.mt5_client.mt5_live_client import MT5LiveClient
        client = MT5LiveClient()
        
        if not client.connect():
            print("❌ MT5 Verbindung fehlgeschlagen")
            return
        
        try:
            print("\n📥 Hole Live-Daten von MT5...")
            df = client.get_historical_data("EURUSD", "M5", 150)
            
            if df.empty:
                print("❌ Keine Daten erhalten - verwende Testdaten")
                df = get_test_data(150)
                print("⚠️  Verwende simulierte Daten")
            
            print(f"✅ {len(df)} Kerzen erhalten")
            print(f"   Letzte Kerze: {df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M') if hasattr(df['time'].iloc[-1], 'strftime') else df['time'].iloc[-1]}")
            print(f"   Close Preis: {df['close'].iloc[-1]:.5f}")
            
            # Daten zu ML hinzufügen
            print("\n⚡ Verarbeite Daten mit ML...")
            batch_size = 50
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                ml.add_live_data('EURUSD', batch)
                
                status = ml.get_buffer_status()
                print(f"   Fortschritt: {status['completion_percentage']:.0f}%")
                
                if status['buffer_ready']:
                    print("✅ ML mit Live-Daten bereit!")
                    break
            
            if ml.is_ready():
                # Live-Signal generieren
                print("\n🎯 Generiere Live-Signal...")
                signal = ml.generate_signal()
                
                print(f"\n{'='*40}")
                print("📊 TRADING SIGNAL REPORT")
                print(f"{'='*40}")
                print(f"Signal:      {signal.get('signal', 'N/A'):>10}")
                print(f"Confidence:  {signal.get('confidence', 0):>10.1%}")
                print(f"Features:    {signal.get('features_used', 0):>10}/41")
                print(f"Timestamp:   {datetime.now():%H:%M:%S}")
                print(f"{'='*40}")
                
                if 'features' in signal:
                    print(f"\n📈 Wichtige Features:")
                    features = signal['features']
                    top_features = features[:3] if len(features) > 3 else features
                    for i, (feat, imp) in enumerate(top_features, 1):
                        print(f"   {i}. {feat}: {imp:.3f}")
            else:
                print("❌ ML nicht bereit - benötige mehr Daten")
                status = ml.get_buffer_status()
                print(f"   Buffer: {status['ml_buffer_rows']}/{status['lookback_required']}")
            
        finally:
            client.disconnect()
            print("✅ MT5 Verbindung getrennt")
    
    except ImportError as e:
        print(f"❌ MT5 Client nicht gefunden: {e}")
        print("💡 Bitte erstelle: src/mt5_client/mt5_live_client.py")
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()

def check_system_dependencies():
    """Prüft System-Abhängigkeiten"""
    print("\n🔧 SYSTEM CHECK")
    print("-"*40)
    
    dependencies = {
        'MetaTrader5': False,
        'ML Module': False,
        'Config': False,
        'Data Models': False
    }
    
    # Prüfe MetaTrader5
    try:
        import MetaTrader5
        dependencies['MetaTrader5'] = True
        print(f"✅ MetaTrader5: v{MetaTrader5.__version__}")
    except ImportError:
        print("❌ MetaTrader5: Nicht installiert")
        print("   Installiere: pip install MetaTrader5")
    
    # Prüfe ML Module
    if os.path.exists('src/ml_integration/ml_signal_generator.py'):
        dependencies['ML Module'] = True
        print("✅ ML Module: Gefunden")
    else:
        print("❌ ML Module: Nicht gefunden")
    
    # Prüfe Config
    if os.path.exists('config/bot_config.yaml'):
        dependencies['Config'] = True
        print("✅ Config: Gefunden")
    else:
        print("⚠️  Config: Nicht gefunden (verwende Defaults)")
    
    # Prüfe Data Models
    if os.path.exists('data/models/random_forest_model.joblib'):
        dependencies['Data Models'] = True
        print("✅ ML Model: Gefunden")
    else:
        print("❌ ML Model: Nicht gefunden")
    
    return all(dependencies.values())

# Hauptmenü
def main():
    print("\n" + "="*60)
    print("🤖 AI TRADING BOT v4.0 - PHASE C: MT5 INTEGRATION")
    print("="*60)
    print("Account: REMOVED_MT5_LOGIN | Symbol: EURUSD | Timeframe: M5")
    print("="*60)
    
    # System Check
    print("\n🔍 Führe System-Check durch...")
    system_ok = check_system_dependencies()
    
    if not system_ok:
        print("\n⚠️  WARNUNG: Nicht alle Abhängigkeiten erfüllt")
        print("   Einige Funktionen sind möglicherweise eingeschränkt")
    
    ml = None
    
    while True:
        print("\n📋 HAUPTMENÜ:")
        print("1. ML-System starten")
        print("2. Testdaten verarbeiten (simuliert)")
        print("3. Signal generieren")
        print("4. Status anzeigen")
        print("5. 🔄 MT5 Integration testen")
        print("6. 📡 MT5 + ML Integration (LIVE)")
        print("7. 🛠️  System Check")
        print("8. 🚪 Beenden")
        print("\nWahl (1-8): ", end="")

        try:
            choice = input().strip()

            if choice == "1":
                config = load_config()
                ml = MLSignalGenerator(config)
                print(f"✅ ML gestartet (Lookback: {ml.lookback})")

            elif choice == "2":
                if not ml:
                    print("❌ ML nicht gestartet. Wähle zuerst Option 1.")
                    continue

                print("\n📊 Erstelle Testdaten...")
                df = get_test_data(200)

                # In Batches verarbeiten
                batch_size = 50
                for i in range(0, len(df), batch_size):
                    batch = df.iloc[i:i+batch_size]
                    ml.add_live_data('EURUSD', batch)

                    status = ml.get_buffer_status()
                    print(f"   Fortschritt: {status['completion_percentage']:.0f}%")

                    if status['buffer_ready']:
                        print("✅ ML Buffer vollständig")
                        break

            elif choice == "3":
                if not ml:
                    print("❌ ML nicht gestartet")
                    continue

                if not ml.is_ready():
                    print("❌ ML nicht bereit. Sammle zuerst Daten.")
                    status = ml.get_buffer_status()
                    print(f"   Buffer: {status['ml_buffer_rows']}/{status['lookback_required']}")
                    continue

                signal = ml.generate_signal()
                print(f"\n📊 SIGNAL REPORT")
                print(f"{'='*30}")
                print(f"Signal:     {signal.get('signal', 'N/A'):>10}")
                print(f"Confidence: {signal.get('confidence', 0):>10.1%}")
                print(f"{'='*30}")

                if 'error' in signal:
                    print(f"⚠️  Warnung: {signal['error']}")

            elif choice == "4":
                if not ml:
                    print("❌ ML nicht gestartet")
                    continue

                status = ml.get_buffer_status()
                info = ml.get_model_info()

                print(f"\n📊 SYSTEM STATUS")
                print(f"{'='*30}")
                print(f"ML Buffer:    {status['ml_buffer_rows']}/{status['lookback_required']}")
                print(f"Bereit:       {'✅' if status['buffer_ready'] else '❌'}")
                print(f"Fertigstellung: {status['completion_percentage']:.1f}%")
                print(f"Modell:       {info['model_type']}")
                print(f"Features:     {info['features_expected']}")
                print(f"{'='*30}")

            elif choice == "5":
                test_mt5_integration()

            elif choice == "6":
                integrate_mt5_with_ml()

            elif choice == "7":
                check_system_dependencies()

            elif choice == "8":
                print("\n👋 Beendet")
                if ml:
                    print("✅ ML System heruntergefahren")
                break

            else:
                print("❌ Ungültige Eingabe. Bitte 1-8 wählen.")

        except KeyboardInterrupt:
            print("\n⚠️  Abbruch durch Benutzer")
            break
        except Exception as e:
            print(f"\n❌ Fehler: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Prüfe ob ML-Module existieren
    if not os.path.exists('src/ml_integration'):
        print("❌ Verzeichnis 'src/ml_integration' fehlt")
        print("💡 Stelle sicher dass das ML-Modul vorhanden ist")
        sys.exit(1)
    
    # Starte Hauptmenü
    try:
        main()
    except Exception as e:
        print(f"\n💥 KRITISCHER FEHLER: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Prüfe ob requirements.txt installiert: pip install -r docs/requirements.txt")
        print("2. Stelle sicher dass MT5 Terminal läuft")
        print("3. Prüfe ob 'data/models/' existiert mit ML Modell")
        input("\nDrücke Enter um zu beenden...")