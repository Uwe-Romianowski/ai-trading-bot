#!/usr/bin/env python3
"""
TEST DER .env KONFIGURATION
===========================
Prüft, ob alle Umgebungsvariablen korrekt geladen werden.
"""

import os
from dotenv import load_dotenv
from datetime import datetime

def test_environment():
    """Testet das Laden der .env-Datei"""
    print("🔧 TEST DER .env KONFIGURATION")
    print("=" * 50)
    
    # .env Datei laden
    load_dotenv()
    
    # Kritische Variablen prüfen
    critical_vars = {
        'MT5_LOGIN': 'MT5 Kontonummer',
        'MT5_SERVER': 'MT5 Server',
        'TRADING_BASE_CURRENCY': 'Basiswährung',
        'MAX_RISK_PER_TRADE': 'Maximales Risiko pro Trade',
        'PAPER_TRADING_INITIAL_BALANCE': 'Startkapital Paper Trading'
    }
    
    print("📋 Kritische Konfigurationsvariablen:")
    all_ok = True
    
    for var, description in critical_vars.items():
        value = os.getenv(var)
        status = "✅" if value else "❌"
        
        if value:
            print(f"  {status} {var:30} = {value:20} # {description}")
        else:
            print(f"  {status} {var:30} = {'NICHT GESETZT':20} # {description}")
            all_ok = False
    
    print("\n📊 Trading Konfiguration:")
    trading_vars = [
        'ML_BUY_THRESHOLD', 'ML_SELL_THRESHOLD', 'ML_MIN_CONFIDENCE',
        'PAPER_TRADING_COMMISSION', 'PAPER_TRADING_SLIPPAGE'
    ]
    
    for var in trading_vars:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var:30} = {value}")
        else:
            print(f"  ⚠️  {var:30} = {'Standardwert wird verwendet'}")
    
    print("\n🔒 Sicherheitscheck:")
    # Prüfen, ob Passwort gesetzt ist (nicht der Platzhalter)
    password = os.getenv('MT5_PASSWORD')
    if password and password != "IHR_MT5_DEMO_PASSWORT_HIER":
        print("  ✅ MT5_PASSWORD ist gesetzt (Sicher)")
    else:
        print("  ❌ MT5_PASSWORD ist NICHT korrekt gesetzt!")
        print("     Bitte bearbeiten Sie die .env Datei mit Ihrem echten Passwort!")
        all_ok = False
    
    print("\n" + "=" * 50)
    
    if all_ok:
        print("🎉 ALLE TESTS BESTANDEN! Ihr System ist konfiguriert.")
        print(f"   Testzeit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
    else:
        print("⚠️  EINIGE PROBLEME GEFUNDEN. Bitte überprüfen Sie Ihre .env Datei.")
        return False

if __name__ == "__main__":
    success = test_environment()
    
    # Zeige Hinweis für nächste Schritte
    if success:
        print("\n📋 NÄCHSTE SCHRITTE:")
        print("1. PaperOrder Klasse testen: python src/paper_trading/order.py")
        print("2. Portfolio erstellen:      python create_portfolio.py")
        print("3. Hauptmenü erweitern:      Fügen Sie 'Paper Trading' zu main.py hinzu")
    else:
        print("\n🔧 ERFORDERLICHE AKTIONEN:")
        print("1. Öffnen Sie die Datei '.env' in Ihrem Projektordner")
        print("2. Ersetzen Sie 'IHR_MT5_DEMO_PASSWORT_HIER' durch Ihr echtes MT5-Passwort")
        print("3. Speichern Sie die Datei und führen Sie diesen Test erneut aus")
    
    exit(0 if success else 1)