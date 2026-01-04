# test_mt5_env.py
"""
Testet speziell die .env Integration für MT5
"""
import os
import sys
from dotenv import load_dotenv

# Pfade einrichten
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_env_config():
    """Testet ob .env korrekt geladen wird."""
    print("="*60)
    print("🔍 TEST DER .env INTEGRATION FÜR MT5")
    print("="*60)
    
    # .env laden
    env_path = os.path.join(os.getcwd(), '.env')
    print(f"📁 Suche .env Datei: {env_path}")
    
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print("✅ .env Datei gefunden und geladen")
    else:
        # Alternative suchen
        load_dotenv()
        if os.getenv('MT5_LOGIN'):
            print("✅ .env Datei geladen (anderer Pfad)")
        else:
            print("❌ Keine .env Datei gefunden!")
            return False
    
    # Prüfe kritische Variablen
    print("\n📋 GELADENE KONFIGURATION:")
    
    # Deine spezifischen Variablennamen
    variables = {
        'MT5_LOGIN': os.getenv('MT5_LOGIN'),
        'MT5_PASSWORD': '*** gesetzt ***' if os.getenv('MT5_PASSWORD') else 'NICHT GESETZT',
        'MT5_SERVER': os.getenv('MT5_SERVER'),
        'TRADING_BASE_CURRENCY': os.getenv('TRADING_BASE_CURRENCY'),
        'MAX_RISK_PER_TRADE': os.getenv('MAX_RISK_PER_TRADE')
    }
    
    all_ok = True
    for key, value in variables.items():
        if value:
            print(f"  ✅ {key:30} = {value}")
        else:
            print(f"  ❌ {key:30} = NICHT GESETZT")
            if key in ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']:
                all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("🎉 .env Konfiguration ist vollständig!")
        
        # Teste MT5 Client direkt
        print("\n🚀 Teste jetzt MT5 Live Client...")
        try:
            from src.live_trading.mt5_client import quick_test
            return quick_test()
        except ImportError as e:
            print(f"❌ MT5 Client nicht verfügbar: {e}")
            return False
    else:
        print("❌ .env Konfiguration unvollständig!")
        print("💡 Bitte folgende Variablen in .env setzen:")
        print("   MT5_LOGIN=REMOVED_MT5_LOGIN")
        print("   MT5_PASSWORD=dein_passwort")
        print("   MT5_SERVER=REMOVED_MT5_SERVER")
        return False


if __name__ == "__main__":
    success = test_env_config()
    if success:
        print("\n✅ Alles funktioniert! Du kannst nun Option 11 in main.py nutzen.")
    else:
        print("\n❌ Es gab Probleme. Bitte oben stehende Fehler beheben.")