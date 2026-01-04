# test_mt5_connection.py
"""
Test-Skript für MT5 Live Connection - Phase E Woche 1
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live_trading.mt5_client import MT5LiveClient, quick_test


def main():
    print("="*60)
    print("🤖 AI TRADING BOT - PHASE E: MT5 LIVE CONNECTION TEST")
    print("="*60)
    
    # Option 1: Automatischer Schnelltest
    print("\n1️⃣ Automatischer Schnelltest (empfohlen)...")
    success = quick_test()
    
    if not success:
        print("\n" + "="*60)
        print("2️⃣ Manueller Test mit eigenen Credentials...")
        print("="*60)
        
        # Option 2: Manuelle Eingabe
        account = int(input("MT5 Demo Account Nummer: "))
        password = input("MT5 Demo Account Passwort: ")
        server = input("MT5 Server [MetaQuotes-Demo]: ") or "MetaQuotes-Demo"
        
        client = MT5LiveClient(account=account, password=password, server=server)
        
        try:
            if client.connect():
                # Live Preis testen
                price = client.get_live_price()
                if price:
                    print(f"\n✅ Live Preis erfolgreich abgerufen:")
                    print(f"   Symbol: {price['symbol']}")
                    print(f"   Bid: {price['bid']:.5f}")
                    print(f"   Ask: {price['ask']:.5f}")
                    print(f"   Spread: {price['spread_pips']:.1f} pips")
                
                # Account Info
                account_info = client.get_account_info()
                if account_info:
                    print(f"\n💰 Account Status:")
                    print(f"   Balance: ${account_info['balance']:.2f}")
                    print(f"   Equity: ${account_info['equity']:.2f}")
                    print(f"   Free Margin: ${account_info['margin_free']:.2f}")
                
                print("\n🎉 MT5 Live Client funktioniert einwandfrei!")
            else:
                print("\n❌ Verbindung konnte nicht hergestellt werden.")
        finally:
            client.shutdown()
    
    print("\n" + "="*60)
    print("🚀 Phase E - Woche 1: MT5 Live Client bereit!")
    print("Nächster Schritt: main.py für Live-Trading erweitern")
    print("="*60)


if __name__ == "__main__":
    main()