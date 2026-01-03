# src/mt5_client/mt5_live_client.py - KORRIGIERTE VERSION
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class MT5LiveClient:
    """MT5 Live Data Client für den AI Trading Bot"""
    
    def __init__(self, login=REMOVED_MT5_LOGIN, server="REMOVED_MT5_SERVER", password=None):
        self.login = login
        self.server = server
        self.password = password
        self.connected = False
        
    def connect(self):
        """Stellt Verbindung zu MT5 her"""
        print(f"\n📡 Verbinde mit MT5 Demo Account {self.login}...")
        
        try:
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                print(f"❌ MT5 Initialize failed: {error}")
                return False
            
            # Login versuchen
            if self.password:
                authorized = mt5.login(self.login, password=self.password, server=self.server)
            else:
                authorized = mt5.login(self.login, server=self.server)
                
            if not authorized:
                print(f"⚠️  Login ohne Passwort fehlgeschlagen")
                # Trotzdem connected setzen falls initialize erfolgreich
                
            # Account Info
            try:
                account_info = mt5.account_info()
                if account_info:
                    print(f"✅ MT5 verbunden")
                    print(f"   • Account: {account_info.login}")
                    print(f"   • Balance: ${account_info.balance:.2f}")
            except:
                print("⚠️  Keine Account Info verfügbar")
            
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Verbindungsfehler: {e}")
            return False
    
    def disconnect(self):
        """Trennt MT5 Verbindung"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("✅ MT5 Verbindung getrennt")
    
    def get_historical_data(self, symbol="EURUSD", timeframe="M5", count=100):
        """Holt historische Daten - ML-KOMPATIBEL"""
        if not self.connected:
            print("❌ Nicht verbunden")
            return pd.DataFrame()
        
        # Zeitframe Mapping
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        try:
            # Daten abrufen
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            
            if rates is None or len(rates) == 0:
                print(f"❌ Keine Daten für {symbol}")
                return pd.DataFrame()
            
            # In DataFrame konvertieren
            df = pd.DataFrame(rates)
            
            # Zeitstempel konvertieren
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # ML-kompatibles Format erstellen
            ml_df = pd.DataFrame()
            ml_df['time'] = df['time']
            ml_df['open'] = df['open'].astype(float)
            ml_df['high'] = df['high'].astype(float)
            ml_df['low'] = df['low'].astype(float)
            ml_df['close'] = df['close'].astype(float)
            
            # Volume extrahieren
            if 'tick_volume' in df.columns:
                ml_df['volume'] = df['tick_volume'].astype(int)
            elif 'volume' in df.columns:
                ml_df['volume'] = df['volume'].astype(int)
            else:
                ml_df['volume'] = 1000
            
            print(f"✅ {len(ml_df)} {symbol} {timeframe} Kerzen geladen")
            print(f"   Zeitraum: {ml_df['time'].iloc[0].strftime('%Y-%m-%d %H:%M')} bis {ml_df['time'].iloc[-1].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Close: {ml_df['close'].iloc[-1]:.5f}")
            
            return ml_df
            
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            return pd.DataFrame()
    
    def test_connection(self):
        """Testet die Verbindung"""
        print("\n" + "="*60)
        print("🧪 MT5 VERBINDUNGSTEST")
        print("="*60)
        
        if not self.connect():
            return False
        
        try:
            # Teste Datenabruf
            print(f"\n📈 Teste Datenabruf EURUSD M5:")
            df = self.get_historical_data("EURUSD", "M5", 20)
            
            if not df.empty:
                print(f"   ✅ {len(df)} Kerzen empfangen")
                print(f"   📊 Letzte Kerze:")
                last = df.iloc[-1]
                print(f"      • Time: {last['time'].strftime('%Y-%m-%d %H:%M')}")
                print(f"      • Close: {last['close']:.5f}")
                print(f"      • Volume: {last['volume']}")
                
                print("\n" + "="*60)
                print("✅ MT5 TEST ERFOLGREICH")
                print("="*60)
                return True
            else:
                print(f"   ❌ Keine Daten")
                return False
                
        finally:
            self.disconnect()

# Einfache Testfunktion
def test_mt5_client():
    """Testet den MT5 Client"""
    print("🧪 Starte MT5 Test...")
    client = MT5LiveClient()
    return client.test_connection()

if __name__ == "__main__":
    test_mt5_client()