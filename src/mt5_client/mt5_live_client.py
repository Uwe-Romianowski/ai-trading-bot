# src/mt5_client/mt5_live_client.py - KORRIGIERTE VERSION

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List
import os
from dotenv import load_dotenv

class MT5LiveClient:
    """
    Einfacher MT5 Client für Live-Trading mit korrekter .env Integration.
    """
    def __init__(self, login=None, server=None, password=None):  # KEINE festen Werte mehr!
        """
        Initialisiert den MT5 Client.
        
        Args:
            login: MT5 Kontonummer (wird aus MT5_LOGIN in .env geladen wenn None)
            server: MT5 Server (wird aus MT5_SERVER in .env geladen wenn None)
            password: MT5 Passwort (wird aus MT5_PASSWORD in .env geladen wenn None)
        """
        # .env Datei laden
        load_dotenv()
        
        # Werte aus .env laden falls nicht explizit angegeben
        self.login = login or os.getenv('MT5_LOGIN')
        self.server = server or os.getenv('MT5_SERVER', 'REMOVED_MT5_SERVER')
        self.password = password or os.getenv('MT5_PASSWORD')
        
        self.connected = False
        self.account_info = None
        
    def connect(self) -> bool:
        """
        Stellt Verbindung zum MT5 Server her.
        
        Returns:
            bool: True wenn Verbindung erfolgreich
        """
        print(f"\n📡 Verbinde mit MT5 Demo Account...")
        
        if not mt5.initialize():
            print("❌ MT5 konnte nicht initialisiert werden")
            return False
        
        try:
            # Login versuchen
            if self.password:
                authorized = mt5.login(int(self.login) if self.login else None, 
                                      password=self.password, 
                                      server=self.server)
            else:
                authorized = mt5.login(int(self.login) if self.login else None, 
                                      server=self.server)
            
            if authorized:
                self.connected = True
                
                # Account Info holen
                account_info = mt5.account_info()
                if account_info:
                    self.account_info = account_info._asdict()
                    print(f"✅ Erfolgreich verbunden mit MT5 Account")
                    print(f"   • Account: {account_info.login}")
                    print(f"   • Balance: ${account_info.balance:.2f}")
                    print(f"   • Server: {self.server}")
                else:
                    print("⚠️  Verbunden, aber keine Account Info verfügbar")
                
                return True
            else:
                print(f"❌ Login fehlgeschlagen")
                error = mt5.last_error()
                print(f"   Error Code: {error[0]}, Description: {error[1]}")
                return False
                
        except Exception as e:
            print(f"❌ Verbindungsfehler: {e}")
            return False
    
    def disconnect(self) -> None:
        """Trennt die Verbindung zum MT5 Server."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("✅ MT5 Verbindung getrennt")
    
    def get_live_price(self, symbol: str = "EURUSD") -> Optional[Dict]:
        """
        Holt den aktuellen Preis für ein Symbol.
        
        Args:
            symbol: Trading Symbol (z.B. "EURUSD")
            
        Returns:
            Dict mit Bid/Ask oder None bei Fehler
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'symbol': symbol,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'time': pd.to_datetime(tick.time, unit='s'),
                    'spread': (tick.ask - tick.bid) * 10000  # In Pips
                }
        except Exception as e:
            print(f"❌ Fehler beim Preisabruf: {e}")
        
        return None
    
    def get_historical_data(self, symbol: str, timeframe: str, count: int = 100) -> Optional[pd.DataFrame]:
        """
        Holt historische Daten von MT5.
        
        Args:
            symbol: Trading Symbol
            timeframe: Zeitrahmen (M1, M5, H1, etc.)
            count: Anzahl der Kerzen
            
        Returns:
            DataFrame mit OHLC Daten oder None
        """
        if not self.connected:
            if not self.connect():
                return None
        
        # Zeitrahmen Mapping
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
        }
        
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        try:
            rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
                return df
        except Exception as e:
            print(f"❌ Fehler beim Datenabruf: {e}")
        
        return None
    
    def get_account_summary(self) -> Dict:
        """
        Gibt eine Zusammenfassung des Accounts zurück.
        
        Returns:
            Dict mit Account Informationen
        """
        if not self.connected:
            self.connect()
        
        summary = {
            'connected': self.connected,
            'login': self.login,
            'server': self.server,
            'account_info': self.account_info
        }
        
        if self.account_info:
            # Wichtige Metriken berechnen
            balance = self.account_info.get('balance', 0)
            equity = self.account_info.get('equity', 0)
            margin = self.account_info.get('margin', 0)
            free_margin = self.account_info.get('margin_free', 0)
            
            summary.update({
                'balance': balance,
                'equity': equity,
                'margin': margin,
                'free_margin': free_margin,
                'margin_level': (equity / margin * 100) if margin > 0 else 0,
                'pl': equity - balance,
                'free_margin_percent': (free_margin / equity * 100) if equity > 0 else 0
            })
        
        return summary
    
    def is_market_open(self, symbol: str = "EURUSD") -> bool:
        """
        Prüft ob der Markt für ein Symbol geöffnet ist.
        
        Args:
            symbol: Trading Symbol
            
        Returns:
            bool: True wenn Markt geöffnet
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            info = mt5.symbol_info(symbol)
            if info:
                return info.visible and info.trade_mode == 0  # 0 = TRADE_MODE_FULL
        except:
            pass
        
        return False


def test_mt5_client():
    """Testet die MT5 Verbindung."""
    print("\n🧪 TESTE MT5 CLIENT...")
    
    client = MT5LiveClient()
    
    if client.connect():
        print("\n✅ Verbindung erfolgreich!")
        
        # Account Zusammenfassung
        summary = client.get_account_summary()
        print(f"\n📊 ACCOUNT ZUSAMMENFASSUNG:")
        for key, value in summary.items():
            if key not in ['account_info', 'connected']:
                print(f"   {key}: {value}")
        
        # Live Preis
        price = client.get_live_price("EURUSD")
        if price:
            print(f"\n💱 LIVE PREIS EURUSD:")
            print(f"   Bid: {price['bid']:.5f}")
            print(f"   Ask: {price['ask']:.5f}")
            print(f"   Spread: {price['spread']:.1f} pips")
        
        client.disconnect()
    else:
        print("❌ Verbindung fehlgeschlagen")


if __name__ == "__main__":
    test_mt5_client()