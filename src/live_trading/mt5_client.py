# src/live_trading/mt5_client.py
"""
MT5 Live Trading Client - Phase E Woche 1
Bietet Live-Daten und Demo-Order Execution für den AI Trading Bot.
"""
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import time
import json
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv


class MT5LiveClient:
    """
    Hauptklasse für MT5 Live-Daten und Order Execution.
    Verbindet die Paper-Trading Engine mit einem echten MT5 Demo Account.
    """
    
    def __init__(self, 
                 account: Optional[int] = None,
                 password: Optional[str] = None, 
                 server: Optional[str] = None,
                 symbol: str = "EURUSD"):
        """
        Initialisiert den MT5 Live Client.
        
        Args:
            account: MT5 Demo Account Nummer (falls None, wird aus .env geladen)
            password: MT5 Demo Account Passwort
            server: MT5 Server (z.B. 'REMOVED_MT5_SERVER')
            symbol: Standard-Symbol für Trading
        """
        # Lade .env Datei
        env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()
        
        self.account = account
        self.password = password
        self.server = server
        self.symbol = symbol
        self.connected = False
        self.initialized = False
        
        # Konfiguration laden
        self._load_config()
            
    def _load_config(self) -> None:
        """Lädt MT5 Konfiguration aus .env Datei."""
        print("🔧 Lade Konfiguration aus .env...")
        
        # Account aus MT5_LOGIN (deine .env verwendet MT5_LOGIN!)
        if not self.account:
            account_str = os.getenv('MT5_LOGIN')
            if account_str:
                try:
                    self.account = int(account_str)
                    print(f"  ✅ Account aus MT5_LOGIN geladen: {self.account}")
                except ValueError:
                    print(f"❌ MT5_LOGIN muss eine Zahl sein: {account_str}")
            else:
                # Fallback
                account_str = os.getenv('MT5_ACCOUNT')
                if account_str:
                    try:
                        self.account = int(account_str)
                        print(f"  ✅ Account aus MT5_ACCOUNT geladen: {self.account}")
                    except ValueError:
                        print(f"❌ MT5_ACCOUNT muss eine Zahl sein: {account_str}")
        
        # Password
        if not self.password:
            self.password = os.getenv('MT5_PASSWORD')
            if self.password:
                print(f"  ✅ Password geladen (Länge: {len(self.password)} Zeichen)")
            else:
                print("❌ MT5_PASSWORD nicht in .env gefunden")
        
        # Server
        if not self.server:
            self.server = os.getenv('MT5_SERVER', 'REMOVED_MT5_SERVER')
            print(f"  ✅ Server: {self.server}")
        
        # Symbol
        if not self.symbol:
            self.symbol = os.getenv('TRADING_BASE_SYMBOL', 'EURUSD')
            print(f"  ✅ Symbol: {self.symbol}")
        
        # Prüfen ob alle benötigten Werte vorhanden sind
        missing = []
        if not self.account:
            missing.append("MT5_LOGIN")
        if not self.password:
            missing.append("MT5_PASSWORD")
        
        if missing:
            print(f"❌ Fehlende Konfiguration in .env: {', '.join(missing)}")
    
    def connect(self) -> bool:
        """Stellt Verbindung zum MT5 Terminal her."""
        if not all([self.account, self.password, self.server]):
            print("❌ Login-Daten unvollständig.")
            return False
        
        print(f"🔗 Verbinde mit MT5 Demo Account {self.account}...")
        print(f"   Server: {self.server}")
        print(f"   Symbol: {self.symbol}")
        
        # MT5 initialisieren
        if not mt5.initialize():
            error = mt5.last_error()
            print(f"❌ MT5 Initialisierung fehlgeschlagen. Error {error}")
            return False
        
        self.initialized = True
        
        # MIT DEN DATEN AUS .env EINLOGGEN
        authorized = mt5.login(
            login=self.account,
            password=self.password,
            server=self.server
        )
        
        if authorized:
            print(f"✅ Erfolgreich verbunden mit MT5 Demo Account: {self.account}")
            
            # Account Info anzeigen
            self._print_account_info()
            
            # Symbol aktivieren
            mt5.symbol_select(self.symbol, True)
            symbol_info = mt5.symbol_info(self.symbol)
            
            if symbol_info:
                print(f"✅ Symbol {self.symbol} aktiviert")
                print(f"   Spread: {symbol_info.spread} points")
                
                # KORREKTUR HIER: trade_allowed gibt es nicht, verwende select oder visible
                # Optional: Alle verfügbaren Attribute anzeigen
                # self._print_symbol_attributes(symbol_info)
                
                # Statt trade_allowed verwenden wir select oder visible
                print(f"   Im Market Watch: {symbol_info.select}")
                print(f"   Sichtbar: {symbol_info.visible}")
                
                # Für Handelsfähigkeit prüfen
                if hasattr(symbol_info, 'trade_mode'):
                    trade_mode = symbol_info.trade_mode
                    print(f"   Trade Mode: {trade_mode}")
                    # trade_mode == 0 bedeutet oft "disabled"
            else:
                print(f"⚠️  Symbol {self.symbol} konnte nicht aktiviert werden")
            
            self.connected = True
            return True
        else:
            error = mt5.last_error()
            print(f"❌ Login fehlgeschlagen. Error: {error}")
            mt5.shutdown()
            self.initialized = False
            return False
    
    def _print_account_info(self) -> None:
        """Zeigt wichtige Account Informationen an."""
        info = mt5.account_info()
        if info:
            print(f"   👤 Name: {info.name}")
            print(f"   💰 Balance: ${info.balance:.2f}")
            print(f"   📈 Equity: ${info.equity:.2f}")
            print(f"   🏦 Margin: ${info.margin:.2f}")
            print(f"   📊 Free Margin: ${info.margin_free:.2f}")
            if hasattr(info, 'margin_level') and info.margin_level is not None:
                print(f"   📉 Margin Level: {info.margin_level:.2f}%")
            print(f"   🎯 Leverage: 1:{info.leverage}")
            print(f"   📅 Währung: {info.currency}")
    
    def _print_symbol_attributes(self, symbol_info) -> None:
        """Hilfsfunktion: Zeigt alle Attribute des SymbolInfo-Objekts an."""
        print(f"\n📋 Verfügbare Attribute für Symbol {self.symbol}:")
        if symbol_info:
            symbol_dict = symbol_info._asdict()
            for key, value in symbol_dict.items():
                print(f"   {key}: {value}")
    
    def get_live_price(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Holt den aktuellen Live-Bid/Ask-Preis."""
        symbol = symbol or self.symbol
        
        if not self.connected:
            print("⚠️ Nicht mit MT5 verbunden. Bitte zuerst connect() aufrufen.")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ Konnte Tick für {symbol} nicht abrufen.")
            return None
        
        symbol_info = mt5.symbol_info(symbol)
        spread = symbol_info.spread if symbol_info else 0
        
        return {
            'symbol': symbol,
            'time': datetime.fromtimestamp(tick.time, tz=timezone.utc),
            'bid': tick.bid,
            'ask': tick.ask,
            'last': tick.last,
            'volume': tick.volume,
            'spread': spread,
            'spread_pips': spread * 10000
        }
    
    def get_account_info(self) -> Optional[Dict]:
        """Holt detaillierte Informationen zum Demo-Konto."""
        if not self.connected:
            return None
        
        info = mt5.account_info()
        if info:
            account_dict = info._asdict()
            
            # Berechnete Metriken hinzufügen
            if info.margin > 0:
                account_dict['margin_level'] = (info.equity / info.margin * 100)
            else:
                account_dict['margin_level'] = 0
                
            if info.equity > 0:
                account_dict['free_margin_percent'] = (info.margin_free / info.equity * 100)
            else:
                account_dict['free_margin_percent'] = 0
            
            return account_dict
        
        return None
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Holt alle offenen Positionen."""
        if not self.connected:
            return []
        
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        
        if positions is None:
            return []
        
        result = []
        for position in positions:
            pos_dict = position._asdict()
            
            # Berechne aktuellen P&L
            current_price = self.get_live_price(position.symbol)
            if current_price:
                if position.type == mt5.POSITION_TYPE_BUY:
                    profit = (current_price['bid'] - position.price_open) * position.volume * 100000
                else:
                    profit = (position.price_open - current_price['ask']) * position.volume * 100000
                
                pos_dict['current_profit'] = profit
                pos_dict['current_profit_pips'] = profit / (position.volume * 10)
            
            result.append(pos_dict)
        
        return result
    
    def test_connection(self) -> Dict:
        """Führt einen umfassenden Verbindungstest durch."""
        print("\n" + "="*60)
        print("🔧 MT5 LIVE VERBINDUNGSTEST (mit .env Konfiguration)")
        print("="*60)
        
        results = {
            'connected': False,
            'config_loaded': False,
            'account_info': None,
            'live_price': None,
            'symbol_info': None,
            'positions': []
        }
        
        # Prüfe Konfiguration
        if all([self.account, self.password, self.server]):
            results['config_loaded'] = True
            print(f"✅ .env Konfiguration geladen: Account={self.account}, Server={self.server}")
        else:
            print("❌ .env Konfiguration unvollständig")
            return results
        
        # Verbindung testen
        if self.connect():
            results['connected'] = True
            print("✅ Verbindungstest: BESTANDEN")
        else:
            print("❌ Verbindungstest: FEHLGESCHLAGEN")
            return results
        
        # Account Info
        account_info = self.get_account_info()
        if account_info:
            results['account_info'] = account_info
            print(f"✅ Account Info: {account_info.get('name', 'N/A')} | Balance: ${account_info.get('balance', 0):.2f}")
        else:
            print("❌ Account Info: FEHLER")
        
        # Live Preis
        live_price = self.get_live_price()
        if live_price:
            results['live_price'] = live_price
            print(f"✅ Live Preis {live_price['symbol']}: Bid={live_price['bid']:.5f}, Ask={live_price['ask']:.5f}")
            print(f"   Spread: {live_price['spread_pips']:.1f} pips")
        else:
            print("❌ Live Preis: FEHLER")
        
        # Symbol Info
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info:
            results['symbol_info'] = symbol_info._asdict()
            print(f"✅ Symbol Info: Lot Size={symbol_info.trade_contract_size}, Min Lot={symbol_info.volume_min}")
        else:
            print("❌ Symbol Info: FEHLER")
        
        # Offene Positionen
        positions = self.get_open_positions()
        results['positions'] = positions
        print(f"✅ Offene Positionen: {len(positions)}")
        
        print("="*60)
        print("📊 TEST ABGESCHLOSSEN")
        
        return results
    
    def shutdown(self) -> None:
        """Trennt die Verbindung zum MT5 Terminal."""
        if self.initialized:
            mt5.shutdown()
            self.connected = False
            self.initialized = False
            print("🔌 Verbindung zu MT5 getrennt.")
    
    def __enter__(self):
        """Context Manager Support."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context Manager Support."""
        self.shutdown()


def quick_test():
    """Schnelltest für die MT5 Verbindung mit .env Konfiguration."""
    print("🚀 MT5 Live Client Schnelltest (mit .env)")
    print("-" * 50)
    
    try:
        client = MT5LiveClient()
        results = client.test_connection()
        
        if results['connected']:
            print("\n🎉 Alle Tests bestanden! Live-Trading bereit.")
            return True
        else:
            print("\n❌ Verbindungstest fehlgeschlagen.")
            return False
            
    except Exception as e:
        print(f"\n💥 Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'client' in locals():
            client.shutdown()


if __name__ == "__main__":
    quick_test()