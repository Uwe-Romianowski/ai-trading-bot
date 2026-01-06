"""
ENHANCED ML TRADING ENGINE v4.2 - MIT FEATURE-BERECHNUNG (KORRIGIERT)
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any

# Import ML Bibliotheken
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
except ImportError as e:
    print(f"⚠️  ML-Bibliotheken fehlen: {e}")

# Import Trading Bibliotheken
try:
    import MetaTrader5 as mt5
except ImportError:
    print("⚠️  MetaTrader5 nicht verfügbar - Simulation Mode")

# Import TA-Lib
try:
    import talib
    TA_LIB_AVAILABLE = True
except ImportError:
    print("⚠️  TA-Lib nicht verfügbar - benutze einfache Berechnungen")
    TA_LIB_AVAILABLE = False

class EnhancedMLTradingEngine:
    """
    Erweiterte ML Trading Engine mit vollständiger Feature-Berechnung.
    """
    
    def __init__(self, portfolio=None):
        """Initialisiert die ML Engine."""
        self.portfolio = portfolio
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.metadata = {}
        self.model_loaded = False
        self.current_price = 0.0
        self.symbol = "EURUSD"
        self.timeframe = mt5.TIMEFRAME_H1 if 'mt5' in globals() else None
        self.model_path = "data/ml_models/forex_signal_model.pkl"
        self.scaler_path = "data/ml_models/forex_scaler.pkl"
        self.features_path = "data/ml_models/feature_columns.pkl"
        self.metadata_path = "data/ml_models/model_metadata.json"
        
        # Lade ML-Modell wenn vorhanden
        self.load_ml_model()
        
    def load_ml_model(self) -> bool:
        """Lädt das gespeicherte ML-Modell."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✅ ML-Modell geladen: {self.model_path}")
                
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                    print(f"✅ Scaler geladen: {self.scaler_path}")
                    
                if os.path.exists(self.features_path):
                    self.feature_columns = joblib.load(self.features_path)
                    print(f"✅ Features geladen: {len(self.feature_columns)} Spalten")
                    
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
                    print(f"✅ Metadaten geladen: Accuracy {self.metadata.get('accuracy', 0):.2%}")
                    
                self.model_loaded = True
                return True
                
        except Exception as e:
            print(f"❌ Fehler beim Laden des ML-Modells: {e}")
            traceback.print_exc()
            
        return False
        
    def get_current_features(self) -> pd.DataFrame:
        """
        Berechnet aktuelle Features für das ML-Modell.
        
        Returns:
            pd.DataFrame: Features im richtigen Format
        """
        try:
            # Hole Features
            features = self.calculate_simulated_features()
            if features is not None and not features.empty:
                return features
                
            # Fallback
            return self.create_dummy_features()
            
        except Exception as e:
            print(f"❌ Fehler in get_current_features: {e}")
            return self.create_dummy_features()
            
    def calculate_simulated_features(self) -> pd.DataFrame:
        """Berechnet Features aus simulierten Daten."""
        try:
            # Simuliere 100 Kerzen
            np.random.seed(int(time.time()))
            
            base_price = 1.10000
            prices = []
            
            # Erstelle simulierten Preisverlauf
            for i in range(100):
                change = np.random.normal(0, 0.0005)
                base_price += change
                prices.append(base_price)
                
            # Erstelle OHLCV DataFrame
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            df = pd.DataFrame({
                'open': np.array(prices) * 0.999 + np.random.normal(0, 0.0001, 100),
                'high': np.array(prices) * 1.001 + np.random.uniform(0, 0.0002, 100),
                'low': np.array(prices) * 0.999 - np.random.uniform(0, 0.0002, 100),
                'close': prices,
                'volume': np.random.randint(100, 1000, 100)
            }, index=dates)
            
            # Berechne Features (nur letzte Werte)
            features = self.calculate_all_features_last(df)
            
            return features
            
        except Exception as e:
            print(f"❌ Fehler bei simulierter Feature-Berechnung: {e}")
            return self.create_dummy_features()
            
    def calculate_all_features_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Berechnet alle technischen Features (nur letzte Werte).
        
        Args:
            df: DataFrame mit OHLCV Daten
            
        Returns:
            pd.DataFrame: Features (nur eine Zeile)
        """
        try:
            # Grundlegende Preise
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values if 'volume' in df.columns else np.ones(len(close))
            
            # Feature-Dictionary (nur skalare Werte)
            features_dict = {}
            
            # 1. Price Changes (letzte Werte)
            features_dict['change_1'] = self.calculate_price_change_last(close, 1)
            features_dict['change_2'] = self.calculate_price_change_last(close, 2)
            features_dict['change_3'] = self.calculate_price_change_last(close, 3)
            features_dict['change_5'] = self.calculate_price_change_last(close, 5)
            features_dict['change_10'] = self.calculate_price_change_last(close, 10)
            
            # 2. RSI (letzter Wert)
            rsi_values = self.calculate_rsi(close)
            features_dict['RSI'] = float(rsi_values[-1]) if len(rsi_values) > 0 else 50.0
            
            # 3. MACD (letzte Werte)
            macd, macd_signal, macd_hist = self.calculate_macd(close)
            features_dict['MACD'] = float(macd[-1]) if len(macd) > 0 else 0.0
            features_dict['MACD_signal'] = float(macd_signal[-1]) if len(macd_signal) > 0 else 0.0
            features_dict['MACD_hist'] = float(macd_hist[-1]) if len(macd_hist) > 0 else 0.0
            
            # 4. ADX (letzter Wert)
            adx_values = self.calculate_adx(high, low, close)
            features_dict['ADX'] = float(adx_values[-1]) if len(adx_values) > 0 else 25.0
            
            # 5. Bollinger Bands (letzte Werte)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            bb_upper_last = float(bb_upper[-1]) if len(bb_upper) > 0 else close[-1] * 1.02
            bb_middle_last = float(bb_middle[-1]) if len(bb_middle) > 0 else close[-1]
            bb_lower_last = float(bb_lower[-1]) if len(bb_lower) > 0 else close[-1] * 0.98
            
            features_dict['bb_upper'] = bb_upper_last
            features_dict['bb_middle'] = bb_middle_last
            features_dict['bb_lower'] = bb_lower_last
            
            # Berechne bb_width und bb_position sicher
            if bb_middle_last != 0:
                features_dict['bb_width'] = (bb_upper_last - bb_lower_last) / bb_middle_last
            else:
                features_dict['bb_width'] = 0.0
                
            if (bb_upper_last - bb_lower_last) != 0:
                features_dict['bb_position'] = (close[-1] - bb_lower_last) / (bb_upper_last - bb_lower_last)
            else:
                features_dict['bb_position'] = 0.5
                
            # 6. ATR (letzter Wert)
            atr_values = self.calculate_atr(high, low, close)
            features_dict['ATR'] = float(atr_values[-1]) if len(atr_values) > 0 else 0.001
            
            # ATR normalized
            if close[-1] != 0:
                features_dict['ATR_norm'] = features_dict['ATR'] / close[-1]
            else:
                features_dict['ATR_norm'] = 0.0
                
            # 7. Moving Averages (letzte Werte)
            sma_5 = self.calculate_sma(close, 5)
            sma_10 = self.calculate_sma(close, 10)
            sma_20 = self.calculate_sma(close, 20)
            sma_50 = self.calculate_sma(close, 50)
            sma_100 = self.calculate_sma(close, 100)
            
            sma_5_last = float(sma_5[-1]) if len(sma_5) > 0 else close[-1]
            sma_10_last = float(sma_10[-1]) if len(sma_10) > 0 else close[-1]
            sma_20_last = float(sma_20[-1]) if len(sma_20) > 0 else close[-1]
            sma_50_last = float(sma_50[-1]) if len(sma_50) > 0 else close[-1]
            sma_100_last = float(sma_100[-1]) if len(sma_100) > 0 else close[-1]
            
            # 8. MA Crossovers (Differenzen)
            features_dict['sma5_20'] = sma_5_last - sma_20_last
            features_dict['sma10_20'] = sma_10_last - sma_20_last
            features_dict['sma20_50'] = sma_20_last - sma_50_last
            features_dict['sma10_50'] = sma_10_last - sma_50_last
            features_dict['sma5_50'] = sma_5_last - sma_50_last
            features_dict['sma20_100'] = sma_20_last - sma_100_last
            features_dict['sma10_100'] = sma_10_last - sma_100_last
            features_dict['sma5_100'] = sma_5_last - sma_100_last
            
            # 9. Williams %R (letzter Wert)
            williams_values = self.calculate_williams_r(high, low, close)
            features_dict['williams_r'] = float(williams_values[-1]) if len(williams_values) > 0 else -50.0
            
            # 10. Momentum (letzter Wert)
            momentum_values = self.calculate_momentum(close, 10)
            features_dict['momentum'] = float(momentum_values[-1]) if len(momentum_values) > 0 else 0.0
            
            # 11. High-Low Spread
            if close[-1] != 0:
                features_dict['hl_spread'] = (high[-1] - low[-1]) / close[-1]
            else:
                features_dict['hl_spread'] = 0.002
                
            # 12. Volume Ratio
            if len(volume) > 10:
                avg_volume = np.mean(volume[-10:])
                if avg_volume != 0:
                    features_dict['volume_ratio'] = volume[-1] / avg_volume
                else:
                    features_dict['volume_ratio'] = 1.0
            else:
                features_dict['volume_ratio'] = 1.0
                
            # Konvertiere zu DataFrame (nur eine Zeile)
            features_df = pd.DataFrame([features_dict])
            
            # Fülle fehlende Werte
            features_df = features_df.fillna(0)
            
            return features_df
            
        except Exception as e:
            print(f"❌ Fehler in calculate_all_features_last: {e}")
            traceback.print_exc()
            return self.create_dummy_features()
            
    def calculate_price_change_last(self, prices, period):
        """Berechnet Preisänderung über Periode (nur letzter Wert)."""
        if len(prices) < period + 1:
            return 0.0
        if prices[-period-1] != 0:
            return (prices[-1] - prices[-period-1]) / prices[-period-1]
        return 0.0
        
    def calculate_rsi(self, prices, period=14):
        """Berechnet RSI."""
        if len(prices) < period + 1:
            return np.array([50.0])
            
        if TA_LIB_AVAILABLE:
            return talib.RSI(prices, timeperiod=period)
        else:
            # Manuelle RSI Berechnung
            deltas = np.diff(prices)
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum()/period
            down = -seed[seed < 0].sum()/period
            rs = up/down if down != 0 else 0
            rsi = np.zeros_like(prices)
            rsi[:period] = 100. - 100./(1.+rs)
            
            for i in range(period, len(prices)):
                delta = deltas[i-1]
                if delta > 0:
                    upval = delta
                    downval = 0.
                else:
                    upval = 0.
                    downval = -delta
                    
                up = (up*(period-1) + upval)/period
                down = (down*(period-1) + downval)/period
                rs = up/down if down != 0 else 0
                rsi[i] = 100. - 100./(1.+rs)
                
            return rsi
            
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Berechnet MACD."""
        if len(prices) < slow:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
            
        if TA_LIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(prices, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd, macd_signal, macd_hist
        else:
            # Manuelle MACD Berechnung
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            macd_signal_line = self.calculate_ema(macd_line, signal)
            macd_histogram = macd_line - macd_signal_line
            
            return macd_line, macd_signal_line, macd_histogram
            
    def calculate_ema(self, prices, period):
        """Berechnet EMA."""
        if len(prices) < period:
            return np.zeros_like(prices)
            
        ema = np.zeros_like(prices)
        ema[period-1] = np.mean(prices[:period])
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
        return ema
        
    def calculate_adx(self, high, low, close, period=14):
        """Berechnet ADX."""
        if len(high) < period * 2:
            return np.full_like(close, 25.0)
            
        if TA_LIB_AVAILABLE:
            return talib.ADX(high, low, close, timeperiod=period)
        else:
            # Vereinfachte ADX Berechnung
            return np.full_like(close, 25.0)
            
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Berechnet Bollinger Bands."""
        if len(prices) < period:
            return np.zeros_like(prices), np.zeros_like(prices), np.zeros_like(prices)
            
        sma = self.calculate_sma(prices, period)
        std = np.zeros_like(prices)
        
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
            
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
        
    def calculate_sma(self, prices, period):
        """Berechnet SMA."""
        if len(prices) < period:
            return np.zeros_like(prices)
            
        sma = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
            
        return sma
        
    def calculate_atr(self, high, low, close, period=14):
        """Berechnet ATR."""
        if len(high) < period + 1:
            return np.full_like(high, 0.001)
            
        if TA_LIB_AVAILABLE:
            return talib.ATR(high, low, close, timeperiod=period)
        else:
            # Manuelle ATR Berechnung
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
                
            atr = np.zeros_like(high)
            atr[period] = np.mean(tr[1:period+1])
            
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                
            return atr
            
    def calculate_williams_r(self, high, low, close, period=14):
        """Berechnet Williams %R."""
        if len(high) < period:
            return np.full_like(close, -50.0)
            
        williams_r = np.zeros_like(close)
        for i in range(period-1, len(high)):
            highest_high = np.max(high[i-period+1:i+1])
            lowest_low = np.min(low[i-period+1:i+1])
            
            if highest_high != lowest_low:
                williams_r[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
            else:
                williams_r[i] = -50
                
        return williams_r
        
    def calculate_momentum(self, prices, period=10):
        """Berechnet Momentum."""
        if len(prices) < period:
            return np.zeros_like(prices)
            
        momentum = np.zeros_like(prices)
        for i in range(period-1, len(prices)):
            momentum[i] = prices[i] - prices[i-period]
            
        return momentum
        
    def create_dummy_features(self) -> pd.DataFrame:
        """Erstellt Dummy-Features als Fallback."""
        try:
            # Standard-Features
            features = {
                'RSI': [50.0],
                'MACD': [0.0],
                'MACD_signal': [0.0],
                'MACD_hist': [0.0],
                'ADX': [25.0],
                'bb_upper': [1.10200],
                'bb_middle': [1.10000],
                'bb_lower': [1.09800],
                'bb_width': [0.004],
                'bb_position': [0.5],
                'ATR': [0.001],
                'ATR_norm': [0.001],
                'sma5_20': [0.0],
                'sma10_20': [0.0],
                'sma20_50': [0.0],
                'sma10_50': [0.0],
                'sma5_50': [0.0],
                'sma20_100': [0.0],
                'sma10_100': [0.0],
                'sma5_100': [0.0],
                'williams_r': [-50.0],
                'momentum': [0.0],
                'hl_spread': [0.002],
                'volume_ratio': [1.0],
                'change_1': [0.0],
                'change_2': [0.0],
                'change_3': [0.0],
                'change_5': [0.0],
                'change_10': [0.0]
            }
            
            # Wenn Feature-Columns bekannt sind, stelle Kompatibilität sicher
            if self.feature_columns is not None:
                for col in self.feature_columns:
                    if col not in features:
                        features[col] = [0.0]
                        
                # Filtere nur bekannte Columns
                features = {k: v for k, v in features.items() if k in self.feature_columns}
                
            features_df = pd.DataFrame(features)
            print("⚠️  Verwende Dummy-Features")
            return features_df
            
        except Exception as e:
            print(f"❌ Fehler bei Dummy-Features: {e}")
            # Letzter Fallback: einfache Features
            return pd.DataFrame({'feature_1': [0.0], 'feature_2': [0.0], 'feature_3': [0.0]})
            
    def generate_signal(self, features: Optional[pd.DataFrame] = None) -> Tuple[str, float]:
        """Generiert ein Trading-Signal."""
        try:
            if not self.model_loaded:
                print("⚠️  Kein ML-Modell geladen - gebe HOLD zurück")
                return "HOLD", 50.0
                
            # Hole Features wenn nicht gegeben
            if features is None:
                features = self.get_current_features()
                
            # Prüfe ob Features mit Modell kompatibel sind
            if self.feature_columns is not None:
                # Finde fehlende Features
                missing_cols = set(self.feature_columns) - set(features.columns)
                
                if missing_cols:
                    # Füge fehlende Features hinzu
                    for col in missing_cols:
                        if col not in features.columns:
                            features[col] = 0.0
                            
                # Stelle sicher, dass Features in richtiger Reihenfolge sind
                features = features[self.feature_columns]
                
            # Fülle NaN Werte
            features = features.fillna(0)
            
            # Skaliere Features
            if self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform(features)
                except Exception as e:
                    print(f"⚠️  Fehler beim Skalieren: {e} - verwende unskalierte Features")
                    features_scaled = features.values
            else:
                features_scaled = features.values
                
            # Mache Vorhersage
            try:
                prediction = self.model.predict(features_scaled)
                probabilities = self.model.predict_proba(features_scaled)
                
                # Bestimme Signal und Confidence
                if prediction[0] == 1:
                    signal = "BUY"
                else:
                    signal = "SELL"
                    
                confidence = float(probabilities[0].max() * 100)
                
                # Debug
                print(f"📊 ML Vorhersage: {signal} mit {confidence:.1f}% Confidence")
                
                return signal, confidence
                
            except Exception as e:
                print(f"❌ Fehler bei ML Vorhersage: {e}")
                traceback.print_exc()
                return "HOLD", 50.0
                
        except Exception as e:
            print(f"❌ Fehler bei Signal-Generierung: {e}")
            traceback.print_exc()
            return "HOLD", 50.0
            
    def calculate_profit_loss(self, position_data: Dict, current_price: float) -> float:
        """
        Berechnet P&L für eine Position.
        
        Args:
            position_data: Dict mit Position-Daten
            current_price: Aktueller Marktpreis
            
        Returns:
            float: Profit/Loss in USD
        """
        try:
            if not position_data:
                return 0.0
                
            # Extrahiere Position-Daten
            entry_price = position_data.get('entry_price', 0)
            volume = position_data.get('volume', 0.01)
            position_type = position_data.get('type', 'BUY')
            symbol = position_data.get('symbol', 'EURUSD')
            
            if entry_price == 0 or current_price == 0:
                return 0.0
                
            # Berechne Preis-Differenz
            price_diff = 0
            if position_type == 'BUY':
                price_diff = current_price - entry_price
            else:  # SELL
                price_diff = entry_price - current_price
                
            # Berechne Profit in Pips
            pip_size = 0.0001
            if 'JPY' in symbol:
                pip_size = 0.01
                
            profit_pips = price_diff / pip_size
            
            # Berechne Geldwert
            pip_value_per_lot = 10  # USD für EURUSD
            
            # Berechne Gesamt-Profit
            profit_usd = profit_pips * pip_value_per_lot * volume
            
            # Berücksichtige Spread
            spread_cost = 2 * pip_value_per_lot * volume
            net_profit = profit_usd - spread_cost
            
            return round(net_profit, 2)
            
        except Exception as e:
            print(f"❌ Fehler in P&L Berechnung: {e}")
            traceback.print_exc()
            return 0.0
            
    def get_portfolio_summary(self) -> Dict:
        """
        Gibt detaillierte Portfolio-Zusammenfassung mit P&L.
        
        Returns:
            Dict: Portfolio-Zusammenfassung
        """
        summary = {
            'balance': 0.0,
            'equity': 0.0,
            'margin': 0.0,
            'free_margin': 0.0,
            'open_positions': 0,
            'total_positions_value': 0.0,
            'total_pnl': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_trades': 0,
            'win_rate': 0.0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        try:
            if self.portfolio is None:
                return summary
                
            # Basiswerte vom Portfolio
            summary['balance'] = float(self.portfolio.balance)
            summary['open_positions'] = len(self.portfolio.positions)
            summary['total_trades'] = len(self.portfolio.trade_history)
            
            # Berechne P&L für offene Positionen
            if hasattr(self, 'current_price') and self.current_price > 0:
                total_unrealized = 0.0
                for position in self.portfolio.positions:
                    pnl = self.calculate_profit_loss(position, self.current_price)
                    position['unrealized_pnl'] = pnl
                    total_unrealized += pnl
                    
                    # Berechne Positionswert
                    position_value = position.get('volume', 0.01) * 100000
                    summary['total_positions_value'] += position_value
                    
                summary['unrealized_pnl'] = round(total_unrealized, 2)
                
            # Berechne realisierten P&L aus Trade History
            total_realized = 0.0
            total_wins = 0.0
            total_losses = 0.0
            wins_count = 0
            losses_count = 0
            largest_win = 0.0
            largest_loss = 0.0
            
            for trade in self.portfolio.trade_history:
                profit = trade.get('profit', 0)
                total_realized += profit
                
                if profit > 0:
                    wins_count += 1
                    total_wins += profit
                    largest_win = max(largest_win, profit)
                    summary['winning_trades'] += 1
                elif profit < 0:
                    losses_count += 1
                    total_losses += abs(profit)
                    largest_loss = min(largest_loss, profit)
                    summary['losing_trades'] += 1
                    
            summary['realized_pnl'] = round(total_realized, 2)
            summary['total_pnl'] = summary['unrealized_pnl'] + summary['realized_pnl']
            summary['equity'] = summary['balance'] + summary['total_pnl']
            
            # Berechne Statistiken
            if summary['winning_trades'] + summary['losing_trades'] > 0:
                summary['win_rate'] = round(
                    (summary['winning_trades'] / (summary['winning_trades'] + summary['losing_trades'])) * 100, 
                    2
                )
                
            summary['largest_win'] = round(largest_win, 2)
            summary['largest_loss'] = round(largest_loss, 2)
            
            if wins_count > 0:
                summary['avg_win'] = round(total_wins / wins_count, 2)
            if losses_count > 0:
                summary['avg_loss'] = round(total_losses / losses_count, 2)
                
            if total_losses > 0:
                summary['profit_factor'] = round(total_wins / total_losses, 2)
                
            # Margin Berechnung
            leverage = 30
            if summary['total_positions_value'] > 0:
                summary['margin'] = round(summary['total_positions_value'] / leverage, 2)
                summary['free_margin'] = round(summary['equity'] - summary['margin'], 2)
                
        except Exception as e:
            print(f"❌ Fehler in Portfolio-Summary: {e}")
            traceback.print_exc()
            
        return summary
        
    def display_portfolio_dashboard(self):
        """Zeigt detailliertes Portfolio-Dashboard an."""
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*80)
        print("📊 PORTFOLIO DASHBOARD - DETAILIERTE ÜBERSICHT")
        print("="*80)
        
        print(f"\n💰 KAPITAL:")
        print(f"   Kontostand:      ${summary['balance']:,.2f}")
        print(f"   Eigenkapital:    ${summary['equity']:,.2f}")
        print(f"   Margin:          ${summary['margin']:,.2f}")
        print(f"   Freie Margin:    ${summary['free_margin']:,.2f}")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Gesamt-P&L:      ${summary['total_pnl']:,.2f}")
        print(f"   Realisiert:      ${summary['realized_pnl']:,.2f}")
        print(f"   Unrealisiert:    ${summary['unrealized_pnl']:,.2f}")
        
        print(f"\n🎯 TRADING STATISTIK:")
        print(f"   Offene Positionen: {summary['open_positions']}")
        print(f"   Gesamt Trades:     {summary['total_trades']}")
        print(f"   Gewinne:          {summary['winning_trades']}")
        print(f"   Verluste:         {summary['losing_trades']}")
        print(f"   Win Rate:         {summary['win_rate']}%")
        print(f"   Profit Faktor:    {summary['profit_factor']}")
        
        print(f"\n📊 DETAILS:")
        print(f"   Größter Gewinn:  ${summary['largest_win']:,.2f}")
        print(f"   Größter Verlust: ${summary['largest_loss']:,.2f}")
        print(f"   Durch. Gewinn:   ${summary['avg_win']:,.2f}")
        print(f"   Durch. Verlust:  ${summary['avg_loss']:,.2f}")
        
        print(f"\n📋 OFFENE POSITIONEN:")
        if summary['open_positions'] > 0 and hasattr(self, 'current_price'):
            for i, pos in enumerate(self.portfolio.positions, 1):
                pnl = self.calculate_profit_loss(pos, self.current_price)
                status = "📈 GRÜN" if pnl >= 0 else "📉 ROT"
                print(f"   {i}. {pos.get('type', 'N/A')} {pos.get('symbol', 'N/A')} "
                      f"@{pos.get('entry_price', 0):.5f} "
                      f"[Vol: {pos.get('volume', 0)}] "
                      f"P&L: ${pnl:.2f} {status}")
        else:
            print("   Keine offenen Positionen")
            
        print("="*80)
        
    def update_current_price(self, price: float):
        """Aktualisiert den aktuellen Preis für P&L Berechnungen."""
        self.current_price = price
        return True


def run_enhanced_ml_trading(iterations: int = 5, symbol: str = "EURUSD"):
    """
    Führt erweitertes ML-Trading aus.
    
    Args:
        iterations: Anzahl der Iterationen
        symbol: Trading-Symbol
    """
    print(f"🚀 Starte Enhanced ML Trading mit {iterations} Iterationen...")
    
    # Simulierte Portfolio-Klasse
    class SimplePortfolio:
        def __init__(self):
            self.balance = 10000.0
            self.positions = []
            self.trade_history = []
            
    portfolio = SimplePortfolio()
    engine = EnhancedMLTradingEngine(portfolio)
    engine.symbol = symbol
    
    # Simulierte Preis-Daten
    base_price = 1.10000
    results = []
    
    for i in range(iterations):
        print(f"\n{'='*60}")
        print(f"📊 ITERATION {i+1}/{iterations}")
        print(f"{'='*60}")
        
        # Simuliere Preis-Update
        price_change = np.random.uniform(-0.002, 0.002)
        current_price = base_price + price_change
        engine.update_current_price(current_price)
        
        # Generiere Signal
        signal, confidence = engine.generate_signal()
        
        # Sammle Ergebnis
        result = {
            "iteration": i + 1,
            "signal": signal,
            "confidence": confidence,
            "price": current_price,
            "time": datetime.now().strftime("%H:%M:%S")
        }
        
        # Simuliere Trade basierend auf Signal
        if confidence > 65 and signal != "HOLD":
            result["action"] = "EXECUTE"
            print(f"🎯 Trade ausführen: {signal} (Confidence: {confidence:.1f}%)")
            
            # Erstelle Position
            position = {
                'type': signal,
                'symbol': symbol,
                'entry_price': current_price,
                'volume': 0.01,
                'timestamp': datetime.now().isoformat(),
                'sl': current_price - (0.002 if signal == "BUY" else -0.002),
                'tp': current_price + (0.004 if signal == "BUY" else -0.004)
            }
            
            portfolio.positions.append(position)
            
            # Simuliere Trade-Abschluss
            if len(portfolio.positions) > 0 and i % 2 == 0:
                closed_pos = portfolio.positions.pop(0)
                profit = np.random.uniform(-10, 20)
                closed_pos['profit'] = profit
                closed_pos['exit_price'] = current_price
                closed_pos['exit_time'] = datetime.now().isoformat()
                portfolio.trade_history.append(closed_pos)
                portfolio.balance += profit
                print(f"💵 Trade geschlossen: ${profit:.2f}")
        else:
            result["action"] = "HOLD"
            result["reason"] = "Confidence zu niedrig" if confidence < 65 else "HOLD Signal"
            print(f"⏸️  Kein Trade - Confidence: {confidence:.1f}% (Threshold: 65%)")
            
        results.append(result)
        
        # Zeige Portfolio-Dashboard
        engine.display_portfolio_dashboard()
        
        time.sleep(1)
        
    print(f"\n✅ Enhanced ML Trading abgeschlossen!")
    return results


if __name__ == "__main__":
    # Test der Enhanced Engine
    results = run_enhanced_ml_trading(iterations=3)
    print(f"\n📋 Ergebnisse: {len(results)} Iterationen")
    for r in results:
        print(f"  Iteration {r['iteration']}: {r['signal']} ({r['confidence']:.1f}%) - {r['action']}")