"""
EURUSD Feature Engineer für M5 Daten 2002-2019
Spezialisiert für Daten ohne Volume
"""

import pandas as pd
import numpy as np
import os  # HIER HINZUGEFÜGT
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

class EURUSDFeatureEngineer:
    """Feature Engineer für EURUSD M5 Daten (ohne Volume)"""
    
    def __init__(self):
        self.feature_count = 0
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt preisbasierte Features hinzu"""
        df = df.copy()
        
        # 1. Basic Price Features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price ranges
        df['hl_range'] = df['high'] - df['low']
        df['oc_range'] = abs(df['close'] - df['open'])
        df['hl_range_pct'] = df['hl_range'] / df['close']
        df['oc_range_pct'] = df['oc_range'] / df['close']
        
        # Price positions
        df['close_position'] = (df['close'] - df['low']) / df['hl_range'].replace(0, 1e-10)
        df['open_position'] = (df['open'] - df['low']) / df['hl_range'].replace(0, 1e-10)
        
        return df
    
    def add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Trend-Features hinzu"""
        df = df.copy()
        
        # Moving Averages
        windows = [5, 10, 20, 50, 100]
        for w in windows:
            df[f'sma_{w}'] = df['close'].rolling(window=w).mean()
            df[f'ema_{w}'] = df['close'].ewm(span=w, adjust=False).mean()
        
        # Trend strength
        df['trend_sma'] = df['sma_10'] - df['sma_50']
        df['trend_ema'] = df['ema_10'] - df['ema_50']
        df['trend_strength'] = abs(df['trend_sma']) / df['close']
        
        # Crossovers
        df['sma_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ema_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Volatilitäts-Features hinzu"""
        df = df.copy()
        
        # Volatility measures
        windows = [10, 20, 50]
        for w in windows:
            df[f'volatility_{w}'] = df['returns'].rolling(window=w).std()
            df[f'log_volatility_{w}'] = df['log_returns'].rolling(window=w).std()
        
        # ATR
        df['tr'] = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)
        
        for w in [14, 20]:
            df[f'atr_{w}'] = df['tr'].rolling(window=w).mean()
            df[f'atr_pct_{w}'] = df[f'atr_{w}'] / df['close']
        
        return df
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt technische Indikatoren hinzu"""
        df = df.copy()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for w in [20, 50]:
            sma = df['close'].rolling(window=w).mean()
            std = df['close'].rolling(window=w).std()
            df[f'bb_upper_{w}'] = sma + 2 * std
            df[f'bb_lower_{w}'] = sma - 2 * std
            df[f'bb_width_{w}'] = df[f'bb_upper_{w}'] - df[f'bb_lower_{w}']
            df[f'bb_position_{w}'] = (df['close'] - df[f'bb_lower_{w}']) / df[f'bb_width_{w}'].replace(0, 1e-10)
        
        return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt zeitliche Features hinzu"""
        df = df.copy()
        
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Time components
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['week_of_year'] = df['datetime'].dt.isocalendar().week
            
            # Trading sessions
            df['is_london'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
            df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] <= 22)).astype(int)
            df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] <= 16)).astype(int)
            df['is_asia'] = ((df['hour'] >= 0) & (df['hour'] <= 9)).astype(int)
            
            # Time of day categories
            df['is_morning'] = ((df['hour'] >= 8) & (df['hour'] <= 12)).astype(int)
            df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] <= 16)).astype(int)
            df['is_evening'] = ((df['hour'] >= 16) & (df['hour'] <= 20)).astype(int)
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt lag-Features hinzu"""
        df = df.copy()
        
        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
        
        # Momentum
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            df[f'roc_{period}'] = df['close'].pct_change(period)
        
        return df
    
    def add_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt Kerzen-Features hinzu"""
        df = df.copy()
        
        # Candle types
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_bearish'] = (df['close'] < df['open']).astype(int)
        df['is_doji'] = (abs(df['close'] - df['open']) / df['hl_range'] < 0.1).astype(int)
        
        # Shadows
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-10)
        
        # Body characteristics
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_to_range'] = df['body_size'] / df['hl_range'].replace(0, 1e-10)
        
        return df
    
    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fügt alle Features hinzu"""
        print("=" * 60)
        print("🔧 EURUSD FEATURE ENGINEERING")
        print("=" * 60)
        
        print(f"📊 Eingabe: {len(df)} Kerzen")
        
        # Sequenziell alle Features hinzufügen
        df = self.add_price_features(df)
        print("✅ Preis-Features hinzugefügt")
        
        df = self.add_trend_features(df)
        print("✅ Trend-Features hinzugefügt")
        
        df = self.add_volatility_features(df)
        print("✅ Volatilitäts-Features hinzugefügt")
        
        df = self.add_technical_indicators(df)
        print("✅ Technische Indikatoren hinzugefügt")
        
        df = self.add_time_features(df)
        print("✅ Zeit-Features hinzugefügt")
        
        df = self.add_lag_features(df)
        print("✅ Lag-Features hinzugefügt")
        
        df = self.add_candle_features(df)
        print("✅ Kerzen-Features hinzugefügt")
        
        # Cleanup
        initial_len = len(df)
        df = df.fillna(method='ffill').fillna(method='bfill')
        df = df.replace([np.inf, -np.inf], 0)
        df = df.dropna()
        
        removed = initial_len - len(df)
        if removed > 0:
            print(f"🧹 {removed} Zeilen mit NaN/Inf entfernt")
        
        print(f"✅ Fertig: {len(df)} Kerzen, {len(df.columns)} Features")
        
        return df
    
    def create_ml_training_samples(self, df: pd.DataFrame, 
                                   lookback: int = 100, 
                                   forward: int = 5) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Erstellt ML-Trainingsdaten
        """
        print(f"\n🎯 ERSTELLE ML-TRAININGSDATEN")
        print(f"   Lookback: {lookback} Kerzen")
        print(f"   Forward: {forward} Kerzen")
        
        # Features berechnen
        df_features = self.add_all_features(df)
        
        # Feature-Spalten identifizieren
        exclude = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df_features.columns if col not in exclude]
        
        print(f"📊 {len(feature_cols)} Feature-Spalten")
        
        # Label generieren
        df_features['future_close'] = df_features['close'].shift(-forward)
        df_features['future_return'] = (df_features['future_close'] - df_features['close']) / df_features['close']
        df_features['label'] = (df_features['future_return'] > 0).astype(int)
        
        # Samples erstellen
        X_samples = []
        y_samples = []
        
        total_possible = len(df_features) - lookback - forward
        
        print(f"🔨 Erstelle Samples...")
        
        for i in range(lookback, len(df_features) - forward):
            # Features aus Fenster
            window = df_features.iloc[i-lookback:i][feature_cols]
            features_flat = window.values.flatten()
            
            # Label
            label = df_features.iloc[i]['label']
            
            X_samples.append(features_flat)
            y_samples.append(label)
            
            # Fortschritt
            if len(X_samples) % 5000 == 0:
                print(f"   {len(X_samples):,} / {total_possible:,}")
        
        # Konvertieren
        X = np.array(X_samples)
        y = np.array(y_samples)
        
        # Feature-Namen
        feature_names = []
        for lag_idx in range(lookback):
            for col in feature_cols:
                feature_names.append(f"{col}_lag{lookback-lag_idx}")
        
        # Ergebnisse
        print(f"\n✅ TRAININGSDATEN:")
        print(f"   X: {X.shape}")
        print(f"   y: {y.shape}")
        print(f"   Features/Sample: {X.shape[1]:,}")
        print(f"   Samples: {X.shape[0]:,}")
        
        # Klassenverteilung
        pos = np.sum(y == 1)
        neg = np.sum(y == 0)
        
        print(f"\n📊 KLASSEN:")
        print(f"   ↑ STEIGEND: {pos:,} ({pos/len(y)*100:.1f}%)")
        print(f"   ↓ FALLEND:  {neg:,} ({neg/len(y)*100:.1f}%)")
        
        return X, y, feature_names

# Hilfsfunktionen
def load_eurusd_data(filepath: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Lädt EURUSD Daten"""
    print(f"📁 Lade: {os.path.basename(filepath)}")
    
    try:
        if max_rows:
            df = pd.read_csv(filepath, nrows=max_rows)
        else:
            df = pd.read_csv(filepath)
        
        print(f"   Kerzen: {len(df):,}")
        
        # Zeitstempel parsen
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            print(f"   Zeit: {df['datetime'].iloc[0]} bis {df['datetime'].iloc[-1]}")
        
        return df
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def save_ml_data(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """Speichert ML-Daten"""
    import joblib
    
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichern
    np.save(os.path.join(output_dir, "X_eurusd.npy"), X)
    np.save(os.path.join(output_dir, "y_eurusd.npy"), y)
    joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))
    
    print(f"\n💾 GESPEICHERT:")
    print(f"   X_eurusd.npy: {X.shape}")
    print(f"   y_eurusd.npy: {y.shape}")
    print(f"   feature_names.pkl: {len(feature_names):,} Features")

if __name__ == "__main__":
    # Quick Test
    print("🧪 EURUSD Feature Engineer Test")
    
    # Testdaten
    dates = pd.date_range('2024-01-01', periods=500, freq='5min')
    test_df = pd.DataFrame({
        'datetime': dates,
        'open': 1.1 + np.random.randn(500).cumsum() * 0.0001,
        'high': 1.1005 + np.random.randn(500).cumsum() * 0.0001,
        'low': 1.0995 + np.random.randn(500).cumsum() * 0.0001,
        'close': 1.1 + np.random.randn(500).cumsum() * 0.0001,
        'volume': 0
    })
    
    engineer = EURUSDFeatureEngineer()
    X, y, features = engineer.create_ml_training_samples(test_df, lookback=20, forward=3)
    
    print(f"\n✅ TEST ERFOLGREICH!")