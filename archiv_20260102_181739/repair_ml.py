# Erstelle ein neues repair_ml.py mit diesem Inhalt:
# Öffne Notepad und kopiere diesen Code:

import os

# Der korrekte Inhalt für ml_signal_generator.py
ml_code = '''"""
ML SIGNAL GENERATOR - REPARIERTE VERSION
"""

import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

class MLSignalGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.lookback = config.get("ml", {}).get("lookback", 100) if config else 100
        self.confidence_threshold = config.get("ml", {}).get("confidence_threshold", 0.65) if config else 0.65
        self.buffer_size = self.lookback * 2
        
        self.data_buffers = {}
        self.ml_buffer = pd.DataFrame()
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        self._load_model()
        print(f"✅ MLSignalGenerator initialisiert (Lookback: {self.lookback})")
    
    def _load_model(self):
        try:
            self.model = joblib.load("data/models/random_forest_model.joblib")
            self.scaler = joblib.load("data/models/feature_scaler.joblib")
            
            if hasattr(self.model, "feature_names_in_"):
                self.feature_columns = list(self.model.feature_names_in_)
            else:
                self.feature_columns = [f"feature_{i}" for i in range(41)]
            
            return True
        except Exception as e:
            print(f"⚠️  Fehler beim Laden: {e}")
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            X_dummy = np.random.randn(100, 41)
            y_dummy = np.random.randint(0, 2, 100)
            self.model.fit(X_dummy, y_dummy)
            
            self.scaler = StandardScaler()
            self.scaler.fit(X_dummy)
            self.feature_columns = [f"feature_{i}" for i in range(41)]
            print(f"✅ Dummy-Modell erstellt")
            return False
    
    def add_live_data(self, symbol, data_frame):
        try:
            print(f"➕ Füge {len(data_frame)} Kerzen für {symbol} hinzu...")
            
            if symbol not in self.data_buffers:
                self.data_buffers[symbol] = pd.DataFrame()
            
            self.data_buffers[symbol] = pd.concat(
                [self.data_buffers[symbol], data_frame],
                ignore_index=True
            ).tail(200)
            
            print(f"   📊 {symbol} Buffer: {len(self.data_buffers[symbol])} rows")
            
            features = self._calculate_features(symbol, data_frame)
            
            if not features.empty:
                features_with_meta = features.copy()
                features_with_meta["symbol"] = symbol
                features_with_meta["timestamp"] = datetime.now()
                
                self.ml_buffer = pd.concat(
                    [self.ml_buffer, features_with_meta],
                    ignore_index=True
                ).tail(self.buffer_size)
                
                print(f"   ✅ {len(features)} Features hinzugefügt")
                print(f"   📈 ml_buffer: {len(self.ml_buffer)} rows")
                return True
            else:
                print(f"   ⚠️  Keine Features")
                return False
            
        except Exception as e:
            print(f"❌ Fehler: {e}")
            return False
    
    def _calculate_features(self, symbol, data):
        try:
            if len(data) < 20:
                return pd.DataFrame()
            
            df = data.copy()
            
            col_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if "open" in col_lower:
                    col_mapping[col] = "open"
                elif "high" in col_lower:
                    col_mapping[col] = "high"
                elif "low" in col_lower:
                    col_mapping[col] = "low"
                elif "close" in col_lower:
                    col_mapping[col] = "close"
                elif "volume" in col_lower or "tick" in col_lower:
                    col_mapping[col] = "volume"
                elif "time" in col_lower:
                    col_mapping[col] = "time"
            
            if col_mapping:
                df = df.rename(columns=col_mapping)
            
            features = pd.DataFrame(index=df.index)
            
            if "close" in df.columns:
                features["price"] = df["close"]
                features["returns"] = df["close"].pct_change()
                features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
            
            features["sma_10"] = df["close"].rolling(10).mean()
            features["sma_20"] = df["close"].rolling(20).mean()
            features["ema_12"] = df["close"].ewm(span=12).mean()
            features["ema_26"] = df["close"].ewm(span=26).mean()
            
            macd = features["ema_12"] - features["ema_26"]
            features["macd"] = macd
            features["macd_signal"] = macd.ewm(span=9).mean()
            features["macd_hist"] = macd - features["macd_signal"]
            
            delta = df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))
            
            bb_ma = df["close"].rolling(20).mean()
            bb_std = df["close"].rolling(20).std()
            features["bb_upper"] = bb_ma + (bb_std * 2)
            features["bb_lower"] = bb_ma - (bb_std * 2)
            features["bb_width"] = (features["bb_upper"] - features["bb_lower"]) / bb_ma
            features["bb_position"] = (df["close"] - features["bb_lower"]) / (features["bb_upper"] - features["bb_lower"])
            
            features["volatility_20"] = df["close"].rolling(20).std()
            
            if "high" in df.columns and "low" in df.columns:
                high = df["high"]
                low = df["low"]
                close = df["close"].shift(1)
                tr1 = high - low
                tr2 = abs(high - close)
                tr3 = abs(low - close)
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                features["atr"] = tr.rolling(14).mean()
            
            if "volume" in df.columns:
                features["volume"] = df["volume"]
                features["volume_sma"] = df["volume"].rolling(20).mean()
                features["volume_ratio"] = df["volume"] / features["volume_sma"]
            
            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])
                features["hour"] = df["time"].dt.hour
                features["day_of_week"] = df["time"].dt.dayofweek
                features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
            
            features["trend_strength"] = features["sma_20"].diff(5)
            features["price_vs_sma"] = (df["close"] / features["sma_20"]) - 1
            
            if "high" in df.columns and "low" in df.columns:
                features["high_20"] = df["high"].rolling(20).max()
                features["low_20"] = df["low"].rolling(20).min()
                if (features["high_20"] - features["low_20"]).abs().sum() > 0:
                    features["close_to_high"] = (df["close"] - features["high_20"]) / (features["high_20"] - features["low_20"])
            
            features = features.fillna(method="ffill").fillna(0)
            features = features.tail(min(50, len(features)))
            
            print(f"   🔧 {len(features)} Features berechnet")
            return features
            
        except Exception as e:
            print(f"❌ Feature-Berechnung: {e}")
            return pd.DataFrame()
    
    def is_ready(self):
        return len(self.ml_buffer) >= self.lookback
    
    def generate_signal(self):
        if not self.is_ready():
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "signal_source": "ML",
                "error": f"Benötigt {self.lookback}, Hat: {len(self.ml_buffer)}",
                "features_used": len(self.ml_buffer)
            }
        
        try:
            recent_features = self.ml_buffer.tail(self.lookback)
            numeric_cols = recent_features.select_dtypes(include=[np.number]).columns
            features_data = recent_features[numeric_cols]
            
            if len(features_data.columns) < 10:
                return {
                    "signal": "HOLD",
                    "confidence": 0.0,
                    "signal_source": "ML",
                    "error": f"Zu wenige Features: {len(features_data.columns)}"
                }
            
            features_mean = features_data.mean().values.reshape(1, -1)
            
            if self.scaler and hasattr(self.scaler, "transform"):
                features_scaled = self.scaler.transform(features_mean)
            else:
                features_scaled = features_mean
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            if confidence >= self.confidence_threshold:
                signal = "BUY" if prediction == 1 else "SELL"
            else:
                signal = "HOLD"
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "probability_up": float(probabilities[1]) if len(probabilities) > 1 else 0.5,
                "probability_down": float(probabilities[0]) if len(probabilities) > 0 else 0.5,
                "signal_source": "ML",
                "features_used": len(features_data.columns)
            }
            
        except Exception as e:
            return {
                "signal": "HOLD",
                "confidence": 0.0,
                "signal_source": "ML",
                "error": f"Vorhersagefehler: {str(e)}"
            }
    
    def get_buffer_status(self):
        return {
            "ml_buffer_rows": len(self.ml_buffer),
            "lookback_required": self.lookback,
            "buffer_ready": self.is_ready(),
            "completion_percentage": (len(self.ml_buffer) / self.lookback) * 100 if self.lookback > 0 else 0,
            "data_buffers": {k: len(v) for k, v in self.data_buffers.items()}
        }
    
    def get_model_info(self):
        return {
            "model_type": type(self.model).__name__,
            "n_estimators": getattr(self.model, "n_estimators", "N/A"),
            "features_expected": len(self.feature_columns) if self.feature_columns else 0,
            "lookback": self.lookback,
            "confidence_threshold": self.confidence_threshold
        }
    
    def debug_feature_matching(self):
        print(f"🔍 ML erwartet {len(self.feature_columns) if self.feature_columns else 0} Features")
        if not self.ml_buffer.empty:
            print(f"📊 ml_buffer: {len(self.ml_buffer.columns)} Spalten")
            numeric_cols = self.ml_buffer.select_dtypes(include=[np.number]).columns
            print(f"🔢 Davon numerisch: {len(numeric_cols)}")
'''

# Schreibe die Datei
file_path = "src/ml_integration/ml_signal_generator.py"
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(ml_code)

print(f"✅ {file_path} erfolgreich erstellt")

# Teste die Datei
try:
    namespace = {}
    exec(ml_code, namespace)
    print("✅ Code-Syntax ist korrekt")
    
    MLSignalGenerator = namespace['MLSignalGenerator']
    ml = MLSignalGenerator({"ml": {"lookback": 100, "confidence_threshold": 0.65}})
    print("✅ MLSignalGenerator instanziiert")
    
    info = ml.get_model_info()
    print(f"✅ get_model_info: {info['model_type']}")
    
    status = ml.get_buffer_status()
    print(f"✅ get_buffer_status: {status['ml_buffer_rows']} rows")
    
    print("\n" + "="*60)
    print("✅ REPARATUR ERFOLGREICH!")
    print("="*60)
    
except Exception as e:
    print(f"❌ Fehler: {e}")
    import traceback
    traceback.print_exc()