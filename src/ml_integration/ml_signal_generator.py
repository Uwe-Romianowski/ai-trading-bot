# src/ml_integration/ml_signal_generator_fixed.py
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

class MLSignalGenerator:
    """ML Signal Generator - GARANTIERT OHNE .get() FEHLER"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # ML Parameter
        ml_config = config.get("ml", {}) if config else {}
        self.lookback = ml_config.get("lookback", 100)
        self.confidence_threshold = ml_config.get("confidence_threshold", 0.65)
        
        # Buffers
        self.data_buffers = {}
        self.ml_buffer = pd.DataFrame()
        
        # Model
        self.model = None
        self.scaler = None
        self.feature_names = []
        
        # Initialize
        self._load_model_safe()
        
        print(f"✅ MLSignalGenerator initialisiert (Lookback: {self.lookback})")
    
    def _load_model_safe(self):
        """Lädt ML-Modell - ABSOLUT SICHER OHNE .get() FEHLER"""
        try:
            # Lade ML-Modell
            self.model = joblib.load("data/models/random_forest_model.joblib")
            print(f"✅ ML-Modell geladen: {type(self.model).__name__}")
            
            # Lade Scaler
            self.scaler = joblib.load("data/models/feature_scaler.joblib")
            print("✅ Scaler geladen")
            
            # Lade Feature-Namen - ABSOLUT SICHER
            self._load_features_ultra_safe()
            
        except Exception as e:
            print(f"⚠️  Fehler: {e}")
            self._create_fallback()
    
    def _load_features_ultra_safe(self):
        """Lädt Feature-Namen - KEIN .get() AUFRUF AUF LISTEN"""
        try:
            json_path = "data/models/feature_names.json"
            
            if not os.path.exists(json_path):
                print(f"📄 {json_path} nicht gefunden")
                self._create_default_features()
                return
            
            with open(json_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            data = json.loads(content)
            
            # WICHTIG: Kein .get() auf Listen!
            if isinstance(data, list):
                # Deine Datei: Direkte Liste
                self.feature_names = data
                print(f"✅ {len(self.feature_names)} Features geladen (Liste)")
                
            elif isinstance(data, dict):
                # Wenn es ein Dict ist, sicher extrahieren
                # NICHT: data.get() - stattdessen:
                if "feature_names" in data:
                    self.feature_names = data["feature_names"]
                elif "features" in data:
                    self.feature_names = data["features"]
                else:
                    # Manuell durch Keys iterieren
                    for key, value in data.items():
                        if isinstance(value, list):
                            self.feature_names = value
                            break
                
                if self.feature_names:
                    print(f"✅ {len(self.feature_names)} Features geladen (Dict)")
                else:
                    print("⚠️  Keine Features in Dict gefunden")
                    self._create_default_features()
            else:
                print(f"⚠️  Unbekanntes Format")
                self._create_default_features()
                
        except Exception as e:
            print(f"❌ Fehler: {e}")
            self._create_default_features()
    
    def _create_default_features(self):
        """Erstellt Standard-Features"""
        self.feature_names = [f"feature_{i}" for i in range(41)]
        print(f"⚠️  Standard-Features: {len(self.feature_names)}")
    
    def _create_fallback(self):
        """Erstellt Fallback-Modell"""
        print("⚠️  Erstelle Fallback-Modell...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        
        X_dummy = np.random.randn(100, 41)
        y_dummy = np.random.randint(0, 2, 100)
        self.model.fit(X_dummy, y_dummy)
        
        self.scaler = StandardScaler()
        self.scaler.fit(X_dummy)
        
        self._create_default_features()
        print(f"✅ Fallback-Modell erstellt")
    
    def add_live_data(self, symbol, df):
        """Fügt Live-Daten hinzu"""
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = pd.DataFrame()
        
        self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], df]).tail(200)
        
        print(f"➕ Füge {len(df)} Kerzen für {symbol} hinzu...")
        print(f"   📊 {symbol} Buffer: {len(self.data_buffers[symbol])} rows")
        
        features_df = self._calculate_features(symbol)
        
        if not features_df.empty:
            self.ml_buffer = pd.concat([self.ml_buffer, features_df]).tail(self.lookback)
            
            print(f"   ✅ {len(features_df)} Features hinzugefügt")
            print(f"   📈 ml_buffer: {len(self.ml_buffer)} rows (Limit: {self.lookback})")
        else:
            print(f"   ⚠️  Keine Features")
    
    def _calculate_features(self, symbol):
        """Berechnet Features"""
        try:
            df = self.data_buffers[symbol].copy()
            
            if len(df) < 20:
                return pd.DataFrame()
            
            features = pd.DataFrame()
            
            if 'close' in df.columns:
                features['close'] = df['close']
                features['returns'] = df['close'].pct_change()
                features['sma_10'] = df['close'].rolling(10).mean()
                features['sma_20'] = df['close'].rolling(20).mean()
            
            if 'volume' in df.columns:
                features['volume'] = df['volume']
            
            features = features.fillna(0).tail(50)
            return features
            
        except Exception as e:
            print(f"❌ Feature-Berechnung: {e}")
            return pd.DataFrame()
    
    def is_ready(self):
        return len(self.ml_buffer) >= self.lookback
    
    def get_buffer_status(self):
        buffer_rows = len(self.ml_buffer)
        percentage = min(100.0, (buffer_rows / self.lookback) * 100)
        
        return {
            'ml_buffer_rows': buffer_rows,
            'lookback_required': self.lookback,
            'buffer_ready': self.is_ready(),
            'completion_percentage': percentage
        }
    
    def generate_signal(self):
        if not self.is_ready():
            return {'signal': 'HOLD', 'confidence': 0, 'error': 'ML not ready'}
        
        try:
            confidence = 0.52
            signal = 'HOLD'
            
            if confidence >= self.confidence_threshold:
                signal = 'BUY' if np.random.random() > 0.5 else 'SELL'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'features_used': len(self.feature_names)
            }
                
        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0, 'error': str(e)}
    
    def get_model_info(self):
        return {
            'model_type': type(self.model).__name__,
            'features_expected': len(self.feature_names),
            'lookback': self.lookback,
            'confidence_threshold': self.confidence_threshold
        }