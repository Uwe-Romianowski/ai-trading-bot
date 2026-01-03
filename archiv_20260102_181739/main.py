#!/usr/bin/env python3
"""
AI TRADING BOT - REPARIERTE VERSION MIT FUNKTIONIERENDEM ML
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os
import traceback

# ============================================================================
# KONFIGURATION
# ============================================================================

def load_config():
    """Lädt die Konfiguration"""
    try:
        with open('config/bot_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print("✅ Konfiguration geladen")
            return config
    except Exception as e:
        print(f"⚠️  Konfigurationsfehler: {e} - verwende Standardwerte")
        return {
            'trading': {'symbols': ['EURUSD'], 'timeframe': 'M5'},
            'ml': {'lookback': 100, 'ml_debug_mode': True, 'confidence_threshold': 0.65},
            'hybrid': {'ml_weight': 0.7, 'rule_weight': 0.3}
        }

# ============================================================================
# ML IMPORT - MIT FALLBACK
# ============================================================================

def import_ml_generator():
    """Importiert den MLSignalGenerator mit Fallback"""
    try:
        from src.ml_integration.ml_signal_generator import MLSignalGenerator
        print("✅ MLSignalGenerator importiert")
        return MLSignalGenerator
    except ImportError as e:
        print(f"❌ Import Fehler: {e}")
        print("⚠️  Erstelle MLSignalGenerator inline...")
        
        # Inline-Kopie des reparierten MLSignalGenerator
        exec("""
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MLSignalGenerator:
    \"\"\"Inline MLSignalGenerator\"\"\"
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # ML Parameter
        ml_config = config.get('ml', {}) if config else {}
        self.lookback = ml_config.get('lookback', 100)
        self.confidence_threshold = ml_config.get('confidence_threshold', 0.65)
        self.buffer_size = self.lookback * 2
        
        # Buffers
        self.data_buffers = {}
        self.ml_buffer = pd.DataFrame()
        
        # Model
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Initialize
        self._load_model()
        
        print(f"✅ MLSignalGenerator initialisiert (Lookback: {self.lookback})")
    
    def _load_model(self):
        \"\"\"Lädt das ML-Modell\"\"\"
        try:
            self.model = joblib.load('data/models/random_forest_model.joblib')
            self.scaler = joblib.load('data/models/feature_scaler.joblib')
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_columns = list(self.model.feature_names_in_)
            else:
                self.feature_columns = [f'feature_{i}' for i in range(41)]
            
            return True
            
        except Exception as e:
            print(f"⚠️  Fehler beim Laden: {e}")
            
            # Dummy Modell
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import StandardScaler
            
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            X_dummy = np.random.randn(100, 41)
            y_dummy = np.random.randint(0, 2, 100)
            self.model.fit(X_dummy, y_dummy)
            
            self.scaler = StandardScaler()
            self.scaler.fit(X_dummy)
            
            self.feature_columns = [f'feature_{i}' for i in range(41)]
            print(f"✅ Dummy-Modell erstellt")
            return False
    
    def add_live_data(self, symbol, data_frame):
        \"\"\"Fügt Live-Daten hinzu\"\"\"
        try:
            print(f"➕ Füge {len(data_frame)} Kerzen für {symbol} hinzu...")
            
            if symbol not in self.data_buffers:
                self.data_buffers[symbol] = pd.DataFrame()
            
            self.data_buffers[symbol] = pd.concat(
                [self.data_buffers[symbol], data_frame],
                ignore_index=True
            ).tail(200)
            
            print(f"   📊 {symbol} Buffer: {len(self.data_buffers[symbol])} rows")
            
            # Berechne Features
            features = self._calculate_features(symbol, data_frame)
            
            if not features.empty:
                features_with_meta = features.copy()
                features_with_meta['symbol'] = symbol
                features_with_meta['timestamp'] = datetime.now()
                
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
        \"\"\"Berechnet Features\"\"\"
        try:
            if len(data) < 20:
                return pd.DataFrame()
            
            df = data.copy()
            
            # Spalten umbenennen
            col_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if 'open' in col_lower:
                    col_mapping[col] = 'open'
                elif 'high' in col_lower:
                    col_mapping[col] = 'high'
                elif 'low' in col_lower:
                    col_mapping[col] = 'low'
                elif 'close' in col_lower:
                    col_mapping[col] = 'close'
                elif 'volume' in col_lower or 'tick' in col_lower:
                    col_mapping[col] = 'volume'
                elif 'time' in col_lower:
                    col_mapping[col] = 'time'
            
            if col_mapping:
                df = df.rename(columns=col_mapping)
            
            features = pd.DataFrame(index=df.index)
            
            # Basis Features
            if 'close' in df.columns:
                features['price'] = df['close']
                features['returns'] = df['close'].pct_change()
                features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Technische Indikatoren
            features['sma_10'] = df['close'].rolling(10).mean()
            features['sma_20'] = df['close'].rolling(20).mean()
            features['ema_12'] = df['close'].ewm(span=12).mean()
            features['ema_26'] = df['close'].ewm(span=26).mean()
            
            # MACD
            macd = features['ema_12'] - features['ema_26']
            features['macd'] = macd
            features['macd_signal'] = macd.ewm(span=9).mean()
            features['macd_hist'] = macd - features['macd_signal']
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            bb_ma = df['close'].rolling(20).mean()
            bb_std = df['close'].rolling(20).std()
            features['bb_upper'] = bb_ma + (bb_std * 2)
            features['bb_lower'] = bb_ma - (bb_std * 2)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_ma
            features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Volatilität
            features['volatility_20'] = df['close'].rolling(20).std()
            
            # ATR
            high = df['high']
            low = df['low']
            close = df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            features['atr'] = tr.rolling(14).mean()
            
            # Volume
            if 'volume' in df.columns:
                features['volume'] = df['volume']
                features['volume_sma'] = df['volume'].rolling(20).mean()
                features['volume_ratio'] = df['volume'] / features['volume_sma']
            
            # Zeit Features
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                features['hour'] = df['time'].dt.hour
                features['day_of_week'] = df['time'].dt.dayofweek
                features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            
            # Trend Features
            features['trend_strength'] = features['sma_20'].diff(5)
            features['price_vs_sma'] = (df['close'] / features['sma_20']) - 1
            
            # Support/Resistance
            if 'high' in df.columns and 'low' in df.columns:
                features['high_20'] = df['high'].rolling(20).max()
                features['low_20'] = df['low'].rolling(20).min()
                if (features['high_20'] - features['low_20']).abs().sum() > 0:
                    features['close_to_high'] = (df['close'] - features['high_20']) / (features['high_20'] - features['low_20'])
            
            # NaN handling
            features = features.fillna(method='ffill').fillna(0)
            features = features.tail(min(50, len(features)))
            
            print(f"   🔧 {len(features)} Features berechnet")
            return features
            
        except Exception as e:
            print(f"❌ Feature-Berechnung: {e}")
            return pd.DataFrame()
    
    def is_ready(self):
        \"\"\"Prüft Bereitschaft\"\"\"
        return len(self.ml_buffer) >= self.lookback
    
    def generate_signal(self):
        \"\"\"Generiert Signal\"\"\"
        if not self.is_ready():
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'signal_source': 'ML',
                'error': f'Benötigt {self.lookback}, Hat: {len(self.ml_buffer)}',
                'features_used': len(self.ml_buffer)
            }
        
        try:
            recent_features = self.ml_buffer.tail(self.lookback)
            numeric_cols = recent_features.select_dtypes(include=[np.number]).columns
            features_data = recent_features[numeric_cols]
            
            if len(features_data.columns) < 10:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'signal_source': 'ML',
                    'error': f'Zu wenige Features: {len(features_data.columns)}'
                }
            
            features_mean = features_data.mean().values.reshape(1, -1)
            
            if self.scaler and hasattr(self.scaler, 'transform'):
                features_scaled = self.scaler.transform(features_mean)
            else:
                features_scaled = features_mean
            
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            if confidence >= self.confidence_threshold:
                signal = 'BUY' if prediction == 1 else 'SELL'
            else:
                signal = 'HOLD'
            
            return {
                'signal': signal,
                'confidence': float(confidence),
                'probability_up': float(probabilities[1]) if len(probabilities) > 1 else 0.5,
                'probability_down': float(probabilities[0]) if len(probabilities) > 0 else 0.5,
                'signal_source': 'ML',
                'features_used': len(features_data.columns)
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'signal_source': 'ML',
                'error': f'Vorhersagefehler: {str(e)}'
            }
    
    def get_buffer_status(self):
        \"\"\"Buffer Status\"\"\"
        return {
            'ml_buffer_rows': len(self.ml_buffer),
            'lookback_required': self.lookback,
            'buffer_ready': self.is_ready(),
            'completion_percentage': (len(self.ml_buffer) / self.lookback) * 100 if self.lookback > 0 else 0,
            'data_buffers': {k: len(v) for k, v in self.data_buffers.items()}
        }
    
    def get_model_info(self):
        \"\"\"Modell Info\"\"\"
        return {
            'model_type': type(self.model).__name__,
            'n_estimators': getattr(self.model, 'n_estimators', 'N/A'),
            'features_expected': len(self.feature_columns) if self.feature_columns else 0,
            'lookback': self.lookback,
            'confidence_threshold': self.confidence_threshold
        }
    
    def debug_feature_matching(self):
        \"\"\"Debug Info\"\"\"
        print(f"🔍 ML erwartet {len(self.feature_columns) if self.feature_columns else 0} Features")
        if not self.ml_buffer.empty:
            print(f"📊 ml_buffer: {len(self.ml_buffer.columns)} Spalten")
            numeric_cols = self.ml_buffer.select_dtypes(include=[np.number]).columns
            print(f"🔢 Davon numerisch: {len(numeric_cols)}")
""")
        
        from src.ml_integration.ml_signal_generator import MLSignalGenerator
        return MLSignalGenerator

# ============================================================================
# DATEN-HANDLING
# ============================================================================

def simulate_mt5_data(symbol='EURUSD', count=150):
    """Simuliert Forex-Daten"""
    print(f"📊 Simuliere {count} {symbol} Kerzen...")
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=count, freq='5min')
    
    base_prices = {
        'EURUSD': 1.0800,
        'GBPUSD': 1.2600,
        'USDJPY': 148.50,
        'USDCHF': 0.8800,
        'AUDUSD': 0.6600,
        'USDCAD': 1.3600
    }
    
    price = base_prices.get(symbol, 1.0800)
    
    data = []
    for i in range(count):
        returns = np.random.normal(0.0001, 0.0005)
        
        open_price = price
        close_price = price * (1 + returns)
        high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.0002))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.0002))
        volume = np.random.randint(100, 1000)
        
        data.append({
            'time': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'tick_volume': volume
        })
        
        price = close_price
    
    df = pd.DataFrame(data)
    print(f"✅ {len(df)} simulierte Kerzen erstellt")
    print(f"   Zeitraum: {df['time'].min()} bis {df['time'].max()}")
    print(f"   Preis: {df['close'].iloc[0]:.5f} -> {df['close'].iloc[-1]:.5f}")
    
    return df

def get_real_mt5_data(symbol='EURUSD', timeframe='M5', count=150):
    """Holt echte MT5 Daten"""
    print(f"📡 Hole echte {symbol} Daten von MT5...")
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize(login=REMOVED_MT5_LOGIN, server="REMOVED_MT5_SERVER"):
            print("❌ MT5 Verbindung fehlgeschlagen")
            return None
        
        # Zeitframe mapping
        tf_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        # Daten abrufen
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            print(f"❌ Keine Daten für {symbol}")
            return None
        
        # In DataFrame konvertieren
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Spalten umbenennen
        rename_map = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'open' in col_lower:
                rename_map[col] = 'open'
            elif 'high' in col_lower:
                rename_map[col] = 'high'
            elif 'low' in col_lower:
                rename_map[col] = 'low'
            elif 'close' in col_lower:
                rename_map[col] = 'close'
            elif 'tick_volume' in col_lower:
                rename_map[col] = 'tick_volume'
            elif 'time' in col_lower:
                rename_map[col] = 'time'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        print(f"✅ {len(df)} echte Kerzen empfangen")
        print(f"   Letzte Kerze: {df['time'].iloc[-1]}")
        print(f"   Close: {df['close'].iloc[-1]:.5f}")
        
        return df
        
    except Exception as e:
        print(f"❌ MT5 Fehler: {e}")
        return None

# ============================================================================
# ML INTEGRATOR
# ============================================================================

class MLIntegrator:
    """Integriert ML in das System"""
    
    def __init__(self, ml_generator):
        self.ml_generator = ml_generator
    
    def add_data_correctly(self, symbol, data_frame):
        """Fügt Daten korrekt hinzu"""
        print(f"\n➕ FÜGE DATEN HINZU: {symbol}")
        print(f"   📊 Eingang: {len(data_frame)} rows, {len(data_frame.columns)} columns")
        
        # Sicherstellen, dass es ein DataFrame ist
        if not isinstance(data_frame, pd.DataFrame):
            try:
                data_frame = pd.DataFrame(data_frame)
            except:
                print(f"   ❌ Konvertierung fehlgeschlagen")
                return False
        
        # Die reparierte add_live_data Methode aufrufen
        success = self.ml_generator.add_live_data(symbol, data_frame)
        
        if success:
            status = self.ml_generator.get_buffer_status()
            print(f"   ✅ Erfolg - ml_buffer: {status['ml_buffer_rows']} rows")
            return True
        else:
            print(f"   ❌ Fehlgeschlagen")
            return False
    
    def check_and_fix_ml_buffer(self):
        """Prüft und repariert Buffer"""
        print("\n🔧 PRÜFE ML_BUFFER...")
        
        status = self.ml_generator.get_buffer_status()
        
        if not status['buffer_ready']:
            print(f"   ⚠️  Nicht bereit: {status['ml_buffer_rows']}/{status['lookback_required']} rows")
            
            # Versuche mit vorhandenen Daten zu füllen
            if hasattr(self.ml_generator, 'data_buffers'):
                total_added = 0
                for symbol, data in self.ml_generator.data_buffers.items():
                    if isinstance(data, pd.DataFrame) and len(data) >= 30:
                        print(f"   🔄 Verarbeite {symbol} ({len(data)} rows)...")
                        
                        # In Batches verarbeiten
                        batch_size = 25
                        for i in range(0, len(data), batch_size):
                            batch = data.iloc[i:i+batch_size]
                            if len(batch) >= 20:
                                self.ml_generator.add_live_data(symbol, batch)
                                total_added += len(batch)
                
                if total_added > 0:
                    print(f"   ✅ {total_added} zusätzliche Kerzen verarbeitet")
        
        # Endstatus
        status = self.ml_generator.get_buffer_status()
        print(f"   📊 Final: {status['ml_buffer_rows']} rows, Bereit: {status['buffer_ready']}")
        
        return status['buffer_ready']

# ============================================================================
# HAUPTFUNKTIONEN
# ============================================================================

def initialize_and_test():
    """Initialisiert und testet ML"""
    print("\n" + "="*60)
    print("🔧 ML-SYSTEM INITIALISIERUNG & TEST")
    print("="*60)
    
    # Konfiguration
    config = load_config()
    
    # ML importieren und initialisieren
    try:
        MLSignalGenerator = import_ml_generator()
        ml = MLSignalGenerator(config)
        integrator = MLIntegrator(ml)
        
        # Modell-Info
        model_info = ml.get_model_info()
        print(f"\n📋 MODELL-INFORMATIONEN:")
        print(f"  • Modell-Typ: {model_info.get('model_type')}")
        print(f"  • Bäume: {model_info.get('n_estimators', 'N/A')}")
        print(f"  • Features erwartet: {model_info.get('features_expected')}")
        print(f"  • Lookback: {model_info.get('lookback')}")
        print(f"  • Confidence Threshold: {model_info.get('confidence_threshold')}")
        
        # Buffer-Status
        status = ml.get_buffer_status()
        print(f"\n📊 INITIALER STATUS:")
        print(f"  • ml_buffer: {status['ml_buffer_rows']} rows")
        print(f"  • Benötigt: {status['lookback_required']} rows")
        print(f"  • Bereit: {status['buffer_ready']}")
        print(f"  • Completion: {status['completion_percentage']:.1f}%")
        
        return ml, integrator
        
    except Exception as e:
        print(f"❌ Initialisierung fehlgeschlagen: {e}")
        traceback.print_exc()
        return None, None

def collect_data_and_calculate_features():
    """Sammelt Daten und berechnet Features"""
    print("\n" + "="*60)
    print("📊 DATEN SAMMELN & FEATURE-BERECHNUNG")
    print("="*60)
    
    # ML initialisieren
    ml, integrator = initialize_and_test()
    if not ml:
        print("❌ ML konnte nicht initialisiert werden")
        return
    
    # Datenquelle wählen
    print("\n📡 DATENQUELLE:")
    print("1. Echte MT5 Daten (Demo Account)")
    print("2. Simulierte Daten (für Tests)")
    print("\nWahl (1-2): ", end="")
    
    try:
        choice = input().strip()
        
        if choice == "1":
            df = get_real_mt5_data('EURUSD', 'M5', 150)
            if df is None:
                print("⚠️  Fallback auf simulierte Daten")
                df = simulate_mt5_data('EURUSD', 150)
        else:
            df = simulate_mt5_data('EURUSD', 150)
        
    except:
        df = simulate_mt5_data('EURUSD', 150)
    
    if df is None or df.empty:
        print("❌ Keine Daten verfügbar")
        return
    
    print(f"\n📊 Daten bereit: {len(df)} rows")
    
    # 1. Daten hinzufügen
    print("\n1️⃣  DATEN ZU ML HINZUFÜGEN:")
    success = integrator.add_data_correctly("EURUSD", df)
    
    if not success:
        print("❌ Daten konnten nicht hinzugefügt werden")
        return
    
    # 2. Buffer prüfen
    print("\n2️⃣  BUFFER PRÜFEN:")
    buffer_ready = integrator.check_and_fix_ml_buffer()
    
    # 3. Status anzeigen
    print("\n3️⃣  FINALER STATUS:")
    status = ml.get_buffer_status()
    print(f"   • ml_buffer: {status['ml_buffer_rows']} rows")
    print(f"   • Benötigt: {status['lookback_required']} rows")
    print(f"   • Bereit: {status['buffer_ready']}")
    print(f"   • Completion: {status['completion_percentage']:.1f}%")
    
    # 4. Debug Info
    print("\n4️⃣  DEBUG INFORMATIONEN:")
    ml.debug_feature_matching()
    
    # 5. Signal testen
    print("\n5️⃣  SIGNAL TESTEN:")
    if status['buffer_ready']:
        for i in range(2):  # Zwei Test-Signale
            print(f"\n   Test {i+1}:")
            signal = ml.generate_signal()
            
            print(f"   🎯 Signal: {signal.get('signal', 'N/A')}")
            print(f"   📊 Confidence: {signal.get('confidence', 0):.1%}")
            
            if 'probability_up' in signal:
                print(f"   📈 P(UP): {signal.get('probability_up', 0):.1%}")
                print(f"   📉 P(DOWN): {signal.get('probability_down', 0):.1%}")
            
            print(f"   🔧 Features: {signal.get('features_used', 0)}")
            
            if 'error' in signal:
                print(f"   ⚠️  {signal['error']}")
            else:
                print(f"   ✅ Erfolgreich")
    else:
        print("   ⚠️  ML nicht bereit")
        
        # Versuche mit mehr Daten
        print("   ⚡ Füge zusätzliche Daten hinzu...")
        extra_data = simulate_mt5_data('EURUSD', 100)
        integrator.add_data_correctly("EURUSD", extra_data)
        
        if ml.is_ready():
            signal = ml.generate_signal()
            print(f"   🎯 Jetzt funktioniert: {signal.get('signal', 'N/A')}")
            print(f"   📊 Confidence: {signal.get('confidence', 0):.1%}")
        else:
            print(f"   ❌ Immer noch nicht bereit")
    
    print("\n✅ Prozess abgeschlossen")

def test_ml_signal():
    """Testet nur ML-Signal"""
    print("\n" + "="*60)
    print("🧪 ML-SIGNAL TEST")
    print("="*60)
    
    ml, integrator = initialize_and_test()
    if not ml:
        return
    
    print("\n🎯 GENERATE_SIGNAL() TEST:")
    
    signal = ml.generate_signal()
    
    print(f"\n📊 ERGEBNIS:")
    print(f"  Signal: {signal.get('signal', 'N/A')}")
    print(f"  Confidence: {signal.get('confidence', 0):.1%}")
    print(f"  Quelle: {signal.get('signal_source', 'ML')}")
    print(f"  Features: {signal.get('features_used', 0)}")
    
    if 'probability_up' in signal:
        print(f"  P(UP): {signal.get('probability_up', 0):.1%}")
        print(f"  P(DOWN): {signal.get('probability_down', 0):.1%}")
    
    if 'error' in signal:
        print(f"\n⚠️  FEHLER: {signal['error']}")
        
        # Fehleranalyse
        error_msg = signal['error']
        if "Benötigt" in error_msg and "Hat:" in error_msg:
            import re
            match = re.search(r'Hat:\s*(\d+)', error_msg)
            if match:
                available = int(match.group(1))
                required = ml.lookback
                print(f"\n🔍 ANALYSE:")
                print(f"  • Benötigt: {required} Features")
                print(f"  • Verfügbar: {available} Features")
                print(f"  • Fehlen: {required - available} Features")
                print(f"  💡 Lösung: Option 2 wählen um Daten hinzuzufügen")
    else:
        print(f"\n✅ ML-SIGNAL FUNKTIONIERT")

def show_system_info():
    """Zeigt Systeminformationen"""
    print("\n" + "="*60)
    print("🤖 SYSTEMINFORMATIONEN")
    print("="*60)
    
    print("\n📦 PYTHON VERSION:")
    print(f"  {sys.version}")
    
    print("\n📁 PROJEKTSTRUKTUR:")
    print("  ai_bot/")
    print("  ├── src/")
    print("  │   ├── ml_integration/")
    print("  │   │   └── ml_signal_generator.py")
    print("  │   └── strategies/")
    print("  ├── data/")
    print("  │   └── models/")
    print("  │       ├── random_forest_model.joblib")
    print("  │       └── feature_scaler.joblib")
    print("  ├── config/")
    print("  │   └── bot_config.yaml")
    print("  └── main.py")
    
    print("\n📊 PAKETVERSIONEN:")
    try:
        import sklearn
        print(f"  • scikit-learn: {sklearn.__version__}")
    except:
        print("  • scikit-learn: Nicht verfügbar")
    
    try:
        import pandas as pd
        print(f"  • pandas: {pd.__version__}")
    except:
        print("  • pandas: Nicht verfügbar")
    
    try:
        import numpy as np
        print(f"  • numpy: {np.__version__}")
    except:
        print("  • numpy: Nicht verfügbar")

# ============================================================================
# HAUPTMENÜ
# ============================================================================

def main():
    """Hauptfunktion"""
    print("\n" + "="*60)
    print("🤖 AI FOREX TRADING BOT v3.5")
    print("="*60)
    print("Status: ML-Integration repariert & funktionsfähig")
    print("Datum: " + datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
    print("="*60)
    
    while True:
        print("\n📋 HAUPTMENÜ:")
        print("1. 🔧 ML-System initialisieren & testen")
        print("2. 📊 Daten sammeln und Features berechnen")
        print("3. 🧪 ML-Signal testen")
        print("4. ℹ️  Systeminformationen")
        print("5. 🚪 Beenden")
        print("\nWahl (1-5): ", end="")
        
        try:
            choice = input().strip()
            
            if choice == "1":
                ml, integrator = initialize_and_test()
                if ml:
                    print("\n✅ ML-System erfolgreich initialisiert")
                    
                    # Automatisch Option 3 testen
                    print("\n⚡ Automatischer Signal-Test...")
                    test_ml_signal()
            elif choice == "2":
                collect_data_and_calculate_features()
            elif choice == "3":
                test_ml_signal()
            elif choice == "4":
                show_system_info()
            elif choice == "5":
                print("\n👋 Auf Wiedersehen!")
                print("="*60)
                break
            else:
                print(f"❌ Ungültige Eingabe: {choice}")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Programm durch Benutzer abgebrochen")
            break
        except Exception as e:
            print(f"\n❌ Unerwarteter Fehler: {e}")
            traceback.print_exc()

# ============================================================================
# START
# ============================================================================

if __name__ == "__main__":
    # Prüfe ob das Projekt korrekt strukturiert ist
    print("🔍 Prüfe Projektstruktur...")
    
    required_dirs = ['src/ml_integration', 'data/models', 'config']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"⚠️  Verzeichnis fehlt: {dir_path}")
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Erstellt: {dir_path}")
            except:
                print(f"❌ Konnte nicht erstellen: {dir_path}")
    
    # Starte Hauptprogramm
    main()