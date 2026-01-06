"""
ML INTEGRATION v4.2 - MIT VERBESSERTEM TRAINING (10.000 BARS)
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
import warnings
warnings.filterwarnings('ignore')

# Import ML Bibliotheken
try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.utils.class_weight import compute_class_weight
except ImportError as e:
    print(f"⚠️  ML-Bibliotheken fehlen: {e}")

# Import Trading Bibliotheken
try:
    import MetaTrader5 as mt5
except ImportError:
    print("⚠️  MetaTrader5 nicht verfügbar")

# Import TA-Lib
try:
    import talib
except ImportError:
    print("⚠️  TA-Lib nicht verfügbar")


def train_ml_model(symbol: str = "EURUSD", timeframe: str = "H1", 
                   bars: int = 10000, future_bars: int = 3,  # FESTER WERT: 10.000 BARS
                   test_size: float = 0.2, random_state: int = 42) -> bool:
    """
    Trainiert ein verbessertes ML-Modell für Forex Trading.
    
    Args:
        symbol: Trading Symbol (z.B. "EURUSD")
        timeframe: Zeitrahmen (z.B. "H1")
        bars: Anzahl der historischen Bars (FEST: 10000)
        future_bars: Anzahl der Bars in der Zukunft für Labeling
        test_size: Anteil der Testdaten
        random_state: Random Seed für Reproduzierbarkeit
        
    Returns:
        bool: True wenn Training erfolgreich
    """
    print(f"\n{'='*80}")
    print(f"🚀 STARTE VERBESSERTES ML-MODELL TRAINING")
    print(f"{'='*80}")
    print(f"📊 Symbol: {symbol}")
    print(f"⏱️  Timeframe: {timeframe}")
    print(f"📈 Bars: {bars}")  # JETZT KORREKT: 10000
    print(f"🔮 Future Bars für Labels: {future_bars}")
    print(f"{'='*80}\n")
    
    try:
        # 1. Hole historische Daten
        print("📥 Hole historische Daten von MT5...")
        df = get_historical_data(symbol, timeframe, bars)
        
        if df is None or len(df) < 100:
            print("❌ Nicht genügend Daten verfügbar")
            return False
            
        print(f"✅ {len(df)} Bars geladen ({df.index[0]} bis {df.index[-1]})")
        
        # 2. Berechne Features
        print("\n🔧 Berechne technische Features...")
        features_df = calculate_features(df)
        
        if features_df is None or features_df.empty:
            print("❌ Features konnten nicht berechnet werden")
            return False
            
        print(f"✅ {features_df.shape[1]} Features berechnet")
        
        # 3. Erstelle Labels (klare Trading-Signale)
        print("\n🏷️  Erstelle Labels für Training...")
        labels = create_labels(df, future_bars=future_bars)
        
        if labels is None or len(labels) == 0:
            print("❌ Labels konnten nicht erstellt werden")
            return False
            
        # Balance Check
        buy_signals = sum(labels == 1)
        sell_signals = sum(labels == 0)
        print(f"📊 Label-Verteilung: BUY={buy_signals}, SELL={sell_signals}")
        
        # 4. Daten vorbereiten
        print("\n⚙️  Bereite Daten für Training vor...")
        X, y, feature_columns = prepare_data(features_df, labels)
        
        if X is None or len(X) == 0:
            print("❌ Trainingsdaten sind leer")
            return False
            
        print(f"📐 Trainingsdaten Shape: {X.shape}")
        
        # 5. Train-Test Split
        print("\n✂️  Teile Daten in Train/Test Sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"📚 Train Set: {X_train.shape[0]} Samples")
        print(f"📝 Test Set: {X_test.shape[0]} Samples")
        
        # 6. Feature Scaling
        print("\n⚖️  Skaliere Features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 7. Berechne Class Weights für besseres Balancing
        print("\n⚖️  Berechne Class Weights...")
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
        print(f"📊 Class Weights: {class_weight_dict}")
        
        # 8. Trainiere verbessertes Random Forest Modell
        print("\n🤖 Trainiere verbessertes Random Forest Modell...")
        
        # Hyperparameter für bessere Performance
        model = RandomForestClassifier(
            n_estimators=200,           # Mehr Bäume für bessere Generalisierung
            max_depth=15,               # Tiefere Bäume für komplexe Muster
            min_samples_split=10,       # Verhindert Overfitting
            min_samples_leaf=5,         # Stabilere Vorhersagen
            max_features='sqrt',        # Optimale Feature-Auswahl
            class_weight=class_weight_dict,  # Berücksichtigt Klassen-Ungleichgewicht
            random_state=random_state,
            n_jobs=-1,                  # Nutzt alle CPU-Kerne
            bootstrap=True,
            oob_score=True              # Out-of-Bag Score für bessere Schätzung
        )
        
        # Training mit Progress
        print("⏳ Training läuft...")
        start_time = time.time()
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        print(f"✅ Training abgeschlossen in {training_time:.1f} Sekunden")
        
        # 9. Model Evaluation
        print("\n📊 EVALUATION DES MODELLS:")
        
        # Train Accuracy
        y_train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"🎯 Train Accuracy: {train_accuracy:.2%}")
        
        # Test Accuracy
        y_test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"🎯 Test Accuracy: {test_accuracy:.2%}")
        
        # AUC-ROC Score
        y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc_score = roc_auc_score(y_test, y_test_proba)
        print(f"📈 AUC-ROC Score: {auc_score:.2%}")
        
        # Classification Report
        print("\n📋 KLASSIFIKATIONS-REPORT:")
        print(classification_report(y_test, y_test_pred, target_names=['SELL', 'BUY']))
        
        # Confidence Analysis
        print("\n🎯 CONFIDENCE ANALYSE:")
        confidences = np.max(model.predict_proba(X_test_scaled), axis=1)
        print(f"📊 Durchschnittliche Confidence: {confidences.mean():.2%}")
        print(f"📈 Maximale Confidence: {confidences.max():.2%}")
        print(f"📉 Minimale Confidence: {confidences.min():.2%}")
        print(f"📊 Confidence > 70%: {(confidences > 0.7).sum() / len(confidences):.2%}")
        print(f"📊 Confidence > 80%: {(confidences > 0.8).sum() / len(confidences):.2%}")
        
        # 10. Feature Importance
        print("\n🔍 FEATURE IMPORTANCE (Top 10):")
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(10).to_string())
        
        # 11. Speichere Modell und Metadaten
        print("\n💾 Speichere Modell und Metadaten...")
        
        # Erstelle Verzeichnisse
        os.makedirs('data/ml_models', exist_ok=True)
        
        # Speichere Modell
        model_path = 'data/ml_models/forex_signal_model.pkl'
        joblib.dump(model, model_path)
        print(f"✅ Modell gespeichert: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")
        
        # Speichere Scaler
        scaler_path = 'data/ml_models/forex_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        print(f"✅ Scaler gespeichert: {scaler_path}")
        
        # Speichere Feature Columns
        features_path = 'data/ml_models/feature_columns.pkl'
        joblib.dump(feature_columns, features_path)
        print(f"✅ Feature Columns gespeichert: {features_path}")
        
        # 12. Erstelle Metadaten
        metadata = {
            'training_date': datetime.now().isoformat(),
            'symbol': symbol,
            'timeframe': timeframe,
            'bars_used': bars,
            'future_bars': future_bars,
            'accuracy': float(test_accuracy),
            'auc': float(auc_score),
            'train_samples': int(X_train.shape[0]),
            'test_samples': int(X_test.shape[0]),
            'features': int(X.shape[1]),
            'buy_signals': int(buy_signals),
            'sell_signals': int(sell_signals),
            'total_signals': int(len(labels)),
            'model_type': 'enhanced_random_forest',
            'parameters': {
                'n_estimators': model.n_estimators,
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'class_weight': str(class_weight_dict)
            },
            'performance': {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'auc_roc': float(auc_score),
                'training_time_seconds': float(training_time)
            }
        }
        
        metadata_path = 'data/ml_models/model_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Metadaten gespeichert: {metadata_path}")
        
        # 13. Finale Bewertung
        print(f"\n{'='*80}")
        print(f"🏆 TRAINING ABGESCHLOSSEN - MODELL BEWERTUNG")
        print(f"{'='*80}")
        
        if test_accuracy > 0.65:
            rating = "🔥 AUSGEZEICHNET"
            color = "🟢"
        elif test_accuracy > 0.60:
            rating = "✅ GUT"
            color = "🟡"
        elif test_accuracy > 0.55:
            rating = "⚠️  AKZEPTABEL"
            color = "🟠"
        else:
            rating = "❌ VERBESSERUNGSWÜRDIG"
            color = "🔴"
            
        print(f"{color} Modell-Genauigkeit: {test_accuracy:.2%} - {rating}")
        print(f"📈 AUC-ROC Score: {auc_score:.2%}")
        print(f"⏱️  Training Time: {training_time:.1f} Sekunden")
        print(f"📊 Features: {X.shape[1]}")
        print(f"📈 Trainings-Samples: {X_train.shape[0]}")
        print(f"{'='*80}")
        
        # Empfehlungen
        print("\n💡 EMPFEHLUNGEN:")
        if test_accuracy < 0.6:
            print("   1. Mehr Daten sammeln (>20,000 Bars)")
            print("   2. Zusätzliche Features hinzufügen")
            print("   3. Anderen Algorithmus ausprobieren (z.B. Gradient Boosting)")
            print("   4. Hyperparameter Tuning durchführen")
            
        print(f"\n✅ Verbessertes ML-Modell erfolgreich trainiert!")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler beim Training: {e}")
        traceback.print_exc()
        return False


def get_historical_data(symbol: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
    """Holt historische Daten von MT5."""
    try:
        if 'mt5' not in globals():
            print("⚠️  MT5 nicht verfügbar - verwende simulierte Daten")
            return generate_simulated_data(bars)
            
        # MT5 Timeframe Mapping
        timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1,
            'W1': mt5.TIMEFRAME_W1,
            'MN1': mt5.TIMEFRAME_MN1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Initialisiere MT5
        if not mt5.initialize():
            print("❌ MT5 konnte nicht initialisiert werden")
            return generate_simulated_data(bars)
            
        # Hole Daten
        rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, bars)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            print("⚠️  Keine Daten von MT5 - verwende simulierte Daten")
            return generate_simulated_data(bars)
            
        # Konvertiere zu DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Benenne Spalten um
        df.columns = ['open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
        
        return df
        
    except Exception as e:
        print(f"❌ Fehler beim Datenabruf: {e}")
        return generate_simulated_data(bars)


def generate_simulated_data(bars: int) -> pd.DataFrame:
    """Generiert simulierte Forex Daten."""
    try:
        np.random.seed(42)
        
        # Simuliere EURUSD Preisverlauf
        base_price = 1.10000
        prices = []
        
        for i in range(bars):
            # Random Walk mit Drift
            drift = 0.000001  # Leichte Aufwärts-Tendenz
            volatility = 0.0005
            
            if i == 0:
                prices.append(base_price)
            else:
                change = drift + np.random.normal(0, volatility)
                new_price = prices[-1] + change
                prices.append(new_price)
        
        # Erstelle OHLCV Daten
        dates = pd.date_range(end=datetime.now(), periods=bars, freq='H')
        df = pd.DataFrame({
            'open': np.array(prices) * 0.999 + np.random.normal(0, 0.0001, bars),
            'high': np.array(prices) * 1.001 + np.random.uniform(0, 0.0002, bars),
            'low': np.array(prices) * 0.999 - np.random.uniform(0, 0.0002, bars),
            'close': prices,
            'tick_volume': np.random.randint(100, 1000, bars),
            'spread': np.random.randint(1, 5, bars),
            'real_volume': np.random.randint(1000, 10000, bars)
        }, index=dates)
        
        print(f"⚠️  Verwende {bars} simulierte Bars")
        return df
        
    except Exception as e:
        print(f"❌ Fehler bei simulierten Daten: {e}")
        return None


def calculate_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Berechnet technische Features."""
    try:
        # Preise extrahieren
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['tick_volume'].values
        
        features = {}
        
        # 1. Momentum Features
        for period in [1, 2, 3, 5, 10, 20]:
            if len(close) > period:
                returns = (close[period:] - close[:-period]) / close[:-period]
                # Align with original length
                padded = np.zeros(len(close))
                padded[period:] = returns
                features[f'return_{period}'] = padded
        
        # 2. RSI
        if 'talib' in globals():
            features['RSI_14'] = talib.RSI(close, timeperiod=14)
        else:
            # Simple RSI calculation
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.convolve(gain, np.ones(14)/14, mode='valid')
            avg_loss = np.convolve(loss, np.ones(14)/14, mode='valid')
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            # Pad to match length
            padded_rsi = np.zeros(len(close))
            padded_rsi[14:] = rsi[:len(rsi)]
            features['RSI_14'] = padded_rsi
        
        # 3. MACD
        if 'talib' in globals():
            macd, macd_signal, macd_hist = talib.MACD(close)
            features['MACD'] = macd
            features['MACD_signal'] = macd_signal
            features['MACD_hist'] = macd_hist
        
        # 4. Moving Averages
        for period in [5, 10, 20, 50, 100]:
            if len(close) > period:
                sma = np.convolve(close, np.ones(period)/period, mode='valid')
                padded = np.zeros(len(close))
                padded[period-1:] = sma
                features[f'SMA_{period}'] = padded
                
                # Price relative to SMA
                if period > 1:
                    features[f'price_sma_{period}_ratio'] = close / (padded + 1e-10)
        
        # 5. Bollinger Bands
        if len(close) > 20:
            sma_20 = features.get('SMA_20', np.zeros(len(close)))
            std_20 = np.zeros(len(close))
            for i in range(19, len(close)):
                std_20[i] = np.std(close[i-19:i+1])
            
            bb_upper = sma_20 + 2 * std_20
            bb_lower = sma_20 - 2 * std_20
            
            features['bb_upper'] = bb_upper
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / (sma_20 + 1e-10)
            features['bb_position'] = (close - bb_lower) / ((bb_upper - bb_lower) + 1e-10)
        
        # 6. ATR (Average True Range)
        if len(high) > 14:
            tr = np.zeros(len(high))
            for i in range(1, len(high)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            atr = np.zeros(len(high))
            atr[14] = np.mean(tr[1:15])
            for i in range(15, len(high)):
                atr[i] = (atr[i-1] * 13 + tr[i]) / 14
            
            features['ATR'] = atr
            features['ATR_norm'] = atr / (close + 1e-10)
        
        # 7. Volume Features
        if len(volume) > 0:
            features['volume'] = volume
            if len(volume) > 10:
                sma_volume = np.convolve(volume, np.ones(10)/10, mode='valid')
                padded_volume_sma = np.zeros(len(volume))
                padded_volume_sma[9:] = sma_volume
                features['volume_sma_10'] = padded_volume_sma
                features['volume_ratio'] = volume / (padded_volume_sma + 1e-10)
        
        # 8. Price Patterns
        features['high_low_spread'] = (high - low) / (close + 1e-10)
        features['close_open_spread'] = (close - df['open'].values) / (df['open'].values + 1e-10)
        
        # 9. Candle Features
        body = abs(close - df['open'].values)
        upper_shadow = high - np.maximum(close, df['open'].values)
        lower_shadow = np.minimum(close, df['open'].values) - low
        total_range = high - low
        
        features['body_ratio'] = body / (total_range + 1e-10)
        features['upper_shadow_ratio'] = upper_shadow / (total_range + 1e-10)
        features['lower_shadow_ratio'] = lower_shadow / (total_range + 1e-10)
        
        # 10. Trend Features
        if len(close) > 50:
            sma_20 = features.get('SMA_20', np.zeros(len(close)))
            sma_50 = features.get('SMA_50', np.zeros(len(close)))
            features['sma_20_50_diff'] = sma_20 - sma_50
            features['sma_20_50_ratio'] = sma_20 / (sma_50 + 1e-10)
        
        # Konvertiere zu DataFrame
        features_df = pd.DataFrame(features)
        
        # Fülle NaN Werte
        features_df = features_df.fillna(0)
        
        # Entferne die ersten Zeilen (wegen Lag)
        min_period = max([20, 50, 100])  # Größte verwendete Periode
        if len(features_df) > min_period:
            features_df = features_df.iloc[min_period:].reset_index(drop=True)
        
        return features_df
        
    except Exception as e:
        print(f"❌ Fehler bei Feature-Berechnung: {e}")
        traceback.print_exc()
        return None


def create_labels(df: pd.DataFrame, future_bars: int = 3, threshold: float = 0.001) -> np.ndarray:
    """Erstellt Labels für das Training (1=BUY, 0=SELL)."""
    try:
        close = df['close'].values
        
        # Berechne zukünftige Returns
        future_returns = np.zeros(len(close))
        for i in range(len(close) - future_bars):
            future_return = (close[i + future_bars] - close[i]) / close[i]
            future_returns[i] = future_return
        
        # Erstelle Labels basierend auf Returns
        labels = np.zeros(len(close))
        
        # BUY wenn starker Anstieg erwartet
        buy_condition = future_returns > threshold
        labels[buy_condition] = 1
        
        # SELL wenn starker Abfall erwartet
        sell_condition = future_returns < -threshold
        labels[sell_condition] = 0
        
        # Neutral (kein klarer Trend) - entferne diese Samples
        neutral_condition = (future_returns >= -threshold) & (future_returns <= threshold)
        labels[neutral_condition] = -1  # Markiere zum Entfernen
        
        # Entferne neutrale Samples
        clear_signals = labels != -1
        filtered_labels = labels[clear_signals]
        
        # Debug Info
        total_samples = len(close)
        clear_samples = len(filtered_labels)
        buy_samples = sum(filtered_labels == 1)
        sell_samples = sum(filtered_labels == 0)
        
        print(f"📊 Labels erstellt: {clear_samples}/{total_samples} klare Signale")
        print(f"   BUY: {buy_samples} ({buy_samples/clear_samples:.1%})")
        print(f"   SELL: {sell_samples} ({sell_samples/clear_samples:.1%})")
        
        return filtered_labels.astype(int)
        
    except Exception as e:
        print(f"❌ Fehler bei Label-Erstellung: {e}")
        traceback.print_exc()
        return np.array([])


def prepare_data(features_df: pd.DataFrame, labels: np.ndarray) -> Tuple:
    """Bereitet Daten für das Training vor."""
    try:
        # Synchronisiere Features und Labels
        min_len = min(len(features_df), len(labels))
        X = features_df.iloc[:min_len].values
        y = labels[:min_len]
        
        # Entferne Samples mit NaN oder Inf
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Feature Columns
        feature_columns = features_df.columns.tolist()
        
        print(f"📐 Finale Daten: {X.shape[0]} Samples, {X.shape[1]} Features")
        
        return X, y, feature_columns
        
    except Exception as e:
        print(f"❌ Fehler bei Datenvorbereitung: {e}")
        return None, None, None


class MLTradingEngine:
    """ML Trading Engine für Paper Trading."""
    
    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.load_model()
        
    def load_model(self):
        """Lädt das ML-Modell."""
        try:
            model_path = 'data/ml_models/forex_signal_model.pkl'
            scaler_path = 'data/ml_models/forex_scaler.pkl'
            features_path = 'data/ml_models/feature_columns.pkl'
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✅ ML-Modell geladen")
                
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ Scaler geladen")
                
            if os.path.exists(features_path):
                self.feature_columns = joblib.load(features_path)
                print(f"✅ {len(self.feature_columns)} Features geladen")
                
        except Exception as e:
            print(f"❌ Fehler beim Laden des ML-Modells: {e}")
            
    def generate_signal(self, features=None):
        """Generiert ein Trading-Signal."""
        if self.model is None:
            return "HOLD", 50.0
            
        try:
            # Wenn keine Features gegeben, erstelle Dummy-Features
            if features is None:
                features = self.create_dummy_features()
                
            # Skaliere Features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
                
            # Mache Vorhersage
            prediction = self.model.predict(features_scaled)
            probabilities = self.model.predict_proba(features_scaled)
            
            # Bestimme Signal
            signal = "BUY" if prediction[0] == 1 else "SELL"
            confidence = float(probabilities[0].max() * 100)
            
            return signal, confidence
            
        except Exception as e:
            print(f"❌ Fehler bei Signal-Generierung: {e}")
            return "HOLD", 50.0
            
    def create_dummy_features(self):
        """Erstellt Dummy-Features für das Modell."""
        import pandas as pd
        
        # Erstelle Features basierend auf Feature-Columns
        if self.feature_columns:
            features = {col: [0.0] for col in self.feature_columns}
        else:
            # Standard Features
            features = {
                'RSI_14': [50.0],
                'MACD': [0.0],
                'MACD_signal': [0.0],
                'MACD_hist': [0.0],
                'SMA_20': [1.10000],
                'SMA_50': [1.10000],
                'bb_width': [0.04],
                'bb_position': [0.5],
                'ATR': [0.001],
                'ATR_norm': [0.001],
                'return_1': [0.0],
                'return_5': [0.0],
                'return_10': [0.0],
                'volume_ratio': [1.0]
            }
            
        return pd.DataFrame(features)


if __name__ == "__main__":
    # Test: Trainiere ein neues Modell
    print("🧪 Teste ML-Modell Training...")
    success = train_ml_model(
        symbol="EURUSD",
        timeframe="H1",
        bars=10000,  # FEST: 10.000 BARS
        future_bars=3,
        test_size=0.2
    )
    
    if success:
        print("✅ ML-Modell erfolgreich trainiert!")
    else:
        print("❌ ML-Modell Training fehlgeschlagen")