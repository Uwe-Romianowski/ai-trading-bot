"""
Aufgabe A.2: ML-Forschungsmodell für Forex-Signale
KORRIGIERTE VERSION für numpy allow_pickle Problem
"""

import numpy as np
import pandas as pd
import joblib
import os
import sys
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ML Bibliotheken
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score)
from sklearn.preprocessing import StandardScaler

class ForexMLModel:
    """Machine Learning Modell für Forex-Signalvorhersage"""
    
    def __init__(self, model_name: str = "random_forest"):
        self.model_name = model_name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
        print("=" * 70)
        print("🤖 FOREX ML MODELL (Random Forest Classifier)")
        print("=" * 70)
    
    def load_training_data(self, data_dir: str = "data/processed") -> Tuple[np.ndarray, np.ndarray, list]:
        """Lädt die vorbereiteten Trainingsdaten"""
        print(f"📁 Lade Trainingsdaten von {data_dir}...")
        
        try:
            # MIT allow_pickle=True
            X = np.load(os.path.join(data_dir, "X_eurusd.npy"), allow_pickle=True)
            y = np.load(os.path.join(data_dir, "y_eurusd.npy"), allow_pickle=True)
            
            # Konvertiere zu float32 falls nötig
            if X.dtype == np.object_:
                print(f"⚠️  X ist Object Array - konvertiere zu float32...")
                X = X.astype(np.float32)
            
            # Feature-Namen laden
            with open(os.path.join(data_dir, "feature_names.pkl"), 'rb') as f:
                feature_names = joblib.load(f)
            
            print(f"✅ Daten geladen:")
            print(f"   X Shape: {X.shape}, dtype: {X.dtype}")
            print(f"   y Shape: {y.shape}")
            print(f"   Features: {len(feature_names)}")
            print(f"   Positive/Negative: {np.sum(y==1)}/{np.sum(y==0)}")
            print(f"   Memory: {X.nbytes / 1024 / 1024:.1f} MB")
            
            # Für schnelleres Testen: Nur ersten Teil verwenden
            if X.shape[0] > 10000:
                print(f"\n⚠️  GROSSER DATENSATZ - verwende nur 10.000 Samples für Test")
                X = X[:10000]
                y = y[:10000]
                print(f"   Reduziert auf: X={X.shape}, y={y.shape}")
            
            return X, y, feature_names
            
        except Exception as e:
            print(f"❌ Fehler beim Laden: {e}")
            raise
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, 
                       test_size: float = 0.3) -> Tuple:
        """
        Teilt und skaliert die Daten
        WICHTIG: Zeitliche Trennung (kein Shuffling!)
        """
        print(f"\n🔧 DATENVORBEREITUNG")
        print(f"   Test Size: {test_size*100}%")
        
        # Zeitliche Trennung (die ersten 70% für Training, letzte 30% für Test)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"   Training Samples: {len(X_train):,}")
        print(f"   Test Samples: {len(X_test):,}")
        print(f"   Train Positive: {np.sum(y_train==1):,} ({np.sum(y_train==1)/len(y_train)*100:.1f}%)")
        print(f"   Test Positive: {np.sum(y_test==1):,} ({np.sum(y_test==1)/len(y_test)*100:.1f}%)")
        
        # Skalierung
        print(f"   Skaliere Features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Trainiert einen Random Forest Classifier"""
        print(f"\n🌲 TRAINIERE RANDOM FOREST")
        
        # Einfaches Training mit Standard-Parametern (für Geschwindigkeit)
        print("   Verwende Standard-Parameter...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced',
            n_jobs=-1,
            verbose=1
        )
        
        print("   Training gestartet...")
        self.model.fit(X_train, y_train)
        
        # Feature Importance
        self.feature_importance = self.model.feature_importances_
        
        print(f"✅ Training abgeschlossen")
        
        return self.model
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluierte das Modell auf Testdaten"""
        print(f"\n📊 MODELLEVALUATION")
        
        # Vorhersagen
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metriken berechnen
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Konfusionsmatrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"✅ EVALUATIONSERGEBNISSE:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\n📈 KONFUSIONSMATRIX:")
        print(f"   True Negative:  {cm[0, 0]}")
        print(f"   False Positive: {cm[0, 1]}")
        print(f"   False Negative: {cm[1, 0]}")
        print(f"   True Positive:  {cm[1, 1]}")
        
        print(f"\n📋 KLASSIFIKATIONSREPORT:")
        print(classification_report(y_test, y_pred, target_names=['FALLEND', 'STEIGEND']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_model(self, output_dir: str = "data/models"):
        """Speichert das trainierte Modell"""
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, "trained_ml_model.pkl")
        scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
        
        # Modell und Scaler speichern
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n💾 MODELL GESPEICHERT:")
        print(f"   Modell: {model_path}")
        print(f"   Scaler: {scaler_path}")
        print(f"   Modelltyp: {type(self.model).__name__}")
    
    def run_full_pipeline(self, data_dir: str = "data/processed", 
                         test_size: float = 0.3):
        """Führt die komplette ML-Pipeline aus"""
        try:
            # 1. Daten laden
            X, y, feature_names = self.load_training_data(data_dir)
            
            # 2. Daten vorbereiten
            X_train, X_test, y_train, y_test = self.preprocess_data(
                X, y, test_size=test_size
            )
            
            # 3. Modell trainieren
            self.train_random_forest(X_train, y_train)
            
            # 4. Modell evaluieren
            results = self.evaluate_model(X_test, y_test)
            
            # 5. Modell speichern
            self.save_model()
            
            print(f"\n" + "=" * 70)
            print("✅ ML-MODELL PIPELINE ABGESCHLOSSEN!")
            print("=" * 70)
            
            # Kriterien für Phase A.2
            print(f"\n🎯 KRITERIEN FÜR PHASE A.2:")
            accuracy_ok = results['accuracy'] > 0.52
            f1_ok = results['f1'] > 0.5
            roc_ok = results['roc_auc'] > 0.6
            
            print(f"   Accuracy > 52%: {'✓' if accuracy_ok else '✗'} ({results['accuracy']:.2%})")
            print(f"   F1-Score > 0.5: {'✓' if f1_ok else '✗'} ({results['f1']:.3f})")
            print(f"   ROC-AUC > 0.6:  {'✓' if roc_ok else '✗'} ({results['roc_auc']:.3f})")
            
            if accuracy_ok:
                print(f"\n✅ PHASE A.2 ERFOLGREICH!")
            else:
                print(f"\n⚠️  PHASE A.2 BEDINGT ERFOLGREICH")
            
            return results
            
        except Exception as e:
            print(f"❌ Fehler in der Pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Hauptfunktion für Aufgabe A.2"""
    print("=" * 70)
    print("🚀 PHASE A.2: ML-MODELL TRAINING")
    print("=" * 70)
    
    # ML Modell initialisieren
    ml_model = ForexMLModel()
    
    # Pipeline ausführen
    try:
        results = ml_model.run_full_pipeline(
            data_dir="data/processed",
            test_size=0.3
        )
        
        # Zusammenfassung
        print(f"\n📋 ZUSAMMENFASSUNG:")
        print(f"   Modell: Random Forest Classifier")
        print(f"   Test Samples: {len(results.get('y_pred', []))}")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Modell gespeichert in: data/models/trained_ml_model.pkl")
        
    except Exception as e:
        print(f"\n❌ PHASE A.2 FEHLGESCHLAGEN: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if training data exists
    if not os.path.exists("data/processed/X_eurusd.npy"):
        print("❌ Trainingsdaten nicht gefunden!")
        print("Bitte zuerst Phase A.1 ausführen (test_eurusd_features.py)")
        sys.exit(1)
    
    success = main()
    
    if success:
        print(f"\n📋 NÄCHSTER SCHRITT:")
        print(f"1. research_backtester.py erstellen (Phase A.3)")
        print(f"2. Historischen Backtest durchführen")
        print(f"3. Performance mit Regel-basiertem Bot vergleichen")
        sys.exit(0)
    else:
        sys.exit(1)