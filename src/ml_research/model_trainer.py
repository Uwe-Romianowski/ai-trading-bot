"""
ML Modell Trainer für Forex Trading
Trainiert Random Forest Modelle auf Forex-Daten
"""

import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

class ForexModelTrainer:
    """Trainiert ML-Modelle für Forex Trading"""
    
    def __init__(self, config_path="config/ml_config.yaml"):
        print("=" * 60)
        print("🤖 FOREX ML MODEL TRAINER")
        print("=" * 60)
        
        # Konfiguration laden
        self.config = self.load_config(config_path)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def load_config(self, config_path):
        """Lädt Konfiguration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except:
            return {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'n_jobs': -1
            }
    
    def load_and_prepare_data(self):
        """
        Lädt und bereitet Daten für Training vor
        Falls keine echten Daten vorhanden, werden Demo-Daten generiert
        """
        print("📊 Datenvorbereitung...")
        
        # Prüfe ob echte Daten existieren
        data_path = self.config.get('data_path', 'data/processed/eurusd_features.csv')
        
        if os.path.exists(data_path):
            print(f"✅ Lade Daten von: {data_path}")
            df = pd.read_csv(data_path)
            
            # Features und Labels trennen
            feature_cols = [col for col in df.columns if col not in ['target', 'datetime']]
            X = df[feature_cols].values
            y = df['target'].values
            
            self.feature_names = feature_cols
            
        else:
            print("⚠️  Keine echten Daten gefunden - generiere Demo-Daten")
            X, y, feature_names = self.generate_demo_data()
            self.feature_names = feature_names
        
        print(f"   Daten Shape: {X.shape}")
        print(f"   Positive Samples: {np.sum(y == 1)}/{len(y)} ({np.mean(y == 1):.1%})")
        
        return X, y
    
    def generate_demo_data(self, n_samples=10000, n_features=4300):
        """Generiert Demo-Daten für Training"""
        print("🧪 Generiere Demo-Daten...")
        
        # Zufällige Features (simuliert Forex-Daten)
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        
        # Simuliere etwas Muster für Targets
        # Einige Features haben Predictive Power
        informative_features = 100
        weights = np.zeros(n_features)
        weights[:informative_features] = np.random.randn(informative_features) * 0.5
        
        # Logistische Funktion für Wahrscheinlichkeiten
        logits = X @ weights + np.random.randn(n_samples) * 0.1
        probabilities = 1 / (1 + np.exp(-logits))
        
        # Targets basierend auf Wahrscheinlichkeiten
        y = (probabilities > 0.5).astype(int)
        
        # Feature-Namen generieren
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        print(f"✅ Demo-Daten generiert: {n_samples} Samples, {n_features} Features")
        
        return X, y, feature_names
    
    def train_model(self, X, y):
        """Trainiert das Random Forest Modell"""
        print("\n🎯 Modelltraining...")
        
        # Daten aufteilen
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.get('test_size', 0.2),
            random_state=self.config.get('random_state', 42),
            stratify=y
        )
        
        print(f"   Trainingsdaten: {X_train.shape}")
        print(f"   Testdaten: {X_test.shape}")
        
        # Features skalieren
        print("📏 Skaliere Features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modell initialisieren
        self.model = RandomForestClassifier(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', 10),
            min_samples_split=self.config.get('min_samples_split', 5),
            n_jobs=self.config.get('n_jobs', -1),
            random_state=self.config.get('random_state', 42),
            verbose=1
        )
        
        # Training
        print("🌲 Trainiere Random Forest...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluation
        print("\n📊 Modell-Evaluation:")
        
        # Trainings-Score
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"   Trainings-Genauigkeit: {train_accuracy:.1%}")
        
        # Test-Score
        test_pred = self.model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        print(f"   Test-Genauigkeit: {test_accuracy:.1%}")
        
        # Cross Validation
        print("   Cross-Validation...")
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        print(f"   CV Genauigkeit: {cv_scores.mean():.1%} (+/- {cv_scores.std() * 2:.1%})")
        
        # Classification Report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, test_pred, target_names=['DOWN', 'UP']))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, test_pred)
        
        return test_accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plottet die Confusion Matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['DOWN', 'UP'], 
                   yticklabels=['DOWN', 'UP'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Speichere Plot
        os.makedirs('data/plots', exist_ok=True)
        plt.savefig('data/plots/confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✅ Confusion Matrix gespeichert: data/plots/confusion_matrix.png")
    
    def save_model(self):
        """Speichert das trainierte Modell"""
        if self.model is None:
            print("❌ Kein Modell zum Speichern vorhanden")
            return
        
        # Erstelle Verzeichnisse
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        # Speichere Modell
        model_path = 'data/models/trained_ml_model.pkl'
        joblib.dump(self.model, model_path)
        print(f"✅ Modell gespeichert: {model_path}")
        
        # Speichere Scaler
        scaler_path = 'data/models/feature_scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler gespeichert: {scaler_path}")
        
        # Speichere Feature-Namen
        if self.feature_names is not None:
            features_path = 'data/processed/feature_names.pkl'
            joblib.dump(self.feature_names, features_path)
            print(f"✅ Feature-Namen gespeichert: {features_path}")
    
    def analyze_feature_importance(self, top_n=20):
        """Analysiert Feature Importance"""
        if self.model is None:
            print("❌ Kein Modell für Feature Importance Analyse")
            return
        
        if self.feature_names is None:
            print("❌ Keine Feature-Namen verfügbar")
            return
        
        print(f"\n📊 FEATURE IMPORTANCE (Top {top_n}):")
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Top N Features anzeigen
        for i in range(min(top_n, len(importances))):
            idx = indices[i]
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'
            print(f"   {i+1:2d}. {feature_name:30s}: {importances[idx]:.4f}")
        
        # Plot Feature Importance
        self.plot_feature_importance(importances, indices, top_n)
    
    def plot_feature_importance(self, importances, indices, top_n=20):
        """Plottet Feature Importance"""
        plt.figure(figsize=(10, 8))
        
        # Top N Features
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        
        # Feature-Namen (gekürzt für bessere Lesbarkeit)
        if self.feature_names is not None:
            feature_labels = []
            for idx in top_indices:
                if idx < len(self.feature_names):
                    name = self.feature_names[idx]
                    # Kürze lange Namen
                    if len(name) > 30:
                        name = name[:27] + "..."
                    feature_labels.append(name)
                else:
                    feature_labels.append(f'Feature {idx}')
        else:
            feature_labels = [f'Feature {i}' for i in range(top_n)]
        
        # Horizontal Bar Plot
        y_pos = np.arange(len(top_importances))
        plt.barh(y_pos, top_importances)
        plt.yticks(y_pos, feature_labels)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()  # Höchste Importance oben
        
        plt.tight_layout()
        
        # Speichern
        os.makedirs('data/plots', exist_ok=True)
        plt.savefig('data/plots/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Feature Importance Plot gespeichert: data/plots/feature_importance.png")

def train_demo_model():
    """Trainiert ein Demo-Modell für schnelle Tests"""
    print("=" * 60)
    print("🧪 DEMO ML-MODELL TRAINING")
    print("=" * 60)
    
    try:
        # Trainer initialisieren
        trainer = ForexModelTrainer()
        
        # Demo-Daten generieren
        X, y = trainer.load_and_prepare_data()
        
        # Modell trainieren
        accuracy = trainer.train_model(X, y)
        
        # Modell speichern
        trainer.save_model()
        
        # Feature Importance analysieren
        trainer.analyze_feature_importance(top_n=15)
        
        print("\n" + "=" * 60)
        print(f"✅ DEMO TRAINING ERFOLGREICH!")
        print(f"   Test Accuracy: {accuracy:.1%}")
        print(f"   Modell gespeichert in: data/models/")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Training fehlgeschlagen: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hauptfunktion für Modelltraining"""
    print("🤖 FOREX ML MODELL TRAINING")
    print("Wählen Sie eine Option:")
    print("  1. Vollständiges Training (mit echten Daten)")
    print("  2. Demo Training (mit generierten Daten)")
    print("  3. Nur Modell laden und analysieren")
    
    choice = input("\nAuswahl (1-3): ")
    
    if choice == "1":
        # Vollständiges Training
        trainer = ForexModelTrainer()
        X, y = trainer.load_and_prepare_data()
        trainer.train_model(X, y)
        trainer.save_model()
        trainer.analyze_feature_importance()
        
    elif choice == "2":
        # Demo Training
        success = train_demo_model()
        if success:
            print("\n🎉 Demo-Modell ist jetzt für Live-Trading bereit!")
            print("   Sie können Option 1 im Hauptmenü für Live-Trading verwenden")
        
    elif choice == "3":
        # Modell laden und analysieren
        try:
            model = joblib.load('data/models/trained_ml_model.pkl')
            print(f"✅ Modell geladen: {type(model).__name__}")
            print(f"   Features: {model.n_features_in_}")
            print(f"   Trees: {model.n_estimators}")
        except Exception as e:
            print(f"❌ Konnte Modell nicht laden: {e}")
    
    else:
        print("❌ Ungültige Auswahl")

if __name__ == "__main__":
    main()