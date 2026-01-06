# 🤖 AI Trading Bot v4.2

Optimierter Forex Trading Bot mit Machine Learning und Paper Trading.

## 🚀 Features
- **ML-Signal-Generierung** mit Random Forest (61% Accuracy)
- **Echtzeit-Daten** von MetaTrader 5
- **Paper Trading** ohne Risiko
- **Risiko-Management** mit SL/TP
- **Performance-Tracking** mit Dashboard

## 📦 Installation
```bash
# 1. Clone repository
git clone <repository-url>
cd ai_bot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train ML model (optional - Modell ist bereits enthalten)
python -c "from src.paper_trading.ml_integration import train_ml_model; train_ml_model()"

# 4. Start bot
python main.py
📁 Projekt-Struktur
text
ai_bot/
├── main.py                    # Hauptprogramm
├── requirements.txt           # Abhängigkeiten
├── README.md                  # Diese Datei
├── .gitignore                # Git Ignore
├── src/paper_trading/        # Paper Trading Module
├── data/ml_models/           # ML Modelle (trainiert)
├── data/config.json          # Konfiguration
├── logs/                     # Log-Dateien
└── scripts/                  # Hilfs-Skripte
🎯 Verwendung
Starte den Bot: python main.py

Wähle Option 1 für ML-Signale

Wähle Option 3 für Paper Trading

Siehe Option 4 für Dashboard

⚠️ Warnung
Dieser Bot ist für EDUCATIONAL PURPOSES und PAPER TRADING.
Verwende kein echtes Geld ohne vollständiges Verständnis der Risiken.

📊 ML-Modell Details
Accuracy: 61.07%

AUC-ROC: 66.83%

Features: 23 technische Indikatoren

Training: 1488 klare Signale

Symbol: EURUSD, Timeframe: H1

📄 Lizenz
Educational Use Only