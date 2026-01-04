# AI Forex ML Scalping Bot v4.0

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-phase%20c%20completed-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![MT5 Integration](https://img.shields.io/badge/MT5-Integrated-blue)
![ML Model](https://img.shields.io/badge/ML-Random%20Forest-orange)

## ?? Projektübersicht

Ein hybrider Forex Trading Bot, der Machine Learning (Random Forest) mit traditioneller technischer Analyse kombiniert. Das System ist vollständig mit MetaTrader 5 integriert und bereit für Paper-Trading.

## ? Features

- **ML-Modell**: Random Forest Classifier mit 41 Features
- **Live Integration**: MT5 Demo Account Verbindung
- **Signalerzeugung**: Confidence-basierte Trading-Signale (65% Threshold)
- **Multi-Timeframe**: M5 Daten mit 100 Kerzen Lookback
- **Risikomanagement**: Integrierte Stop-Loss und Position Sizing Logik

## ??? Systemarchitektur

\\\
ai_bot/
+-- config/              # Konfigurationseinstellungen
+-- data/               # ML Modelle und Daten
¦   +-- models/
¦   ¦   +-- random_forest_model.joblib
¦   ¦   +-- feature_scaler.joblib
¦   ¦   +-- feature_names.json
+-- src/                # Quellcode
¦   +-- ml_integration/
¦   ¦   +-- ml_signal_generator.py
+-- docs/               # Dokumentation
+-- main.py            # Hauptskript
+-- requirements.txt   # Abhängigkeiten
\\\

## ?? Schnellstart

### Voraussetzungen
- Python 3.8+
- MetaTrader 5 Account (Demo oder Live)
- Git

### Installation

1. **Repository klonen**
\\\ash
git clone https://github.com/Uwe-Romianowski/ai-trading-bot.git
cd ai-trading-bot
\\\

2. **Virtuelle Umgebung erstellen (empfohlen)**
\\\ash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
\\\

3. **Abhängigkeiten installieren**
\\\ash
pip install -r requirements.txt
\\\

4. **Konfiguration anpassen**
\\\yaml
# config/bot_config.yaml
mt5:
  login: REMOVED_MT5_LOGIN      # Ihr MT5 Demo Account
  server: "YourServer"
  password: "YourPassword"
  
trading:
  symbol: "EURUSD"
  timeframe: "M5"
  risk_per_trade: 0.01  # 1% Risiko pro Trade
\\\

5. **Bot starten**
\\\ash
python main.py
\\\

## ?? Technische Spezifikationen

| Parameter | Wert | Beschreibung |
|-----------|------|--------------|
| Algorithmus | Random Forest | 50 Decision Trees Ensemble |
| Features | 41 | Technische Indikatoren |
| Lookback | 100 Kerzen | 8.3 Stunden (M5) |
| Confidence Threshold | 0.65 | Minimum für Trade-Signal |
| Training Data | EURUSD M5 | Historische Forex Daten |
| Modell-Größe | ~150KB | Serialisiertes .joblib |

## ?? Phase Status

### ? Abgeschlossen
- **Phase A**: ML Research Modul (16.12.2025)
- **Phase B**: Hybride Integration (02.01.2026)
- **Phase C**: Live-Testing mit MT5 Demo (02.01.2026)

### ?? In Entwicklung
- **Phase D**: Paper-Trading Engine & Performance Tracking

## ?? Performance Metriken

- **Modell-Genauigkeit**: 51-55% (Backtest)
- **Signal Confidence**: 52% (aktuell im HOLD Modus)
- **Buffer Management**: 100/100 Kerzen stabil
- **MT5 Verbindung**: Erfolgreich (Demo Account REMOVED_MT5_LOGIN)

## ??? Risikomanagement

- Maximal 1% Risiko pro Trade
- Stop-Loss basierend auf ATR
- Daily Loss Limit: 5%
- Positionsgröße basierend auf ML-Confidence

## ?? Beitragen

1. Fork das Repository
2. Feature Branch erstellen (\git checkout -b feature/AmazingFeature\)
3. Änderungen committen (\git commit -m 'Add AmazingFeature'\)
4. Push zum Branch (\git push origin feature/AmazingFeature\)
5. Pull Request öffnen

## ?? Lizenz

Dieses Projekt ist unter der MIT License lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## ?? Kontakt

Uwe Romianowski - [GitHub](https://github.com/Uwe-Romianowski)

Projekt Link: [https://github.com/Uwe-Romianowski/ai-trading-bot](https://github.com/Uwe-Romianowski/ai-trading-bot)

## ?? Danksagung

- MetaTrader 5 Python Integration
- Scikit-learn für ML Algorithmen
- Alle Contributoren und Tester

---

**Letzte Aktualisierung:** 02.01.2026  
**Version:** v4.0  
**Status:** Produktionsbereit für Phase D
