"""
Backtesting für ML Forex Strategien
"""

import numpy as np
import pandas as pd
import yaml
from datetime import datetime
import matplotlib.pyplot as plt
import os

class Backtester:
    """Backtesting Engine für ML-Strategien"""
    
    def __init__(self):
        print("=" * 60)
        print("📊 ML STRATEGY BACKTESTER")
        print("=" * 60)
    
    def run_demo_backtest(self):
        """Führt einen Demo-Backtest durch"""
        print("🧪 Starte Demo-Backtest...")
        
        # Simulierte Performance
        np.random.seed(42)
        n_periods = 100
        returns = np.random.randn(n_periods) * 0.01  # 1% tägliche Volatilität
        
        # ML-Strategy outperforms baseline
        ml_returns = returns + np.random.randn(n_periods) * 0.002
        
        # Kumulative Returns berechnen
        baseline_cum = np.cumprod(1 + returns)
        ml_cum = np.cumprod(1 + ml_returns)
        
        # Plotten
        plt.figure(figsize=(12, 6))
        plt.plot(baseline_cum, label='Baseline Strategy', linewidth=2)
        plt.plot(ml_cum, label='ML Strategy', linewidth=2)
        plt.title('Backtest Performance: ML vs Baseline')
        plt.xlabel('Trading Periods')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance-Metriken
        print("\n📈 PERFORMANCE METRIKEN:")
        print(f"   Baseline Return: {(baseline_cum[-1] - 1) * 100:.1f}%")
        print(f"   ML Strategy Return: {(ml_cum[-1] - 1) * 100:.1f}%")
        print(f"   Outperformance: {(ml_cum[-1] - baseline_cum[-1]) * 100:.1f}%")
        
        # Speichere Plot
        os.makedirs('data/plots', exist_ok=True)
        plt.savefig('data/plots/backtest_results.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Backtest abgeschlossen - Plot gespeichert")
        return True

def main():
    """Hauptfunktion für Backtesting"""
    backtester = Backtester()
    backtester.run_demo_backtest()

if __name__ == "__main__":
    main()