#!/usr/bin/env python3
"""
PHASE D TEST - UMFACHT IMPORT PROBLEM
"""

import os
import sys

print("🚀 PHASE D DIRECT TEST")
print("="*50)

# 1. Pfad anpassen
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'src'))
sys.path.insert(0, os.path.join(current_dir, 'src', 'paper_trading'))

print("📁 Python Pfade:")
for p in sys.path[:3]:
    print(f"   {p}")

# 2. Portfolio importieren
print("\n📦 Versuche Portfolio zu importieren...")
try:
    # OPTION A: Als Modul importieren
    import importlib.util
    
    portfolio_path = os.path.join(current_dir, 'src', 'paper_trading', 'portfolio.py')
    spec = importlib.util.spec_from_file_location("portfolio", portfolio_path)
    portfolio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portfolio_module)
    
    PaperPortfolio = portfolio_module.PaperPortfolio
    OrderType = portfolio_module.OrderType
    
    print("✅ Portfolio erfolgreich importiert!")
    
    # 3. Teste Portfolio
    print("\n🧪 Teste Portfolio Erstellung...")
    portfolio = PaperPortfolio(initial_balance=10000.0)
    print(f"✅ Portfolio erstellt: {portfolio.portfolio_id}")
    print(f"💰 Balance: {portfolio.current_balance:.2f} USD")
    
    # 4. Teste Trade
    print("\n🎯 Teste Trade...")
    order = portfolio.open_position(
        symbol="EURUSD",
        order_type=OrderType.BUY,
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0900,
        signal_confidence=0.75
    )
    
    if order:
        print(f"✅ Trade eröffnet: {order.order_id}")
        
        # Schließe Trade
        pnl = portfolio.close_position("EURUSD", 1.0875)
        print(f"💰 Trade geschlossen mit P&L: {pnl:.2f}")
        
        # Zeige Report
        print("\n📊 FINALER REPORT:")
        summary = portfolio.get_portfolio_summary()
        print(f"   Balance: {summary['current_balance']:.2f} USD")
        print(f"   Trades: {summary['total_trades']}")
        print(f"   P&L: {summary['performance_metrics']['total_pnl']:.2f} USD")
        
        # Speichern
        portfolio.save_performance_report()
        print("💾 Report gespeichert")
        
except Exception as e:
    print(f"❌ FEHLER: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("🎉 PHASE D TEST ABGESCHLOSSEN")