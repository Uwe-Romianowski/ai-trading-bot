"""
PAPER PORTFOLIO MODULE
======================
Verwaltet einen kompletten Paper-Trading Account.
"""

import json
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

# ============================================
# EINFACHER IMPORT: Kopiere die PaperOrder Klasse direkt hier
# ============================================

class OrderType(Enum):
    """Art der Order (Kauf oder Verkauf)"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Status einer Order"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


@dataclass
class PaperOrder:
    """
    Repräsentiert eine simulierte Trading-Order.
    (Vollständige Kopie aus order.py)
    """
    symbol: str
    order_type: OrderType
    entry_price: float
    quantity: float
    signal_confidence: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.OPEN
    order_id: str = field(default_factory=lambda: f"ORD_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.urandom(2).hex()}")
    
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    exit_price: Optional[float] = None
    close_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    
    def close(self, exit_price: float) -> None:
        """Schließt die Order mit einem bestimmten Preis."""
        if self.status != OrderStatus.OPEN:
            raise ValueError(f"Order {self.order_id} ist bereits {self.status.value}")
        
        self.exit_price = exit_price
        self.close_time = datetime.now()
        self.status = OrderStatus.CLOSED
        
        if self.order_type == OrderType.BUY:
            self.pnl = (exit_price - self.entry_price) * self.quantity * 100000
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity * 100000
        
        if self.entry_price > 0:
            self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity * 100000)) * 100
        else:
            self.pnl_percentage = 0
    
    def get_current_info(self) -> dict:
        """Gibt aktuelle Order-Informationen als Dictionary zurück."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'type': self.order_type.value,
            'status': self.status.value,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'current_price': self.exit_price if self.exit_price else self.entry_price,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'confidence': self.signal_confidence,
            'open_time': self.timestamp,
            'close_time': self.close_time,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }
    
    def __str__(self) -> str:
        """String-Repräsentation der Order."""
        status_str = f"{self.status.value} Order {self.order_id}"
        details = f"{self.order_type.value} {self.quantity} lots {self.symbol} @ {self.entry_price}"
        
        if self.status == OrderStatus.CLOSED and self.pnl is not None:
            pnl_sign = "+" if self.pnl > 0 else ""
            details += f" | Closed @ {self.exit_price} | P&L: {pnl_sign}{self.pnl:.2f} ({self.pnl_percentage:.2f}%)"
        
        return f"{status_str}: {details}"


# ============================================
# PAPER PORTFOLIO KLASSE
# ============================================

@dataclass
class PerformanceMetrics:
    """Performance-Kennzahlen des Portfolios"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_percentage: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    sharpe_ratio: float = 0.0
    expectancy: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_trade_duration: timedelta = timedelta(0)
    
    def to_dict(self) -> dict:
        """Konvertiert Metriken zu Dictionary"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate * 100, 2),
            'total_pnl': round(self.total_pnl, 2),
            'total_pnl_percentage': round(self.total_pnl_percentage, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_percentage': round(self.max_drawdown_percentage, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 3),
            'expectancy': round(self.expectancy, 2),
            'best_trade': round(self.best_trade, 2),
            'worst_trade': round(self.worst_trade, 2),
            'avg_trade_duration': str(self.avg_trade_duration).split('.')[0]
        }


class PaperPortfolio:
    """
    Verwaltet einen kompletten Paper-Trading Account.
    """
    
    def __init__(self, initial_balance: float = 10000.0, portfolio_id: str = None):
        """
        Initialisiert ein neues Paper Portfolio.
        """
        self.initial_balance = initial_balance
        self.current_balance = self.initial_balance
        self.equity = self.initial_balance
        
        self.max_risk_per_trade = 0.02
        self.max_daily_loss = 0.05
        self.trading_currency = "USD"
        
        self.commission = 0.0001
        self.slippage = 0.0001
        
        self.portfolio_id = portfolio_id or f"PORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.positions: Dict[str, PaperOrder] = {}
        self.trade_history: List[PaperOrder] = []
        self.daily_pnl: Dict[str, float] = {}
        
        self.peak_equity = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.max_drawdown_percentage = 0.0
        
        self.base_path = Path("data/paper_trading")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.journal_file = self.base_path / f"journal_{self.portfolio_id}.csv"
        self.performance_file = self.base_path / f"performance_{self.portfolio_id}.json"
        
        self._init_journal()
        
        print(f"✅ Paper Portfolio initialisiert: {self.portfolio_id}")
        print(f"   Startkapital: {self.initial_balance:.2f} {self.trading_currency}")
        print(f"   Max. Risiko/Trade: {self.max_risk_per_trade*100:.1f}%")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                               stop_loss: float, risk_percentage: float = None) -> float:
        """Berechnet die Positionsgröße basierend auf Risikomanagement."""
        if risk_percentage is None:
            risk_percentage = self.max_risk_per_trade
        
        risk_amount = self.equity * risk_percentage
        pip_distance = abs(entry_price - stop_loss)
        
        if 'JPY' in symbol:
            pip_value = 0.01
        else:
            pip_value = 0.0001
        
        pip_count = pip_distance / pip_value if pip_value > 0 else 0
        pip_value_per_lot = 10.0
        
        if pip_count == 0:
            return 0.0
        
        position_size = risk_amount / (pip_count * pip_value_per_lot)
        position_size = min(position_size, 10.0)
        
        return round(position_size, 2)
    
    def open_position(self, symbol: str, order_type: OrderType, 
                     entry_price: float, stop_loss: float = None,
                     take_profit: float = None, signal_confidence: float = 0.5,
                     custom_size: float = None) -> Optional[PaperOrder]:
        """Öffnet eine neue Position mit Risikomanagement."""
        if symbol in self.positions:
            print(f"⚠️  Position in {symbol} bereits offen")
            return None
        
        if custom_size is not None:
            position_size = custom_size
        elif stop_loss is not None:
            position_size = self.calculate_position_size(symbol, entry_price, stop_loss)
        else:
            position_size = 0.1
        
        trade_cost = entry_price * position_size * 100000 * self.commission
        total_cost = trade_cost + (entry_price * position_size * 100000 * self.slippage)
        
        if total_cost > self.current_balance * 0.1:
            print(f"❌ Nicht genug Kapital für Trade. Kosten: {total_cost:.2f}")
            return None
        
        order = PaperOrder(
            symbol=symbol,
            order_type=order_type,
            entry_price=entry_price,
            quantity=position_size,
            signal_confidence=signal_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = order
        self.current_balance -= total_cost
        
        self._log_trade("OPEN", order, total_cost)
        
        print(f"✅ Position eröffnet: {order}")
        print(f"   Größe: {position_size} Lots, Kosten: {total_cost:.2f}")
        
        return order
    
    def close_position(self, symbol: str, exit_price: float) -> Optional[float]:
        """Schließt eine offene Position."""
        if symbol not in self.positions:
            print(f"❌ Keine offene Position in {symbol}")
            return None
        
        order = self.positions[symbol]
        order.close(exit_price)
        
        commission_cost = exit_price * order.quantity * 100000 * self.commission
        slippage_cost = exit_price * order.quantity * 100000 * self.slippage
        total_costs = commission_cost + slippage_cost
        
        if order.pnl is not None:
            order.pnl -= total_costs
            if order.entry_price > 0:  # FEHLER KORRIGIERT: order.entry_price statt self.entry_price
                order.pnl_percentage = (order.pnl / (order.entry_price * order.quantity * 100000)) * 100
        
        self.current_balance += order.pnl if order.pnl else 0
        self.equity = self.current_balance
        
        self.trade_history.append(order)
        del self.positions[symbol]
        
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_pnl[today] = self.daily_pnl.get(today, 0) + (order.pnl or 0)
        
        self._update_performance_metrics(order)
        self._log_trade("CLOSE", order, total_costs)
        
        print(f"✅ Position geschlossen: {order}")
        
        return order.pnl
    
    def get_portfolio_summary(self) -> dict:
        """Gibt eine Zusammenfassung des Portfolios zurück."""
        unrealized_pnl = 0.0
        for order in self.positions.values():
            current_price = order.entry_price
            if order.order_type == OrderType.BUY:
                unrealized_pnl += (current_price - order.entry_price) * order.quantity * 100000
            else:
                unrealized_pnl += (order.entry_price - current_price) * order.quantity * 100000
        
        self.equity = self.current_balance + unrealized_pnl
        
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        self.current_drawdown = self.peak_equity - self.equity
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        if self.peak_equity > 0:
            self.max_drawdown_percentage = (self.max_drawdown / self.peak_equity) * 100
        
        metrics = self.calculate_performance_metrics()
        
        return {
            'portfolio_id': self.portfolio_id,
            'timestamp': datetime.now().isoformat(),
            'initial_balance': round(self.initial_balance, 2),
            'current_balance': round(self.current_balance, 2),
            'equity': round(self.equity, 2),
            'unrealized_pnl': round(unrealized_pnl, 2),
            'open_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'peak_equity': round(self.peak_equity, 2),
            'current_drawdown': round(self.current_drawdown, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_percentage': round(self.max_drawdown_percentage, 2),
            'daily_pnl': self.daily_pnl,
            'performance_metrics': metrics.to_dict()
        }
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Berechnet detaillierte Performance-Kennzahlen."""
        metrics = PerformanceMetrics()
        
        if not self.trade_history:
            return metrics
        
        metrics.total_trades = len(self.trade_history)
        winning_trades = [t for t in self.trade_history if t.pnl and t.pnl > 0]
        losing_trades = [t for t in self.trade_history if t.pnl and t.pnl <= 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        pnls = [t.pnl for t in self.trade_history if t.pnl is not None]
        if pnls:
            metrics.total_pnl = sum(pnls)
            metrics.total_pnl_percentage = (metrics.total_pnl / self.initial_balance) * 100
            
            if winning_trades:
                win_amounts = [t.pnl for t in winning_trades if t.pnl]
                metrics.avg_win = sum(win_amounts) / len(win_amounts)
                metrics.best_trade = max(win_amounts)
            
            if losing_trades:
                loss_amounts = [t.pnl for t in losing_trades if t.pnl]
                metrics.avg_loss = sum(loss_amounts) / len(loss_amounts)
                metrics.worst_trade = min(loss_amounts)
        
        total_win = sum([t.pnl for t in winning_trades if t.pnl]) if winning_trades else 0
        total_loss = abs(sum([t.pnl for t in losing_trades if t.pnl])) if losing_trades else 0
        
        if total_loss > 0:
            metrics.profit_factor = total_win / total_loss
        
        if metrics.total_trades > 0:
            metrics.expectancy = (metrics.win_rate * metrics.avg_win - 
                                (1 - metrics.win_rate) * abs(metrics.avg_loss))
        
        durations = []
        for trade in self.trade_history:
            if trade.close_time and trade.timestamp:
                durations.append(trade.close_time - trade.timestamp)
        
        if durations:
            avg_seconds = sum([d.total_seconds() for d in durations]) / len(durations)
            metrics.avg_trade_duration = timedelta(seconds=avg_seconds)
        
        return metrics
    
    def _init_journal(self):
        """Initialisiert das Trade-Journal CSV File."""
        if not self.journal_file.exists():
            with open(self.journal_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'action', 'order_id', 'symbol', 'type',
                    'entry_price', 'exit_price', 'quantity', 'pnl', 'pnl_percentage',
                    'confidence', 'stop_loss', 'take_profit', 'costs'
                ])
    
    def _log_trade(self, action: str, order: PaperOrder, costs: float = 0):
        """Loggt einen Trade ins Journal."""
        with open(self.journal_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                action,
                order.order_id,
                order.symbol,
                order.order_type.value,
                order.entry_price,
                order.exit_price if order.exit_price else '',
                order.quantity,
                order.pnl if order.pnl else '',
                order.pnl_percentage if order.pnl_percentage else '',
                order.signal_confidence,
                order.stop_loss if order.stop_loss else '',
                order.take_profit if order.take_profit else '',
                costs
            ])
    
    def _update_performance_metrics(self, order: PaperOrder):
        """Aktualisiert Performance-Metriken nach geschlossenem Trade."""
        if len(self.trade_history) % 10 == 0:
            self.save_performance_report()
    
    def save_performance_report(self):
        """Speichert einen Performance-Report als JSON."""
        report = {
            'portfolio_id': self.portfolio_id,
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_portfolio_summary(),
            'trade_history': [order.get_current_info() for order in self.trade_history[-50:]],
            'open_positions': [order.get_current_info() for order in self.positions.values()]
        }
        
        with open(self.performance_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Performance-Report gespeichert: {self.performance_file}")
    
    def print_detailed_report(self):
        """Gibt einen detaillierten Performance-Report aus."""
        summary = self.get_portfolio_summary()
        metrics = summary['performance_metrics']
        
        print("\n" + "="*60)
        print("📊 DETAILLIERTER PORTFOLIO REPORT")
        print("="*60)
        
        print(f"\n💰 KAPITALÜBERSICHT:")
        print(f"   Startkapital:      {summary['initial_balance']:>10.2f} {self.trading_currency}")
        print(f"   Aktuelles Kapital: {summary['current_balance']:>10.2f} {self.trading_currency}")
        print(f"   Equity:            {summary['equity']:>10.2f} {self.trading_currency}")
        print(f"   Unrealisierter P&L:{summary['unrealized_pnl']:>10.2f} {self.trading_currency}")
        
        print(f"\n📈 PERFORMANCE:")
        print(f"   Gesamt-P&L:        {metrics['total_pnl']:>10.2f} {self.trading_currency}")
        print(f"   P&L %:             {metrics['total_pnl_percentage']:>10.2f}%")
        print(f"   Max Drawdown:      {metrics['max_drawdown']:>10.2f} ({metrics['max_drawdown_percentage']:.2f}%)")
        
        print(f"\n🎯 TRADING STATISTIKEN:")
        print(f"   Trades gesamt:     {metrics['total_trades']:>10}")
        print(f"   Gewinner:          {metrics['winning_trades']:>10}")
        print(f"   Verlierer:         {metrics['losing_trades']:>10}")
        print(f"   Win Rate:          {metrics['win_rate']:>10.2f}%")
        print(f"   Avg Win:           {metrics['avg_win']:>10.2f}")
        print(f"   Avg Loss:          {metrics['avg_loss']:>10.2f}")
        print(f"   Profit Factor:     {metrics['profit_factor']:>10.2f}")
        print(f"   Expectancy:        {metrics['expectancy']:>10.2f}")
        
        print(f"\n📅 AKTUELLE POSITIONEN: {summary['open_positions']}")
        if self.positions:
            for symbol, order in self.positions.items():
                print(f"   {symbol}: {order.quantity} Lots @ {order.entry_price}")
        else:
            print("   Keine offenen Positionen")
        
        print(f"\n💾 Berichte gespeichert in: {self.base_path}/")
        print("="*60)


def test_paper_portfolio():
    """Testet die PaperPortfolio-Klasse."""
    print("🧪 Teste PaperPortfolio Klasse...")
    
    portfolio = PaperPortfolio(initial_balance=10000.0)
    
    print("\n1. ÖFFNE TEST-POSITIONEN:")
    
    order1 = portfolio.open_position(
        symbol="EURUSD",
        order_type=OrderType.BUY,
        entry_price=1.0850,
        stop_loss=1.0800,
        take_profit=1.0950,
        signal_confidence=0.72
    )
    
    order2 = portfolio.open_position(
        symbol="GBPUSD",
        order_type=OrderType.SELL,
        entry_price=1.2650,
        signal_confidence=0.65,
        custom_size=0.05
    )
    
    print("\n2. PORTFOLIO STATUS NACH ÖFFNEN:")
    summary = portfolio.get_portfolio_summary()
    print(f"   Offene Positionen: {summary['open_positions']}")
    print(f"   Current Balance:   {summary['current_balance']:.2f}")
    print(f"   Equity:            {summary['equity']:.2f}")
    
    print("\n3. SCHLIESSE TEST-POSITIONEN:")
    
    if order1:
        pnl1 = portfolio.close_position("EURUSD", exit_price=1.0900)
        print(f"   EURUSD geschlossen mit P&L: {pnl1:.2f}")
    
    if order2:
        pnl2 = portfolio.close_position("GBPUSD", exit_price=1.2700)
        print(f"   GBPUSD geschlossen mit P&L: {pnl2:.2f}")
    
    print("\n4. DETAILLIERTER PERFORMANCE-REPORT:")
    portfolio.print_detailed_report()
    
    portfolio.save_performance_report()
    
    print("\n✅ PaperPortfolio Test erfolgreich abgeschlossen!")
    return portfolio


if __name__ == "__main__":
    test_paper_portfolio()