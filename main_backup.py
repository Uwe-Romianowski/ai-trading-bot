#!/usr/bin/env python3
"""
AI Trading Bot v3.5 - Autonomous Evolution
Selbstoptimierender Forex Trading Bot mit ML-Integration
"""

import yaml
import logging
import sys
import os
import time
import json
import threading
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Logging Setup OHNE EMOJIS für Windows
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Importiere eigene Module
try:
    from src.ml_integration.ml_signal_generator import MLSignalGenerator
    from src.strategies.hybrid_engine import HybridEngine
    from src.self_improvement.performance_tracker import PerformanceTracker
    from src.self_improvement.adaptive_weight_manager import AdaptiveWeightManager
    from src.self_improvement.market_regime_detector import MarketRegimeDetector
except ImportError as e:
    logger.error(f"Import Fehler: {e}")
    logger.error("Stelle sicher dass alle Module existieren:")
    logger.error("  - src.ml_integration.ml_signal_generator")
    logger.error("  - src.strategies.hybrid_engine")
    logger.error("  - src.self_improvement.*")
    sys.exit(1)

def load_config(config_path: str = "config/bot_config.yaml") -> Dict:
    """Lädt die Konfiguration aus YAML Datei"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Konfiguration geladen von {config_path}")
        return config
    except Exception as e:
        logger.error(f"Fehler beim Laden der Konfiguration: {e}")
        return {}

def setup_ml_system(config: Dict) -> Optional[MLSignalGenerator]:
    """Initialisiert das ML-System"""
    logger.info("Initialisiere ML-System...")
    
    try:
        ml_generator = MLSignalGenerator(config)
        time.sleep(0.5)
        
        if ml_generator.model is not None:
            logger.info(f"ML-Modell geladen: {type(ml_generator.model).__name__}")
            logger.info("ML-System erfolgreich initialisiert")
            return ml_generator
        else:
            logger.warning("ML-Modell konnte nicht geladen werden")
            return None
            
    except Exception as e:
        logger.error(f"Fehler bei ML-System Initialisierung: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def setup_mt5_client(config: Dict) -> Optional[Any]:
    """Initialisiert den MT5 Client"""
    logger.info("Initialisiere MT5 Client...")
    
    mt5_config = config.get('mt5', {})
    login = mt5_config.get('login')
    password = mt5_config.get('password') 
    server = mt5_config.get('server')
    
    if not all([login, password, server]):
        logger.error("MT5 Login-Daten unvollstandig in Konfiguration")
        return None
    
    logger.info(f"Versuche Verbindung zu MT5 (Login: {login}, Server: {server})...")
    
    try:
        import MetaTrader5 as mt5
        
        try:
            mt5_version = mt5.__version__
            logger.info(f"MetaTrader5 Version: {mt5_version}")
        except:
            logger.warning("MetaTrader5 Version kann nicht ermittelt werden")
        
        try:
            initialized = mt5.initialize(
                login=login,
                password=password,
                server=server,
                timeout=10000,
                portable=False
            )
            
            if initialized:
                logger.info("MT5 Verbindung erfolgreich!")
                logger.info(f"   Server: {server}")
                logger.info(f"   Konto: {login}")
                
                try:
                    terminal_info = mt5.terminal_info()
                    if terminal_info:
                        logger.info(f"   Handelsumgebung: {'DEMO' if terminal_info.trade_allowed else 'REAL'}")
                except:
                    logger.info("   Handelsumgebung: Unbekannt")
                
                try:
                    rates = mt5.copy_rates_from_pos("EURUSD", mt5.TIMEFRAME_M5, 0, 10)
                    if rates is not None and len(rates) > 0:
                        logger.info(f"   Daten-Test: {len(rates)} EURUSD Kerzen empfangen")
                    else:
                        logger.warning("   Daten-Test: Keine historischen Daten empfangen")
                except Exception as e:
                    logger.warning(f"   Daten-Test fehlgeschlagen: {e}")
                
                class SimpleMT5Client:
                    def __init__(self, mt5_instance):
                        self.mt5 = mt5_instance
                        self.connected = True
                    
                    def get_historical_data(self, symbol: str, timeframe: int, count: int):
                        try:
                            rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
                            return rates if rates is not None else None
                        except:
                            return None
                    
                    def get_current_tick(self, symbol: str):
                        try:
                            return self.mt5.symbol_info_tick(symbol)
                        except:
                            return None
                    
                    def disconnect(self):
                        try:
                            self.mt5.shutdown()
                            self.connected = False
                        except:
                            pass
                    
                    def is_connected(self):
                        return self.connected
                
                return SimpleMT5Client(mt5)
            else:
                error = mt5.last_error()
                logger.error(f"MT5 Initialisierung fehlgeschlagen: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Fehler bei MT5 Initialisierung: {e}")
            return None
            
    except ImportError as e:
        logger.error(f"MetaTrader5 Bibliothek nicht verfügbar: {e}")
        logger.info("Erstelle Dummy MT5 Client für Paper Trading...")
        return create_dummy_mt5_client(config)
    except Exception as e:
        logger.error(f"Fehler bei MT5 Client Initialisierung: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_dummy_mt5_client(config: Dict) -> Any:
    """Erstellt einen Dummy MT5 Client für Fallback"""
    logger.warning("Erstelle Dummy MT5 Client für Paper Trading...")
    
    class DummyMT5:
        def __init__(self):
            self.connected = True
        
        def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
            np = __import__('numpy')
            now = datetime.now()
            rates = []
            
            for i in range(count):
                if "EURUSD" in symbol:
                    base_price = 1.1000
                elif "GBPUSD" in symbol:
                    base_price = 1.2500
                elif "USDJPY" in symbol:
                    base_price = 150.0
                else:
                    base_price = 1.0000
                
                import random
                close = base_price + random.uniform(-0.001, 0.001)
                open_price = close + random.uniform(-0.0005, 0.0005)
                high = max(open_price, close) + abs(random.uniform(0, 0.0003))
                low = min(open_price, close) - abs(random.uniform(0, 0.0003))
                volume = random.randint(100, 1000)
                
                rates.append((
                    int((now - timedelta(minutes=5*i)).timestamp()),
                    float(open_price),
                    float(high),
                    float(low),
                    float(close),
                    int(volume),
                    0,
                    int(volume)
                ))
            
            dtype = np.dtype([
                ('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
                ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
                ('spread', 'i4'), ('real_volume', 'i8')
            ])
            
            return np.array(rates, dtype=dtype)
        
        def symbol_info_tick(self, symbol):
            class Tick:
                def __init__(self):
                    import random
                    if "EURUSD" in symbol:
                        base = 1.1000
                    elif "GBPUSD" in symbol:
                        base = 1.2500
                    elif "USDJPY" in symbol:
                        base = 150.0
                    else:
                        base = 1.0000
                    
                    self.bid = base + random.uniform(-0.0001, 0.0001)
                    self.ask = self.bid + 0.0001
                    self.time = datetime.now()
                    self.volume = 1000
            return Tick()
        
        def shutdown(self):
            self.connected = False
            return True
    
    dummy_mt5 = DummyMT5()
    
    class DummyMT5Client:
        def __init__(self, mt5_instance):
            self.mt5 = mt5_instance
        
        def get_historical_data(self, symbol: str, timeframe: int, count: int):
            return self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        def get_current_tick(self, symbol: str):
            return self.mt5.symbol_info_tick(symbol)
        
        def disconnect(self):
            self.mt5.shutdown()
        
        def is_connected(self):
            return self.mt5.connected
    
    logger.info("Dummy MT5 Client initialisiert (Paper Trading Mode)")
    return DummyMT5Client(dummy_mt5)

def setup_hybrid_engine(config: Dict, ml_generator: MLSignalGenerator) -> Optional[Any]:
    """Initialisiert die Hybrid Engine"""
    logger.info("Initialisiere Hybrid Engine...")
    
    try:
        hybrid_config = config.get('hybrid', {})
        ml_weight = hybrid_config.get('ml_weight', 0.7)
        rule_weight = hybrid_config.get('rule_weight', 0.3)
        
        class BasicHybridEngine:
            def __init__(self, ml_generator=None, ml_weight=0.7, rule_weight=0.3):
                self.ml_generator = ml_generator
                self.ml_weight = ml_weight
                self.rule_weight = rule_weight
                
                total = ml_weight + rule_weight
                if total > 0:
                    self.ml_weight = ml_weight / total
                    self.rule_weight = rule_weight / total
                
                logger.info(f"Hybrid Engine initialisiert (ML: {self.ml_weight:.2f}, Rules: {self.rule_weight:.2f})")
            
            def generate_rule_signal(self, symbol: str, current_price: float) -> Dict:
                import random
                from datetime import datetime
                
                signals = ['BUY', 'SELL', 'HOLD']
                weights = [0.3, 0.3, 0.4]
                signal = np.random.choice(signals, p=weights)
                confidence = random.uniform(0.4, 0.8)
                
                return {
                    'action': signal,
                    'confidence': confidence,
                    'source': 'RULES',
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'price': current_price
                }
            
            def combine_signals(self, ml_signal=None, symbol="EURUSD", current_price=0.0) -> Dict:
                from datetime import datetime
                
                default_signal = {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'source': 'HYBRID',
                    'ml_confidence': 0.0,
                    'rule_confidence': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'price': current_price
                }
                
                rule_signal = self.generate_rule_signal(symbol, current_price)
                
                if ml_signal is None or 'error' in ml_signal:
                    default_signal['action'] = rule_signal['action']
                    default_signal['confidence'] = rule_signal['confidence']
                    default_signal['source'] = 'RULES_ONLY'
                    default_signal['rule_confidence'] = rule_signal['confidence']
                    return default_signal
                
                ml_action = ml_signal.get('signal', 'HOLD')
                ml_confidence = ml_signal.get('confidence', 0.0)
                
                if ml_action == 'HOLD' and rule_signal['action'] == 'HOLD':
                    default_signal['action'] = 'HOLD'
                    default_signal['confidence'] = 0.0
                elif ml_action == rule_signal['action']:
                    default_signal['action'] = ml_action
                    default_signal['confidence'] = (
                        ml_confidence * self.ml_weight + 
                        rule_signal['confidence'] * self.rule_weight
                    )
                else:
                    ml_score = ml_confidence * self.ml_weight
                    rule_score = rule_signal['confidence'] * self.rule_weight
                    
                    if ml_score > rule_score:
                        default_signal['action'] = ml_action
                        default_signal['confidence'] = ml_confidence
                        default_signal['source'] = 'ML_DOMINANT'
                    else:
                        default_signal['action'] = rule_signal['action']
                        default_signal['confidence'] = rule_signal['confidence']
                        default_signal['source'] = 'RULES_DOMINANT'
                
                default_signal['ml_confidence'] = ml_confidence
                default_signal['rule_confidence'] = rule_signal['confidence']
                
                if default_signal['confidence'] > 0.65 and default_signal['action'] != 'HOLD':
                    logger.info(
                        f"Hybrid Signal: {symbol} {default_signal['action']} "
                        f"(Confidence: {default_signal['confidence']:.2%}, "
                        f"Source: {default_signal['source']})"
                    )
                
                return default_signal
        
        hybrid_engine = BasicHybridEngine(ml_generator, ml_weight, rule_weight)
        logger.info(f"Hybrid Engine erfolgreich initialisiert (ML: {ml_weight}, Rules: {rule_weight})")
        return hybrid_engine
        
    except Exception as e:
        logger.error(f"Fehler bei Hybrid Engine Initialisierung: {e}")
        return None

def setup_self_improvement(config: Dict) -> Any:
    """Initialisiert das Self-Improvement System - REPARIERTE VERSION"""
    logger.info("Initialisiere Self-Improvement System...")
    
    # Extrahiere ML Weight sicher aus der Konfiguration
    try:
        hybrid_config = config.get('hybrid', {})
        ml_weight_raw = hybrid_config.get('ml_weight', 0.7)
        
        # Konvertiere sicher zu float
        if isinstance(ml_weight_raw, (int, float)):
            initial_ml_weight = float(ml_weight_raw)
        elif isinstance(ml_weight_raw, str):
            initial_ml_weight = float(ml_weight_raw)
        elif isinstance(ml_weight_raw, dict):
            # Fallback falls es ein dict ist
            initial_ml_weight = 0.7
            logger.warning(f"ML Weight ist ein Dictionary, verwende Default: {initial_ml_weight}")
        else:
            initial_ml_weight = 0.7
            logger.warning(f"Unbekannter ML Weight Typ: {type(ml_weight_raw)}, verwende Default: {initial_ml_weight}")
        
        # Sicherstellen dass Wert zwischen 0 und 1 ist
        initial_ml_weight = max(0.1, min(0.9, initial_ml_weight))
        
    except Exception as e:
        logger.warning(f"Fehler beim Extrahieren von ML Weight: {e}, verwende Default 0.7")
        initial_ml_weight = 0.7
    
    # Konfiguration für Self-Improvement
    si_config = {
        'initial_ml_weight': initial_ml_weight,
        'min_ml_weight': 0.3,
        'max_ml_weight': 0.9,
        'adjustment_step': 0.05,
        'max_history_days': 30,
        'min_trades_for_analysis': 10,
        'atr_period': 14,
        'trend_strength_period': 20,
        'volatility_threshold': 1.5
    }
    
    logger.info(f"Self-Improvement Konfiguration: ML Weight = {initial_ml_weight}")
    
    try:
        # Versuche die Self-Improvement Module zu importieren und zu verwenden
        try:
            performance_tracker = PerformanceTracker()
            adaptive_weight_manager = AdaptiveWeightManager(si_config)
            market_regime_detector = MarketRegimeDetector(si_config)
            
            class SelfImprovementManager:
                def __init__(self, tracker, weight_manager, regime_detector):
                    self.performance_tracker = tracker
                    self.adaptive_weight_manager = weight_manager
                    self.market_regime_detector = regime_detector
                    
                def update_after_trade(self, trade_result, ml_confidence):
                    """Aktualisiert nach einem Trade"""
                    try:
                        # Create trade data for PerformanceTracker
                        trade_data = {
                            'profit': trade_result.get('profit', 0) if isinstance(trade_result, dict) else trade_result,
                            'confidence': ml_confidence,
                            'symbol': trade_result.get('symbol', 'UNKNOWN') if isinstance(trade_result, dict) else 'UNKNOWN',
                            'action': trade_result.get('action', 'UNKNOWN') if isinstance(trade_result, dict) else 'UNKNOWN'
                        }
                        
                        # Update performance tracker
                        self.performance_tracker.record_trade(trade_data)
                        
                        # Prüfe ob Gewicht angepasst werden sollte
                        metrics = self.performance_tracker.get_performance_metrics(window=20)
                        if metrics['total_trades'] >= 10:
                            new_weights = self.adaptive_weight_manager.calculate_new_weights(metrics)
                            return new_weights
                    except Exception as e:
                        logger.error(f"Fehler in update_after_trade: {e}")
                    return None
                    
                def get_status(self):
                    """Gibt Status zurück"""
                    try:
                        tracker_status = self.performance_tracker.get_status()
                        return {
                            'ml_weight': self.adaptive_weight_manager.current_ml_weight,
                            'rule_weight': self.adaptive_weight_manager.current_rule_weight,
                            'total_trades': tracker_status.get('total_trades', 0),
                            'win_rate': tracker_status.get('win_rate', 0.0),
                            'performance_score': tracker_status.get('performance_score', 0.0)
                        }
                    except Exception as e:
                        logger.error(f"Fehler in get_status: {e}")
                        return {
                            'ml_weight': si_config['initial_ml_weight'],
                            'rule_weight': 1.0 - si_config['initial_ml_weight'],
                            'total_trades': 0,
                            'win_rate': 0.0,
                            'performance_score': 0.0
                        }
            
            si_manager = SelfImprovementManager(
                performance_tracker,
                adaptive_weight_manager,
                market_regime_detector
            )
            
            logger.info("Self-Improvement System initialisiert")
            return si_manager
            
        except Exception as module_error:
            logger.error(f"Fehler in Self-Improvement Modulen: {module_error}")
            raise module_error
            
    except Exception as e:
        logger.error(f"Fehler bei Self-Improvement Initialisierung: {e}")
        # Fallback: Erstelle einfache Dummy-Version
        logger.warning("Erstelle Basic Self-Improvement als Fallback")
        
        class BasicSelfImprovement:
            def __init__(self, initial_ml_weight=0.7):
                self.ml_weight = initial_ml_weight
                self.rule_weight = 1.0 - initial_ml_weight
                self.total_trades = 0
                self.winning_trades = 0
                self.win_rate = 0.0
                self.performance_score = 0.0
                
            def update_after_trade(self, trade_result, ml_confidence):
                """Aktualisiert nach einem Trade"""
                try:
                    self.total_trades += 1
                    
                    # Extrahiere Trade-Ergebnis
                    if isinstance(trade_result, dict):
                        if trade_result.get('profit', 0) > 0:
                            self.winning_trades += 1
                    elif isinstance(trade_result, (int, float)):
                        if trade_result > 0:
                            self.winning_trades += 1
                    
                    # Berechne Win Rate
                    if self.total_trades > 0:
                        self.win_rate = self.winning_trades / self.total_trades
                    
                    # Passe ML Weight basierend auf Performance an
                    if self.total_trades >= 10:
                        if self.win_rate < 0.4:
                            # Reduziere ML Vertrauen bei schlechter Performance
                            self.ml_weight = max(0.3, self.ml_weight - 0.05)
                            self.rule_weight = 1.0 - self.ml_weight
                            logger.info(f"ML Weight reduziert auf {self.ml_weight:.2f} (Win Rate: {self.win_rate:.1%})")
                        elif self.win_rate > 0.6:
                            # Erhöhe ML Vertrauen bei guter Performance
                            self.ml_weight = min(0.9, self.ml_weight + 0.05)
                            self.rule_weight = 1.0 - self.ml_weight
                            logger.info(f"ML Weight erhöht auf {self.ml_weight:.2f} (Win Rate: {self.win_rate:.1%})")
                    
                    return None
                    
                except Exception as e:
                    logger.error(f"Fehler in update_after_trade (Basic): {e}")
                    return None
                
            def get_status(self):
                """Gibt Status zurück"""
                return {
                    'ml_weight': self.ml_weight,
                    'rule_weight': self.rule_weight,
                    'total_trades': self.total_trades,
                    'win_rate': self.win_rate,
                    'performance_score': self.performance_score
                }
        
        basic_si = BasicSelfImprovement(initial_ml_weight)
        logger.info(f"Basic Self-Improvement initialisiert mit ML Weight: {initial_ml_weight}")
        return basic_si

def start_paper_trading(config: Dict, ml_generator: MLSignalGenerator, 
                        hybrid_engine: Any, si_manager: Any) -> None:
    """Startet Paper Trading Mode mit Stopp-Mechanismus"""
    print("\nSTARTE PAPER TRADING MODUS...")
    
    mt5_client = setup_mt5_client(config)
    
    if mt5_client is None:
        print("\nWARNUNG: PAPER TRADING FEHLGESCHLAGEN")
        print("   Grund: MT5 Verbindung nicht moglich")
        print("\nLOSUNGSVORSCHLAGE:")
        print("   1. Stelle sicher dass MT5 Terminal lauft")
        print("   2. Prufe Login-Daten in config/bot_config.yaml")
        
        response = input("\nMochtest du mit Dummy-Daten fortfahren? (j/n): ")
        if response.lower() == 'j':
            print("Starte Paper Trading mit Dummy-Daten...")
            mt5_client = create_dummy_mt5_client(config)
        else:
            return
    
    trading_config = config.get('trading', {})
    symbols = trading_config.get('symbols', ['EURUSD'])
    timeframe = trading_config.get('timeframe', 'M5')
    paper_trading = trading_config.get('paper_trading', True)
    
    print("\nPAPER TRADING KONFIGURATION:")
    print(f"   Symbole: {', '.join(symbols)}")
    print(f"   Timeframe: {timeframe}")
    print(f"   Modus: {'PAPER' if paper_trading else 'LIVE'}")
    
    si_status = si_manager.get_status()
    print(f"SELF-IMPROVEMENT STATUS:")
    print(f"   ML Gewicht: {si_status['ml_weight']}")
    print(f"   Regel Gewicht: {si_status['rule_weight']}")
    
    print("\n" + "=" * 60)
    print("PAPER TRADING GESTARTET")
    print("=" * 60)
    print(f"Symbole: {', '.join(symbols)}")
    print(f"ML-System: AKTIV (Gewicht: {si_status['ml_weight']})")
    print(f"Self-Improvement: AKTIV")
    print(f"Modus: PAPER TRADING")
    print("=" * 60)
    print("Drücke STRG+C zum Beenden")
    print("=" * 60)
    
    # Stopp-Flag für sauberes Beenden
    stop_event = threading.Event()
    
    def check_for_stop():
        """Prüft auf Benutzereingabe zum Stoppen"""
        try:
            while not stop_event.is_set():
                user_input = input()
                if user_input.strip().lower() == 'stop':
                    print("\nStoppe Paper Trading...")
                    stop_event.set()
                    break
        except:
            pass
    
    # Starte Stopp-Thread
    stop_thread = threading.Thread(target=check_for_stop, daemon=True)
    stop_thread.start()
    
    # Haupt-Trading Loop
    try:
        timeframe_map = {
            'M1': 1, 'M5': 5, 'M15': 15, 'M30': 30,
            'H1': 60, 'H4': 240, 'D1': 1440, 'W1': 10080, 'MN1': 43200
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, 5)
        
        print(f"\nStarte Daten-Sammlung... (benotigt {ml_generator.lookback} Kerzen)")
        print("   Der Bot sammelt jetzt historische Daten fur ML-Analyse")
        print("   Tippe 'stop' und Enter zum Beenden")
        print("   Bitte warten...")
        
        for symbol in symbols:
            logger.info(f"Sammle historische Daten fur {symbol}...")
            
            historical_data = mt5_client.get_historical_data(symbol, mt5_timeframe, 1000)
            
            if historical_data is not None and len(historical_data) > 0:
                df = pd.DataFrame(historical_data)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                logger.info(f"   {len(df)} historische Kerzen fur {symbol} empfangen")
                
                success = ml_generator.add_live_data(symbol, df)
                
                if success:
                    buffer_status = ml_generator.get_buffer_status()
                    logger.info(f"   ML-Buffer: {buffer_status['ml_buffer_rows']}/{ml_generator.lookback} Kerzen")
                else:
                    logger.warning(f"   Daten konnten nicht zum ML-Buffer hinzugefügt werden")
            else:
                logger.warning(f"   Keine historischen Daten fur {symbol} verfügbar")
        
        buffer_status = ml_generator.get_buffer_status()
        buffer_ready = buffer_status['buffer_ready']
        
        if buffer_ready:
            logger.info(f"ML-Buffer bereit! ({buffer_status['ml_buffer_rows']}/{ml_generator.lookback} Kerzen)")
            print(f"\nML-SYSTEM BEREIT!")
            print(f"   Buffer: {buffer_status['ml_buffer_rows']}/{ml_generator.lookback} Kerzen")
            print(f"   Fertigstellung: {buffer_status['completion_percentage']:.1f}%")
        else:
            logger.warning(f"ML-Buffer noch nicht voll: {buffer_status['ml_buffer_rows']}/{ml_generator.lookback}")
            print(f"\nML-SYSTEM WIRD VORBEREITET...")
            print(f"   Buffer: {buffer_status['ml_buffer_rows']}/{ml_generator.lookback} Kerzen")
            print(f"   Fertigstellung: {buffer_status['completion_percentage']:.1f}%")
            print(f"   Benotigt noch: {ml_generator.lookback - buffer_status['ml_buffer_rows']} Kerzen")
        
        print("\n" + "=" * 60)
        print("TRADING BOT AKTIV")
        print("=" * 60)
        print("Der Bot lauft jetzt im Paper Trading Modus.")
        print("Tippe 'stop' und Enter zum Beenden.")
        print("=" * 60)
        
        trade_count = 0
        last_print_time = time.time()
        
        while not stop_event.is_set():
            current_time = time.time()
            
            if current_time - last_print_time > 30:
                buffer_status = ml_generator.get_buffer_status()
                si_status = si_manager.get_status()
                
                print(f"\nSTATUSUPDATE [{datetime.now().strftime('%H:%M:%S')}]")
                print(f"   ML-Buffer: {buffer_status['ml_buffer_rows']}/{ml_generator.lookback} Kerzen")
                print(f"   ML-Gewicht: {si_status['ml_weight']:.2f}")
                print(f"   Trades: {trade_count}")
                print(f"   Win-Rate: {si_status['win_rate']:.1%}")
                
                last_print_time = current_time
            
            for symbol in symbols:
                if stop_event.is_set():
                    break
                    
                try:
                    recent_data = mt5_client.get_historical_data(symbol, mt5_timeframe, 10)
                    
                    if recent_data is not None and len(recent_data) > 0:
                        df = pd.DataFrame(recent_data)
                        df['time'] = pd.to_datetime(df['time'], unit='s')
                        
                        ml_generator.add_live_data(symbol, df.tail(1))
                        
                        if ml_generator.is_ready():
                            ml_signal = ml_generator.generate_signal()
                            
                            if ml_signal and 'error' not in ml_signal:
                                current_price = df.iloc[-1]['close'] if len(df) > 0 else 0
                                
                                try:
                                    final_signal = hybrid_engine.combine_signals(
                                        ml_signal=ml_signal,
                                        symbol=symbol,
                                        current_price=current_price
                                    )
                                    
                                    if final_signal.get('action') in ['BUY', 'SELL']:
                                        trade_count += 1
                                        logger.info(
                                            f"PAPER TRADE #{trade_count}: {symbol} {final_signal.get('action')} "
                                            f"(Confidence: {final_signal.get('confidence', 0):.2%})"
                                        )
                                        
                                        # Aktualisiere Self-Improvement
                                        trade_result = {
                                            'symbol': symbol,
                                            'action': final_signal.get('action'),
                                            'confidence': final_signal.get('confidence', 0),
                                            'profit': np.random.uniform(-10, 20),  # Simulierter Profit für Testing
                                            'timestamp': datetime.now().isoformat()
                                        }
                                        
                                        # Update Self-Improvement Manager
                                        si_manager.update_after_trade(
                                            trade_result,
                                            final_signal.get('confidence', 0)
                                        )
                                except Exception as e:
                                    logger.error(f"Fehler in Hybrid Engine: {e}")
                    
                    if not stop_event.is_set():
                        time.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Fehler bei Verarbeitung von {symbol}: {e}")
            
            if stop_event.is_set():
                break
            
            sleep_time = 300  # Default 5 Minuten
            if timeframe == 'M1':
                sleep_time = 60
            elif timeframe == 'M15':
                sleep_time = 900
            elif timeframe == 'H1':
                sleep_time = 3600
            
            # Sleep mit Unterbrechungsmöglichkeit
            for _ in range(int(sleep_time)):
                if stop_event.is_set():
                    break
                time.sleep(1)
        
        print("\nPAPER TRADING GESTOPPT")
        logger.info("Paper Trading gestoppt")
        
    except KeyboardInterrupt:
        print("\n\nPAPER TRADING GESTOPPT (Strg+C)")
        logger.info("Paper Trading durch Strg+C gestoppt")
    except Exception as e:
        logger.error(f"Fehler in Paper Trading Loop: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        print(f"\nFEHLER: {e}")
    finally:
        stop_event.set()
        if mt5_client:
            mt5_client.disconnect()
        print("\nPaper Trading beendet")

def show_system_status(ml_generator: Optional[MLSignalGenerator], 
                      hybrid_engine: Optional[Any],
                      si_manager: Any) -> None:
    """Zeigt System-Status"""
    print("\nSYSTEM-STATUS:")
    print("=" * 40)
    
    if ml_generator:
        print("ML-SYSTEM:")
        model_info = ml_generator.get_model_info()
        for key, value in model_info.items():
            if key == 'error':
                print(f"   FEHLER: {value}")
            else:
                print(f"   {key}: {value}")
        
        buffer_status = ml_generator.get_buffer_status()
        print(f"   Buffer: {buffer_status['ml_buffer_rows']}/{ml_generator.lookback}")
        print(f"   Fertigstellung: {buffer_status['completion_percentage']:.1f}%")
        
        if ml_generator.is_ready():
            print("   Status: BEREIT")
        else:
            print("   Status: WARTET AUF DATEN")
    else:
        print("ML-SYSTEM: NICHT INITIALISIERT")
    
    print("\nHYBRID ENGINE:")
    if hybrid_engine:
        print(f"   ML Gewicht: {getattr(hybrid_engine, 'ml_weight', 'N/A')}")
        print(f"   Regel Gewicht: {getattr(hybrid_engine, 'rule_weight', 'N/A')}")
        print("   Status: AKTIV")
    else:
        print("   Status: NICHT INITIALISIERT")
    
    print("\nSELF-IMPROVEMENT:")
    if si_manager:
        si_status = si_manager.get_status()
        for key, value in si_status.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2%}" if 'rate' in key else f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        print("   Status: AKTIV")
    else:
        print("   Status: NICHT INITIALISIERT")

def show_performance_report(ml_generator: Optional[MLSignalGenerator]) -> None:
    """Zeigt Performance-Report"""
    print("\nPERFORMANCE-REPORT:")
    print("=" * 40)
    
    if ml_generator:
        stats = ml_generator.get_performance_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                if 'rate' in key:
                    print(f"   {key}: {value:.2%}")
                else:
                    print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
    else:
        print("   ML-System nicht initialisiert")

def main():
    """Hauptfunktion des Trading Bots"""
    print("\n" + "=" * 60)
    print("AI TRADING BOT v3.5 STARTET...")
    print("   Version: Autonomous Evolution")
    print(f"   Datum: {datetime.now().strftime('%d.%m.%Y')}")
    print("   Feature: Vollstandiges ML + Self-Improvement System")
    print("=" * 60)
    
    config = load_config()
    
    if not config:
        print("Konfiguration konnte nicht geladen werden")
        return
    
    ml_generator = setup_ml_system(config)
    if ml_generator is None:
        print("ML-System konnte nicht initialisiert werden")
        response = input("Mochtest du ohne ML-System fortfahren? (j/n): ")
        if response.lower() != 'j':
            return
        ml_generator = None
    
    hybrid_engine = None
    if ml_generator:
        hybrid_engine = setup_hybrid_engine(config, ml_generator)
        if hybrid_engine is None:
            print("Hybrid Engine konnte nicht initialisiert werden")
    
    si_manager = setup_self_improvement(config)
    
    print("\nSelf-Improvement Manager initialisiert")
    si_status = si_manager.get_status()
    print(f"     - ML Weight: {si_status['ml_weight']:.2f}")
    print(f"     - Auto-Adjust: ENABLED")
    
    print("\n" + "=" * 60)
    print("AI TRADING BOT v3.5 - AUTONOMOUS EVOLUTION")
    print("=" * 60)
    
    while True:
        print("\nMODI:")
        print("   1. Paper Trading (Empfohlen zum Testen)")
        print("   2. Live Trading (Echtgeld - VORSICHT!)")
        print("   3. System-Status & Diagnose")
        print("   4. ML-Modell analysieren")
        print("   5. Buffer leeren & neu starten")
        print("   6. Performance-Report")
        print("   7. Self-Improvement Einstellungen")
        print("   8. Beenden")
        
        try:
            choice = input("\nWahle Option (1-8): ").strip()
            
            if choice == "1":
                if ml_generator is None:
                    print("ML-System nicht verfügbar für Paper Trading")
                    continue
                start_paper_trading(config, ml_generator, hybrid_engine, si_manager)
            elif choice == "2":
                print("\nLIVE TRADING MODUS")
                print("   Dieser Modus verwendet ECHTGELD!")
                print("   Bitte zuerst im Paper Trading Modus testen.")
                confirm = input("\nBist du sicher? (j/n): ").lower()
                if confirm == 'j':
                    print("Live Trading noch nicht implementiert - bitte teste zuerst Paper Trading")
                else:
                    print("Live Trading abgebrochen")
            elif choice == "3":
                show_system_status(ml_generator, hybrid_engine, si_manager)
            elif choice == "4":
                if ml_generator:
                    print("\nML-MODELL ANALYSE:")
                    model_info = ml_generator.get_model_info()
                    for key, value in model_info.items():
                        print(f"   {key}: {value}")
                else:
                    print("ML-System nicht initialisiert")
            elif choice == "5":
                if ml_generator:
                    ml_generator.clear_buffers()
                    print("Alle Buffer geleert")
                else:
                    print("ML-System nicht initialisiert")
            elif choice == "6":
                show_performance_report(ml_generator)
            elif choice == "7":
                print("\nSELF-IMPROVEMENT EINSTELLUNGEN:")
                si_status = si_manager.get_status()
                for key, value in si_status.items():
                    if isinstance(value, float):
                        print(f"   {key}: {value:.2%}" if 'rate' in key else f"   {key}: {value:.2f}")
                    else:
                        print(f"   {key}: {value}")
            elif choice == "8":
                print("\nBeende AI Trading Bot...")
                logger.info("Bot wird beendet")
                break
            else:
                print("Ungultige Eingabe, bitte 1-8 wahlen")
                
        except KeyboardInterrupt:
            print("\n\nBeende AI Trading Bot...")
            break
        except Exception as e:
            logger.error(f"Fehler im Hauptmenu: {e}")
            print(f"Fehler: {e}")

if __name__ == "__main__":
    main()