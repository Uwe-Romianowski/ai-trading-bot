#!/usr/bin/env python3
"""
Fix für PerformanceTracker Attribute Fehler
"""

import os

def fix_performance_tracker():
    """Repariert die PerformanceTracker Klasse"""
    file_path = "src/self_improvement/performance_tracker.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Datei nicht gefunden: {file_path}")
        return False
    
    print(f"🔧 Repariere: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup erstellen
        backup_path = file_path + ".backup2"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  📁 Backup erstellt: {backup_path}")
        
        # Prüfe ob die Klasse die benötigten Attribute hat
        lines = content.split('\n')
        
        # Finde die __init__ Methode
        init_start = -1
        for i, line in enumerate(lines):
            if 'def __init__' in line:
                init_start = i
                break
        
        if init_start == -1:
            print("  ❌ __init__ Methode nicht gefunden")
            return False
        
        # Finde das Ende der __init__ Methode (basierend auf Einrückung)
        init_end = init_start
        for i in range(init_start + 1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(' ' * 8):
                init_end = i
                break
        
        # Prüfe ob Attribute existieren
        has_total_trades = False
        has_win_rate = False
        has_performance_score = False
        
        for i in range(init_start, min(init_end, len(lines))):
            if 'self.total_trades =' in lines[i]:
                has_total_trades = True
            if 'self.win_rate =' in lines[i]:
                has_win_rate = True
            if 'self.performance_score =' in lines[i]:
                has_performance_score = True
        
        print(f"  📊 Attribute gefunden: total_trades={has_total_trades}, win_rate={has_win_rate}, performance_score={has_performance_score}")
        
        # Füge fehlende Attribute hinzu
        if not (has_total_trades and has_win_rate and has_performance_score):
            # Finde wo self.logger.info steht
            insert_index = -1
            for i in range(init_start, min(init_end, len(lines))):
                if 'self.logger.info' in lines[i]:
                    insert_index = i + 1
                    break
            
            if insert_index == -1:
                # Falls kein logger.info, füge nach der ersten self. Zeile hinzu
                for i in range(init_start, min(init_end, len(lines))):
                    if 'self.' in lines[i]:
                        insert_index = i + 1
                        break
            
            if insert_index != -1:
                # Füge Attribute ein
                attributes = []
                if not has_total_trades:
                    attributes.append('        self.total_trades = 0')
                if not has_win_rate:
                    attributes.append('        self.win_rate = 0.0')
                if not has_performance_score:
                    attributes.append('        self.performance_score = 0.0')
                
                for attr in reversed(attributes):  # Rückwärts einfügen um Reihenfolge zu behalten
                    lines.insert(insert_index, attr)
                
                print(f"  ✅ {len(attributes)} Attribute hinzugefügt")
        
        # Prüfe ob die record_trade Methode die Attribute aktualisiert
        if 'def record_trade' in content:
            # Finde die record_trade Methode
            for i, line in enumerate(lines):
                if 'def record_trade' in line:
                    # Suche bis zum Ende der Methode
                    method_end = i
                    for j in range(i + 1, len(lines)):
                        if lines[j].strip() and not lines[j].startswith(' ' * 8):
                            method_end = j
                            break
                    
                    # Prüfe ob self.total_trades aktualisiert wird
                    has_total_trades_update = False
                    has_win_rate_update = False
                    
                    for j in range(i, method_end):
                        if 'self.total_trades +=' in lines[j] or 'self.total_trades = self.total_trades +' in lines[j]:
                            has_total_trades_update = True
                        if 'self.win_rate =' in lines[j] and 'self.winning_trades / self.total_trades' in content:
                            has_win_rate_update = True
                    
                    # Füge Updates hinzu falls nötig
                    if not has_total_trades_update:
                        # Finde wo self.trades.append steht
                        for j in range(i, method_end):
                            if 'self.trades.append' in lines[j]:
                                # Füge nach dieser Zeile ein
                                lines.insert(j + 1, '            self.total_trades += 1')
                                print("  ✅ total_trades Update hinzugefügt")
                                break
                    
                    if not has_win_rate_update:
                        # Finde wo winning/losing trades aktualisiert wird
                        for j in range(i, method_end):
                            if 'self.winning_trades +=' in lines[j] or 'self.losing_trades +=' in lines[j]:
                                # Füge win_rate Berechnung nach der letzten solchen Zeile hinzu
                                lines.insert(j + 2, '            if self.total_trades > 0:')
                                lines.insert(j + 3, '                self.win_rate = self.winning_trades / self.total_trades')
                                print("  ✅ win_rate Update hinzugefügt")
                                break
        
        # Aktualisierte Datei speichern
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"  ✅ Datei repariert: {file_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_fix():
    """Testet ob der Fix funktioniert"""
    print("\n🧪 Teste Reparatur...")
    
    try:
        import sys
        sys.path.insert(0, 'src')
        
        from self_improvement.performance_tracker import PerformanceTracker
        
        pt = PerformanceTracker()
        
        # Teste Attribute
        print(f"  ✅ total_trades: {pt.total_trades}")
        print(f"  ✅ win_rate: {pt.win_rate}")
        print(f"  ✅ performance_score: {pt.performance_score}")
        
        # Teste record_trade
        pt.record_trade({'profit': 10.0}, 0.8)
        print(f"  ✅ Nach Trade - total_trades: {pt.total_trades}")
        print(f"  ✅ Nach Trade - win_rate: {pt.win_rate}")
        
        # Teste get_status
        status = pt.get_status()
        print(f"  ✅ get_status - total_trades: {status.get('total_trades', 'N/A')}")
        print(f"  ✅ get_status - win_rate: {status.get('win_rate', 'N/A')}")
        print(f"  ✅ get_status - performance_score: {status.get('performance_score', 'N/A')}")
        
        print("\n🎉 ALLE TESTS BESTANDEN!")
        return True
        
    except Exception as e:
        print(f"❌ TEST FEHLGESCHLAGEN: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 FIX PERFORMANCE TRACKER ATTRIBUTE")
    print("=" * 60)
    
    if fix_performance_tracker():
        if test_fix():
            print("\n" + "=" * 60)
            print("✅ REPARATUR ERFOLGREICH!")
            print("=" * 60)
            print("\nStarte den Bot neu mit: python main.py")
            print("Dann Option 7 wählen um Self-Improvement zu prüfen")
        else:
            print("\n⚠️  Reparatur getestet, aber Test fehlgeschlagen")
    else:
        print("\n❌ Reparatur fehlgeschlagen")

if __name__ == "__main__":
    main()