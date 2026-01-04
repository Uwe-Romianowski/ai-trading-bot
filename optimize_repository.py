#!/usr/bin/env python3
"""
AI TRADING BOT REPOSITORY OPTIMIERER
======================================
Führt alle besprochenen Optimierungen automatisch durch:
1. Optimiert .gitignore für Python-Trading-Projekte
2. Erstellt .env-Template für sichere Konfiguration
3. Entfernt bereits commitete ignorierte Dateien
4. Legt Grundstruktur für Phase D (Paper-Trading) an
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

class RepositoryOptimizer:
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path).absolute()
        self.gitignore_path = self.repo_path / ".gitignore"
        self.env_template_path = self.repo_path / ".env.template"
        self.env_path = self.repo_path / ".env"
        
        print(f"🔧 AI Trading Bot Repository Optimizer")
        print(f"📁 Repository-Pfad: {self.repo_path}")
        print("=" * 60)
    
    def check_git_repo(self):
        """Überprüft, ob wir in einem Git-Repository sind"""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            print("❌ FEHLER: Kein Git-Repository gefunden!")
            print("   Führen Sie 'git init' zuerst aus oder wechseln Sie ins richtige Verzeichnis.")
            return False
        return True
    
    def optimize_gitignore(self):
        """Ersetzt oder erstellt eine optimierte .gitignore-Datei"""
        
        optimized_gitignore = """# ============================================
# OPTIMIERTE .gitignore FÜR AI TRADING BOT v4.0
# Generiert am: {date}
# ============================================

# 1. BYTE-COMPILED / OPTIMIZED PYTHON FILES
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# 2. PYTHON UMGEBUNGEN & VIRTUAL ENVS
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
pythonenv*

# 3. PYCHARM & VS CODE IDE FILES
.idea/
.vscode/
*.swp
*.swo
*~
*.orig
*.bak

# 4. UNIT TESTS & COVERAGE
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
test_output/
*.pytest_cache

# 5. OPERATING SYSTEM FILES
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
desktop.ini

# 6. JUPYTER NOTEBOOKS & IPYTHON
.ipynb_checkpoints/
*.ipynb
.ipython/

# 7. TRADING-BOT SPEZIFISCHE IGNORES
# ------------------------------------
# Temporäre Backtest-Daten (können regeneriert werden)
data/performance/session_*.json

# ML-Training-Zwischenergebnisse
data/models/training_*.pkl
data/models/temp_*

# Sensible Zustandsdateien (lokal speichern)
data/self_improvement_state.json

# Temporäre Logs und Caches
*.log
logs/
cache/
tmp/
temp/

# MT5-spezifische temporäre Dateien
*.chr
*.dat

# 8. LARGE DATA FILES (bereits mit Git LFS verwaltet)
# HINWEIS: Diese Dateien werden bereits von Git LFS verwaltet
# und sollten NICHT hier ignoriert werden:
# data/raw/eurusdm5_2002_2019.csv
# data/processed/X_eurusd_float32.npy
# data/processed/y_eurusd.npy

# 9. PROJEKT-METADATEN
.project
.pydevproject
.settings/
*.sublime-*
*.komodoproject

# 10. WINDOWS SPEZIFISCHES
[Tt]humbs.db
[Bb]in/
[Oo]bj/
[Ll]og/
[Dd]ebug/
[Rr]elease/
*.user
*.suo
*.aps
*.ncb
*.opensdf
*.sdf
Cache/
Backup*/
[Tt]emp*/

# ============================================
# ENDE OPTIMIERTE .gitignore
# ============================================
""".format(date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        # Backup der alten .gitignore falls vorhanden
        if self.gitignore_path.exists():
            backup_path = self.gitignore_path.with_suffix('.gitignore.backup')
            shutil.copy2(self.gitignore_path, backup_path)
            print(f"📦 Backup der alten .gitignore erstellt: {backup_path.name}")
        
        # Neue .gitignore schreiben
        with open(self.gitignore_path, 'w', encoding='utf-8') as f:
            f.write(optimized_gitignore)
        
        print("✅ .gitignore erfolgreich optimiert")
        print(f"   - {len(optimized_gitignore.splitlines())} Zeilen")
        print(f"   - Enthält Sicherheits- und Performance-Optimierungen")
        return True
    
    def create_env_template(self):
        """Erstellt eine .env.template-Datei mit allen benötigten Umgebungsvariablen"""
        
        env_template = """# ============================================
# .env TEMPLATE FÜR AI TRADING BOT v4.0
# ============================================
# KOPIEREN SIE DIESE DATEI NACH '.env' UND FÜLLEN SIE DIE WERTE AUS
# NIEMALS .env COMMITEN! (steht in .gitignore)
# ============================================

# 1. METATRADER 5 (MT5) KONFIGURATION
# ====================================
# MT5 Demo Account (wie in Phase C verwendet)
MT5_LOGIN="REMOVED_MT5_LOGIN"
MT5_PASSWORD="IHR_MT5_DEMO_PASSWORT_HIER"
MT5_SERVER="MetaQuotes-Demo"
MT5_PATH="C:/Program Files/MetaTrader 5/terminal64.exe"

# 2. HANDELSKONFIGURATION
# ====================================
# Basiswährung und Risikomanagement
TRADING_BASE_CURRENCY="USD"
MAX_RISK_PER_TRADE=0.02  # 2% des Kontos pro Trade
MAX_DAILY_LOSS=0.05      # 5% maximaler Tagesverlust

# 3. ML-MODELL KONFIGURATION
# ====================================
# Confidence-Schwellenwerte für Trading-Signale
ML_BUY_THRESHOLD=0.60    # Mindest-Confidence für BUY-Signal
ML_SELL_THRESHOLD=0.60   # Mindest-Confidence für SELL-Signal
ML_MIN_CONFIDENCE=0.52   # Minimale Confidence für HOLD

# 4. PAPER TRADING (PHASE D) KONFIGURATION
# ====================================
PAPER_TRADING_INITIAL_BALANCE=10000.0
PAPER_TRADING_COMMISSION=0.0001  # 0.01% pro Trade
PAPER_TRADING_SLIPPAGE=0.0001    # 0.01% Slippage

# 5. API-KEYS (falls zukünftig benötigt)
# ====================================
# ALPHA_VANTAGE_API_KEY=""
# FRED_API_KEY=""
# OANDA_API_KEY=""

# 6. LOGGING & DEBUGGING
# ====================================
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR
ENABLE_PERFORMANCE_TRACKING=true
SAVE_TRADE_LOGS=true

# ============================================
# HINWEIS: Verwenden Sie in Ihrem Code python-dotenv:
# from dotenv import load_dotenv
# load_dotenv()
# import os
# login = os.getenv('MT5_LOGIN')
# ============================================
"""
        
        with open(self.env_template_path, 'w', encoding='utf-8') as f:
            f.write(env_template)
        
        # Falls .env nicht existiert, Template kopieren
        if not self.env_path.exists():
            shutil.copy2(self.env_template_path, self.env_path)
            print("⚠️  .env Datei erstellt. BITTE BEARBEITEN SIE DIE SENSIBLEN DATEN!")
        
        print("✅ .env.template erfolgreich erstellt")
        print("   - Enthält alle benötigten Konfigurationsvariablen")
        print("   - Sensible Daten müssen in .env eingetragen werden")
        return True
    
    def remove_ignored_from_git(self):
        """Entfernt bereits commitete Dateien, die nun ignoriert werden sollen"""
        
        patterns_to_remove = [
            "__pycache__/",
            "*.pyc",
            ".env",
            "data/performance/session_*.json",
            "data/self_improvement_state.json",
            ".vscode/",
            ".idea/"
        ]
        
        print("🗑️  Entferne ignorierte Dateien aus Git-History...")
        
        removed_count = 0
        for pattern in patterns_to_remove:
            try:
                # Git-Befehl zum Entfernen aus Cache
                cmd = ['git', 'rm', '--cached', '-r', '--ignore-unmatch', pattern]
                result = subprocess.run(
                    cmd, 
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout:
                    print(f"   ✓ {pattern}")
                    removed_count += 1
            except Exception as e:
                print(f"   ✗ {pattern}: {e}")
        
        if removed_count > 0:
            print(f"\n📝 {removed_count} Muster aus Git-Cache entfernt")
            print("   Führen Sie 'git commit' manuell aus, um die Änderungen zu bestätigen")
        else:
            print("   Keine zu entfernenden Dateien gefunden")
        
        return removed_count
    
    def create_phase_d_structure(self):
        """Legt die Grundstruktur für Phase D (Paper-Trading) an"""
        
        phase_d_dirs = [
            "src/paper_trading/",
            "src/paper_trading/orders/",
            "src/paper_trading/portfolio/",
            "src/paper_trading/risk/",
            "data/paper_trading/",
            "tests/paper_trading/"
        ]
        
        phase_d_files = {
            "src/paper_trading/__init__.py": "# Paper Trading Engine\n",
            "src/paper_trading/core.py": "# Kernklassen für Paper Trading\n",
            "docs/phase_d_plan.md": "# Phase D: Paper Trading Engine Plan\n"
        }
        
        print("🏗️  Erstelle Grundstruktur für Phase D...")
        
        # Verzeichnisse erstellen
        for directory in phase_d_dirs:
            dir_path = self.repo_path / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / "__init__.py").touch(exist_ok=True)
        
        # Dateien erstellen
        for file_path, content in phase_d_files.items():
            full_path = self.repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not full_path.exists():
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"   ✓ {file_path}")
        
        print("✅ Phase D Grundstruktur erstellt")
        return True
    
    def install_requirements(self):
        """Installiert benötigte Python-Pakete für verbesserte Sicherheit"""
        
        requirements = [
            "python-dotenv",    # Für .env Verwaltung
            "python-gitignore", # Für .gitignore Parsing
            "colorama"          # Für farbige Konsolenausgabe
        ]
        
        print("📦 Installiere empfohlene Pakete...")
        
        for package in requirements:
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", package],
                    cwd=self.repo_path,
                    capture_output=True
                )
                print(f"   ✓ {package}")
            except Exception as e:
                print(f"   ✗ {package}: {e}")
        
        # requirements.txt aktualisieren
        req_path = self.repo_path / "requirements.txt"
        if req_path.exists():
            with open(req_path, 'a', encoding='utf-8') as f:
                f.write("\n# Phase D Optimierungen\n")
                for package in requirements:
                    f.write(f"{package}\n")
        
        return True
    
    def show_summary(self):
        """Zeigt eine Zusammenfassung der durchgeführten Änderungen"""
        
        print("\n" + "=" * 60)
        print("📊 ZUSAMMENFASSUNG DER OPTIMIERUNGEN")
        print("=" * 60)
        
        summary = {
            ".gitignore optimiert": self.gitignore_path.exists(),
            ".env.template erstellt": self.env_template_path.exists(),
            "Phase D Struktur angelegt": (self.repo_path / "src/paper_trading").exists(),
            "Git Repository geprüft": self.check_git_repo()
        }
        
        for item, status in summary.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {item}")
        
        print("\n📋 NÄCHSTE SCHRITTE:")
        print("1. Bearbeiten Sie die .env Datei mit Ihren MT5 Zugangsdaten")
        print("2. Prüfen Sie die .gitignore auf Ihre spezifischen Bedürfnisse")
        print("3. Führen Sie aus, um Änderungen zu committen:")
        print("   git add .gitignore .env.template")
        print("   git commit -m 'Repository optimiert für Phase D'")
        print("   git push origin main")
        
        if self.env_path.exists():
            print("\n⚠️  WICHTIGER SICHERHEITSHINWEIS:")
            print(f"   Die Datei '{self.env_path.name}' enthält SENSIBLE DATEN.")
            print("   Stellen Sie sicher, dass sie in .gitignore steht!")
            print("   Committen Sie diese Datei NIEMALS!")

def main():
    """Hauptfunktion des Optimierers"""
    
    optimizer = RepositoryOptimizer()
    
    # Prüfe Git Repository
    if not optimizer.check_git_repo():
        sys.exit(1)
    
    try:
        # Führe alle Optimierungen durch
        optimizer.optimize_gitignore()
        print()
        
        optimizer.create_env_template()
        print()
        
        optimizer.remove_ignored_from_git()
        print()
        
        optimizer.create_phase_d_structure()
        print()
        
        optimizer.install_requirements()
        print()
        
        optimizer.show_summary()
        
    except Exception as e:
        print(f"\n❌ FEHLER WÄHREND DER OPTIMIERUNG: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()