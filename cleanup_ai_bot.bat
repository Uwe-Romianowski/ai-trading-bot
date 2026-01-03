@echo off
echo ========================================
echo 🧹 FINAL CLEANUP - PHASE C ABGESCHLOSSEN
echo ========================================
echo.
echo 🔍 Lösche unnötige Dateien...
echo.

:: 1. ALTE ML GENERATOR BACKUPS
if exist "src\ml_integration\ml_signal_generator_old.py" (
    echo ✅ Lösche: ml_signal_generator_old.py
    del "src\ml_integration\ml_signal_generator_old.py"
)

if exist "src\ml_integration\ml_signal_generator_fixed.py" (
    echo ✅ Lösche: ml_signal_generator_fixed.py
    del "src\ml_integration\ml_signal_generator_fixed.py"
)

:: 2. VERALTETE MT5 INTEGRATION
if exist "src\mt5_integration.py" (
    echo ✅ Lösche: mt5_integration.py
    del "src\mt5_integration.py"
)

:: 3. REPAIR SCRIPTS
for %%f in (repair_ml.py fix_performance_*.py repair_self_improvement.py archivierung.py) do (
    if exist "%%f" (
        echo ✅ Lösche: %%f
        del "%%f"
    )
)

:: 4. ALTE LOGS (behalte nur trading_bot.log)
for %%f in (trading_bot_fixed.log trading_bot_ml_fixed.log trading_bot_final.log trading_bot_final_fix.log trading_bot_ultimate_fix.log trading_bot_ml_buffer_fix.log trading_bot_buffer_sync.log trading_bot_data_fix.log debug_bot.py) do (
    if exist "%%f" (
        echo ✅ Lösche: %%f
        del "%%f"
    )
)

:: 5. PYTHON CACHE
echo ✅ Lösche Python Cache...
rmdir /s /q __pycache__ 2>nul
rmdir /s /q src\__pycache__ 2>nul
del /s /q *.pyc 2>nul

echo.
echo ========================================
echo 🎉 CLEANUP ABGESCHLOSSEN!
echo 📊 System ist jetzt sauber für Git Commit.
echo.
pause