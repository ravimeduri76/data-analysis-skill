@echo off
REM Kaggle Daily Analysis — scheduled launcher
REM Runs kaggle_daily.py with --notify flag for email/WhatsApp alerts

cd /d "C:\Users\Ravi\research\data-analysis"

REM Load .env if present (for POSTGRES_URL, ANTHROPIC_API_KEY, etc.)
if exist "C:\Users\Ravi\Data_Analysis\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("C:\Users\Ravi\Data_Analysis\.env") do (
        set "%%A=%%B"
    )
)

REM Python script handles its own logging to kaggle_daily.log — no redirect needed
"C:\Users\Ravi\AppData\Local\Programs\Python\Python313\python.exe" kaggle_daily.py --notify
