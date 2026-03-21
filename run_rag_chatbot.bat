@echo off
setlocal

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"

REM Go to scripts directory
cd /d "%SCRIPT_DIR%scripts"

REM Activate the environment
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" ragproject

REM Run with Streamlit
streamlit run rag_chatbot.py"

pause