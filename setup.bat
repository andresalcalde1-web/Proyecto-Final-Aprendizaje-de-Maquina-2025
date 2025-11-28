@echo off
cls
echo ========================================
echo   Red Wine Quality - Instalacion
echo ========================================
echo.

echo [1/3] Verificando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado
    echo.
    echo Descargue Python desde: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
python --version
echo.

echo [2/3] Instalando dependencias...
echo Esto puede tomar 1-2 minutos...
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Fallo la instalacion
    echo Intente ejecutar manualmente: pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo.

echo [3/3] Instalacion completada exitosamente!
echo.
echo ==========================================
echo   SIGUIENTE PASO:
echo ==========================================
echo.
echo 1. Cierre esta ventana
echo 2. Abra una NUEVA ventana de comandos
echo 3. Ejecute run.bat
echo.
echo O ejecute manualmente en la nueva ventana:
echo    streamlit run app.py
echo.
pause
