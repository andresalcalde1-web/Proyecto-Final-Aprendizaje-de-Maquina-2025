@echo off
cls
echo ========================================
echo   Red Wine Quality - Ejecutar App
echo ========================================
echo.
echo Verificando Streamlit...
where streamlit >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: Streamlit no esta instalado o no esta en el PATH
    echo.
    echo Por favor:
    echo 1. Cierre esta ventana
    echo 2. Ejecute setup.bat primero
    echo 3. Cierre y abra una nueva ventana de comandos
    echo 4. Ejecute run.bat de nuevo
    echo.
    echo Si el problema persiste, ejecute manualmente:
    echo    python -m streamlit run app.py
    echo.
    pause
    exit /b 1
)

echo Iniciando aplicación...
echo Se abrirá en su navegador automáticamente.
echo.
echo IMPORTANTE: NO cierre esta ventana mientras usa la app.
echo Para detener la aplicación, presione Ctrl+C aquí.
echo.
streamlit run app.py

if errorlevel 1 (
    echo.
    echo ERROR: La aplicación falló.
    echo Intente ejecutar manualmente:
    echo    python -m streamlit run app.py
    echo.
    pause
)
