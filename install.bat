@echo off
setlocal EnableDelayedExpansion
set KMP_DUPLICATE_LIB_OK=TRUE

:: =====================================================
:: 1. Scarica ed installa Tesseract-OCR-w64 (ultima versione)
:: =====================================================
echo Vuoi scaricare ed installare Tesseract-OCR-w64 (ultima versione)? (S/N)
set /p ansTesseract=
if /I "!ansTesseract!"=="S" (
    echo Scaricando Tesseract-OCR-w64...
    powershell -Command "Invoke-WebRequest -Uri 'https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe' -OutFile 'TesseractInstaller.exe' -UserAgent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'"
    echo Installazione in corso...
    start /wait TesseractInstaller.exe /SILENT
    del TesseractInstaller.exe
    echo Tesseract installato.
) else (
    echo Salto installazione Tesseract.
)
echo.

:: =====================================================
:: 2. Scarica ed installa Ollama
:: =====================================================
echo Vuoi scaricare ed installare Ollama? (S/N)
set /p ansOllama=
if /I "!ansOllama!"=="S" (
    echo Scaricando Ollama...
    powershell -Command "Invoke-WebRequest -Uri 'https://ollama.com/download/OllamaSetup.exe' -OutFile 'OllamaInstaller.exe' -UserAgent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'"
    echo Installazione in corso...
    start /wait OllamaInstaller.exe /S
    del OllamaInstaller.exe
    echo Ollama installato.
    
    :: Chiede se installare il modello llama3.2
    echo Vuoi installare il modello llama3.2? (S/N)
    set /p ansLlama=
    if /I "!ansLlama!"=="S" (
        echo Avvio dell'installazione del modello llama3.2...
        ollama run llama3.2:latest
    ) else (
        echo Salto installazione modello llama3.2.
    )
) else (
    echo Salto installazione Ollama.
)
echo.
echo Operazione completata.
pause
endlocal

