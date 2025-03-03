#!/bin/bash
set -e  # Ferma lo script in caso di errore

export KMP_DUPLICATE_LIB_OK=TRUE

# =====================================================
# 1. Installa Tesseract-OCR tramite Homebrew
# =====================================================
echo "Vuoi installare Tesseract-OCR tramite Homebrew? (S/N)"
read -r ansTesseract
if [[ "$ansTesseract" =~ ^[Ss]$ ]]; then
    echo "Installazione di Tesseract-OCR in corso..."
    brew install tesseract
    echo "Tesseract installato."
else
    echo "Salto installazione Tesseract."
fi

echo
# =====================================================
# 2. Installa Ollama tramite Homebrew
# =====================================================
echo "Vuoi installare Ollama tramite Homebrew? (S/N)"
read -r ansOllama
if [[ "$ansOllama" =~ ^[Ss]$ ]]; then
    echo "Installazione di Ollama in corso..."
    brew install ollama
    echo "Ollama installato."
    
    # Chiede se installare il modello llama3.2
    echo "Vuoi installare il modello llama3.2? (S/N)"
    read -r ansLlama
    if [[ "$ansLlama" =~ ^[Ss]$ ]]; then
        echo "Avvio dell'installazione del modello llama3.2..."
        ollama pull llama3.2:latest
    else
        echo "Salto installazione modello llama3.2."
    fi
else
    echo "Salto installazione Ollama."
fi

echo "Operazione completata."