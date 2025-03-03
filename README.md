<table border="0">
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/1f43e592-a560-4e46-90c9-03322d3e233d" width="90" height="90" /> 
    </td>
    <td>
       <h1><b>NbMultiRag - un GPT tutto in LOCALE</b></h1>
    </td>
  </tr>
</table>
 
###  ---------------- I tuoi dati rimangono sul tuo computer ---------------   
## ITALIANO  
Un framework in Italiano ed Inglese, che permette di chattare con i propri documenti in RAG, anche multimediali (audio, video, immagini e OCR).  
Il framework è una GUI per chattare con un modello GPT scaricato da OLLAMA, si consiglia LLAMA 3.2 (2Gb) perfettamente performante anche su macchine  
medie.  
Inoltre, bisogna installare il software Tesseract, per il riconoscimento OCR, si consiglia di scegliere le lingue italiano ed inglese durante l'istallazione.  
NBMultiRag, permette di:
1) Chattare col modello senza RAG.
2) Creare dei workspace personalizzati e configurabili.
3) Indicizzare per il RAG una cartella di documenti di varia tipologia.
4) Interrogare il sistema, che provvederà a trascrivere gli audio e video presenti nei documenti, fare OCR sulle immagini e pure descrivere 10 frame equamente distributi nel video.
5) Si può anche inviare nella chat, tramite upload, un singolo file alla volta ed il sistema provvederà a descriverlo.
6) Al sistema serve la connessione alla rete Internet solo al lancio per scaricare i modelli da HuggingFace, poi si può anche sconnettere il computer.

## ISTRUZIONI PER SISTEMI WINDOWS  

1) Lanciare il file install.bat
2) Nel framework seguire gli avvisi (es. scaricare un modello).
3) Creare un workspace
4) Scegliere un embedder (di default c'è bert-base-italian-uncased per l'Italiano e bert-base-uncased per l'Inglese.
5) Aggiungere una cartella che contiene i documenti da indicizzare.
6) Aggiornare l'indice.
7) CHATTARE
8) Il programma scarica nella C:\Users\YOUR_USER_NAME\.cache\huggingface\hub i file: models--bert-base-uncased, models--dbmdz--bert-base-italian-uncased, models--Salesforce--blip-image-captioning-base

## Per chi vuole usare Python:  
1) crea la cartella nbmultirag
2) copia tutto il contenuto di questo repository, eccetto il file ZIP che contiene il compilato.
3) crea un enviroment Python: python -m venv nbmultirag
4) Attivare l'enviroment (per Windows: nbmultirag/Scripts/activate)
5) pip install -r requirements
6) streamlit run nbmultirag.py






