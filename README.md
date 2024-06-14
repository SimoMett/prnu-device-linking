# Istruzioni d'uso
- Modificare i parametri nel modulo [params.py](params.py)
- Eseguire [fetch-dataset-py](fetch-dataset.py) per scaricare il dataset originario da https://lesc.dinfo.unifi.it/FloreView/
- Eseguire [generate-video-sequences.py](generate-video-sequences.py) per generare le sequenze video
- Per ogni video generato eseguire lo script [prnu-extract-fingerprints.py](prnu_extract_fingerprints.py) usando la sintassi `python prnu_extract_fingerprints.py <file.mp4>`.  
Per eseguire la procedura su tutte le sequenze è conveniente spostare tutti i video in una cartella ed eseguire la procedura con `python prnu_extract_fingerprints.py cartella/*.mp4`.
Questo modulo genererà delle cartelle in cui saranno salvati tutti i risultati come la Cross-Correlation, Peak to Correlation Energy e altre statistiche in formato .csv e raw con [pickle](https://docs.python.org/3/library/pickle.html). 
