import yfinance as yf
import os
import pandas as pd
from datetime import datetime
import hashlib
from pathlib import Path

def sha256sum(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# Dictionnaire des tickers Yahoo Finance du CAC 40
CAC40_TICKERS = {
    "Air Liquide": "AI.PA",
    "Airbus": "AIR.PA",
    "ArcelorMittal": "MT.AS",
    "AXA": "CS.PA",
    "BNP Paribas": "BNP.PA",
    "Bouygues": "EN.PA",
    "Capgemini": "CAP.PA",
    "Carrefour": "CA.PA",
    "Crédit Agricole": "ACA.PA",
    "Danone": "BN.PA",
    "Dassault Systèmes": "DSY.PA",
    "Engie": "ENGI.PA",
    "EssilorLuxottica": "EL.PA",
    "Hermès": "RMS.PA",
    "Kering": "KER.PA",
    "Legrand": "LR.PA",
    "L'Oréal": "OR.PA",
    "LVMH": "MC.PA",
    "Michelin": "ML.PA",
    "Orange": "ORA.PA",
    "Pernod Ricard": "RI.PA",
    "Publicis": "PUB.PA",
    "Renault": "RNO.PA",
    "Safran": "SAF.PA",
    "Saint-Gobain": "SGO.PA",
    "Sanofi": "SAN.PA",
    "Schneider Electric": "SU.PA",
    "Société Générale": "GLE.PA",
    "Thales": "HO.PA",
    "TotalEnergies": "TTE.PA",
    "Veolia": "VIE.PA",
    "Vinci": "DG.PA",
    "Vivendi": "VIV.PA",
    "Worldline": "WLN.PA",
}

# Crée le dossier 'data' s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Dates
start_date = "2015-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Téléchargement et nettoyage des données
for name, ticker in CAC40_TICKERS.items():
    print(f"Téléchargement de {name} ({ticker})...")
    try:
        # Télécharger les données avec yfinance
        data = yf.download(ticker, start=start_date, end=end_date)

        if not data.empty:
            # Réinitialiser les indices
            data.reset_index(inplace=True)

            # Ne conserver que les colonnes utiles
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

            # Écrire dans un buffer pour supprimer la 2e ligne (celle avec les tickers en double)
            from io import StringIO
            buffer = StringIO()
            data.to_csv(buffer, index=False)
            buffer.seek(0)

            # Lire les lignes, enlever la 2e ligne
            lines = buffer.readlines()
            if len(lines) > 1 and lines[1].startswith(','):
                del lines[1]

            # Construction du chemin sécurisé
            data_path = Path("data").resolve()
            csv_path = (data_path / f"{ticker}.csv").resolve()

            # Vérification de sécurité
            if not str(csv_path).startswith(str(data_path)):
                raise ValueError("Chemin non autorisé en dehors du dossier data")

            # Écriture sécurisée du fichier
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                f.writelines(lines)

            # Calcul et sauvegarde du hash
            hash = sha256sum(csv_path)
            hash_path = csv_path.with_suffix(".hash")
            with open(hash_path, "w") as h:
                h.write(hash)

            print(f"✅ Données de {name} sauvegardées sous {csv_path}")
            print(f"🔐 SHA256 : {hash}")

        else:
            print(f"⚠️ Aucune donnée reçue pour {ticker}")
    except Exception as e:
        print(f"❌ Erreur pour {ticker}: {e}")

print("✅ Téléchargement et nettoyage terminé.")
