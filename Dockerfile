FROM python:3.10-slim

# Créer le dossier de l'application
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Mettre à jour pip pour éviter les vulnérabilités détectées par safety
RUN pip install --upgrade pip

# Installer les dépendances du projet
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Fixer les permissions sur /app/data
RUN chmod 755 /app/data

# Mettre à jour setuptools pour éviter la vulnérabilité
RUN pip install --upgrade "setuptools>=70.0.0"

# Installer Safety v2 et faire un scan, sans casser la build
RUN pip install "safety<3.0.0" \
 && safety check


# Exposer le port (exemple: Streamlit, FastAPI, etc.)
EXPOSE 8501

# Lancer ton application (à adapter)
CMD ["streamlit", "run", "agent-trading.py"]
