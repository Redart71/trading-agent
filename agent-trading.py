import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import html
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.graph_objects as go
from pathlib import Path
import hashlib
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

st.set_page_config(page_title="Agent de Trading CAC 40", layout="wide")


# Chargement de la configuration
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Instanciation de l'authentificateur
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Vérification si l'utilisateur est déjà authentifié via la session
if 'authentication_status' in st.session_state and st.session_state['authentication_status'] is True:
    # L'utilisateur est déjà connecté
    name = st.session_state['name']
    username = st.session_state['username']
    st.sidebar.success(f"Bienvenue {name} 👋")
    authenticator.logout("Se déconnecter", "sidebar")
else:
    # L'utilisateur n'est pas encore authentifié, nous demandons la connexion
    login_result = authenticator.login()

    if login_result is None:
        st.warning("Veuillez entrer vos identifiants.")
        st.stop()  # Arrêter le script si aucun login n'est effectué
    else:
        name, authentication_status, username = login_result

        if authentication_status is False:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
            st.stop()
        elif authentication_status is None:
            st.warning("Veuillez entrer vos identifiants.")
            st.stop()
        else:
            # L'utilisateur est authentifié avec succès
            st.sidebar.success(f"Bienvenue {name} 👋")
            authenticator.logout("Se déconnecter", "sidebar")



def sha256sum(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = Path("data").resolve()

ALL_CAC40_TICKERS = {
    "Air Liquide": "AI.PA", "Airbus": "AIR.PA", "ArcelorMittal": "MT.AS",
    "AXA": "CS.PA", "BNP Paribas": "BNP.PA", "Bouygues": "EN.PA",
    "Capgemini": "CAP.PA", "Carrefour": "CA.PA", "Crédit Agricole": "ACA.PA",
    "Danone": "BN.PA", "Dassault Systèmes": "DSY.PA", "Engie": "ENGI.PA",
    "EssilorLuxottica": "EL.PA", "Hermès": "RMS.PA", "Kering": "KER.PA",
    "Legrand": "LR.PA", "L'Oréal": "OR.PA", "LVMH": "MC.PA", "Michelin": "ML.PA",
    "Orange": "ORA.PA", "Pernod Ricard": "RI.PA", "PSA Group": "STLA.PA",
    "Publicis": "PUB.PA", "Renault": "RNO.PA", "Safran": "SAF.PA",
    "Saint-Gobain": "SGO.PA", "Sanofi": "SAN.PA", "Schneider Electric": "SU.PA",
    "Société Générale": "GLE.PA", "STMicroelectronics": "STM.PA",
    "Thales": "HO.PA", "TotalEnergies": "TTE.PA", "Unibail-Rodamco-Westfield": "URW.AS",
    "Veolia": "VIE.PA", "Vinci": "DG.PA", "Vivendi": "VIV.PA", "Worldline": "WLN.PA",
}

available_files = os.listdir("data")
available_tickers = {name: ticker for name, ticker in ALL_CAC40_TICKERS.items() if f"{ticker}.csv" in available_files}

@st.cache_data
def load_data():
    data = {}
    for name, ticker in available_tickers.items():
        try:
            # Construction sécurisée des chemins
            file_path = (DATA_DIR / f"{ticker}.csv").resolve()
            hash_path = file_path.with_suffix(".hash")

            # Vérification que le fichier est bien dans le dossier 'data/'
            if not str(file_path).startswith(str(DATA_DIR)):
                raise ValueError("Tentative d'accès non autorisée en dehors de /data")

            # Vérification d'intégrité
            if hash_path.exists():
                with open(hash_path, "r") as h:
                    saved_hash = h.read().strip()
                current_hash = sha256sum(file_path)

                if saved_hash != current_hash:
                    st.error(f"⚠️ Fichier {ticker}.csv modifié ou corrompu (hash invalide).")
                    continue  # Ne pas charger ce fichier
            else:
                st.warning(f"🔍 Pas de fichier .hash pour {ticker}.csv, intégrité non vérifiée.")

            # Chargement du fichier
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

            # Sécurité contre empoisonnement : validation des colonnes et des types
            required_columns = {"Open", "High", "Low", "Close", "Volume"}
            if not required_columns.issubset(df.columns):
                st.error(f"❌ Fichier {ticker}.csv invalide : colonnes manquantes.")
                continue

            if not all(np.issubdtype(df[col].dtype, np.number) for col in required_columns):
                st.error(f"❌ Données non numériques détectées dans {ticker}.csv")
                continue

            if (df[list(required_columns)] < 0).any().any():
                st.error(f"❌ Données invalides (valeurs négatives) dans {ticker}.csv")
                continue

            if df.empty:
                st.warning(f"⚠️ Le fichier {ticker}.csv est vide.")
                continue

            data[ticker] = df

        except Exception as e:
            st.warning(f"Erreur lors de la lecture de {ticker}.csv : {e}")

    return data



def add_indicators(df):
    df = df.copy()
    df["RSI"] = RSIIndicator(close=df["Close"]).rsi()
    df["MACD"] = MACD(close=df["Close"]).macd_diff()
    df["SMA"] = SMAIndicator(close=df["Close"], window=20).sma_indicator()
    df["EMA"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()
    df["BB_bbm"] = BollingerBands(close=df["Close"]).bollinger_mavg()
    df["STOCH"] = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"]).stoch()
    df["OBV"] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    return df.dropna()

def generate_model(df):
    df = df.copy()

    delta = df["Close"].pct_change(periods=5).shift(-5)
    df["Target"] = 0
    df.loc[delta > 0.005, "Target"] = 1
    df.loc[delta < -0.005, "Target"] = -1

    df = df.dropna()

    if df.shape[0] < 100:
        return None, None, None, "Pas assez de données."

    features = ["RSI", "MACD", "SMA", "EMA", "BB_bbm", "STOCH", "OBV"]
    X = df[features]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2
    )

    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5, 
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    return model, X, df, report

def backtest_strategy(df, model, X):
    df = df.copy()
    df = df.iloc[-len(X):]
    df["Prediction"] = model.predict(X)
    df["Retours"] = df["Close"].pct_change()
    df["Strat"] = df["Prediction"].shift(1) * df["Retours"]
    df["Cumul_Strat"] = (1 + df["Strat"]).cumprod()
    df["Cumul_BuyHold"] = (1 + df["Retours"]).cumprod()
    return df

st.title("Agent de Trading sur le CAC 40")
data = load_data()

if not data:
    st.error("Aucune donnée disponible.")
else:
    selected_name = st.selectbox("Choisissez une entreprise :", list(available_tickers.keys()))
    selected_ticker = available_tickers[selected_name]
    df = data[selected_ticker]

    if not df.empty:
        df = add_indicators(df)
        model, X, df, report = generate_model(df)

        if model is None:
            st.warning(report)
        else:
            st.subheader(f"Recommandation pour {selected_name} ({selected_ticker})")
            latest = X.iloc[[-1]]
            # Validation anti-évasion : détecter NaN ou outliers extrêmes dans la dernière ligne
            if latest.isnull().values.any():
                st.error("⚠️ Valeurs manquantes détectées dans les entrées du modèle.")
                st.stop()

            # Détection de valeurs anormales (z-score > 5 par exemple)
            from scipy.stats import zscore
            z_scores = np.abs(zscore(latest))
            if (z_scores > 5).any():
                st.error("⚠️ Entrée suspecte détectée (valeur extrême). Prédiction bloquée.")
                st.stop()

            prediction = model.predict(latest)[0]
            # Calcul de la probabilité
            prediction_proba = model.predict_proba(latest)[0]
            prediction_classes = model.classes_
            certainty = prediction_proba[list(prediction_classes).index(prediction)]

            # Texte de la prédiction
            prediction_text = (
                "Acheter" if prediction == 1
                else "Vendre" if prediction == -1
                else "Neutre"
            )
            # Echappement du texte pour éviter les attaques XSS
            safe_prediction_text = html.escape(prediction_text)
            # Couleur du badge
            badge_color = (
                "#28a745" if certainty > 0.7 else
                "#ffc107" if certainty > 0.4 else
                "#dc3545"
            )

            # Carte centrale de prédiction à la manière du design Tailwind avec alignement et icône
            st.markdown(f"""
            <div style='
                display: flex;
                justify-content: center;
                align-items: center;
                margin: 2em 0;
            '>
                <div style='
                    position: relative;
                    background-color: white;
                    border: 1px solid #e5e7eb;
                    border-radius: 1rem;
                    padding: 2rem;
                    width: 100%;
                    max-width: 500px;
                    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
                    font-family: "Segoe UI", sans-serif;
                '>
                    <div style='
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                        margin-bottom: 0.75rem;
                    '>
                        <p style='
                            color: #6b7280;
                            font-size: 0.875rem;
                            margin: 0;
                        '>Prédiction</p>
                        <div style='
                            background-color: {badge_color};
                            color: white;
                            font-size: 0.75rem;
                            font-weight: 600;
                            padding: 0.25rem 0.5rem;
                            border-radius: 0.5rem;
                            display: flex;
                            align-items: center;
                            gap: 0.25rem;
                        '>
                            Certitude : {certainty:.1%}
                        </div>
                    </div>
                    <h2 style='
                        font-size: 2rem;
                        font-weight: 600;
                        margin: 0 0 1rem 0;
                        color: #111827;
                        display: flex;
                        align-items: center;
                        gap: 0.5rem;
                    '>
                        {"📈" if prediction == 1 else "📉" if prediction == -1 else "⏸️"} {safe_prediction_text}
                    </h2>
                    <div style='
                        font-size: 0.875rem;
                        color: #6b7280;
                    '>
                        Basé sur l'analyse des indicateurs techniques.
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


            result_df = backtest_strategy(df, model, X)
            st.subheader("Analyse Graphique")

            tabs = st.tabs(["🔁 Backtest", "📊 RSI", "📈 MACD", "📉 Moyennes", "📦 Bollinger"])

            with tabs[0]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result_df.index, y=result_df["Cumul_Strat"], mode='lines', name='Stratégie'))
                fig.add_trace(go.Scatter(x=result_df.index, y=result_df["Cumul_BuyHold"], mode='lines', name='Buy & Hold'))
                fig.update_layout(title="Backtest Performance", xaxis_title="Date", yaxis_title="Cumulatif")
                st.plotly_chart(fig, use_container_width=True)

            with tabs[1]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name='RSI'))
                fig.update_layout(title="RSI (Relative Strength Index)", yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)

            with tabs[2]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], mode='lines', name='MACD'))
                fig.update_layout(title="MACD")
                st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df["SMA"], mode='lines', name='SMA 20'))
                fig.add_trace(go.Scatter(x=df.index, y=df["EMA"], mode='lines', name='EMA 50'))
                fig.update_layout(title="Prix avec Moyennes (SMA & EMA)")
                st.plotly_chart(fig, use_container_width=True)

            with tabs[4]:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close'))
                fig.add_trace(go.Scatter(x=df.index, y=df["BB_bbm"], mode='lines', name='Bollinger Moyen'))
                fig.update_layout(title="Bandes de Bollinger")
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Données récentes")
            st.dataframe(df.tail())
    else:
        st.warning("Les données sont vides.")
