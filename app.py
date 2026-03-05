"""
Heating Coherence Diagnostic - MVP
Analyse la cohérence du chauffage via capteur intérieur HA + météo Open-Meteo
"""

import io
import json
import warnings
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1. CSV PARSING
# ──────────────────────────────────────────────

def load_and_clean_csv(file, tz: str = "Europe/Paris") -> pd.DataFrame:
    """
    Charge le CSV Home Assistant, filtre l'entité température,
    retourne un df horaire [ts_hour, temp_int].
    """
    df = pd.read_csv(file)

    # Colonnes attendues
    required = {"entity_id", "state", "last_changed"}
    if not required.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes. Attendu: {required}. Trouvé: {set(df.columns)}")

    # Filtrer entités température
    mask = (
        df["entity_id"].str.contains("temperature", case=False, na=False)
    )
    df_temp = df[mask].copy()

    if df_temp.empty:
        raise ValueError(
            "Aucune entité température trouvée. "
            "Vérifiez que entity_id contient 'temperature'."
        )

    # Conversion datetime (ISO 8601, timezone-aware)
    df_temp["last_changed"] = pd.to_datetime(df_temp["last_changed"], utc=True)
    df_temp["last_changed"] = df_temp["last_changed"].dt.tz_convert(tz)

    # Conversion state -> float
    df_temp["state"] = pd.to_numeric(df_temp["state"], errors="coerce")
    df_temp.dropna(subset=["state"], inplace=True)

    # Filtrer valeurs aberrantes (ex: -999 ou +999)
    df_temp = df_temp[(df_temp["state"] > -30) & (df_temp["state"] < 60)]

    if df_temp.empty:
        raise ValueError("Aucune donnée numérique valide après nettoyage.")

    # Agrégation horaire
    df_temp["ts_hour"] = df_temp["last_changed"].dt.floor("h")
    df_hourly = (
        df_temp.groupby("ts_hour")["state"]
        .mean()
        .reset_index()
        .rename(columns={"state": "temp_int"})
    )
    df_hourly.sort_values("ts_hour", inplace=True)

    return df_hourly


# ──────────────────────────────────────────────
# 2. MÉTÉO OPEN-METEO
# ──────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def geocode_city(city: str, country: str) -> tuple[float, float]:
    """Geocode via Open-Meteo geocoding API."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 5, "language": "fr", "format": "json"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        raise ValueError(f"Ville '{city}' introuvable via geocoding.")
    # Préférer le résultat qui matche le pays si possible
    country_upper = country.strip().upper()
    for res in results:
        if res.get("country_code", "").upper() == country_upper:
            return res["latitude"], res["longitude"]
    return results[0]["latitude"], results[0]["longitude"]


@st.cache_data(ttl=3600, show_spinner=False)
def get_outdoor_hourly(
    lat: float, lon: float, start: str, end: str, tz: str
) -> pd.DataFrame:
    """
    Récupère la température extérieure horaire via Open-Meteo.
    start/end: format YYYY-MM-DD
    Retourne df [ts_hour, temp_ext]
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "start_date": start,
        "end_date": end,
        "timezone": tz,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise ValueError("Réponse Open-Meteo vide.")

    df = pd.DataFrame({
        "ts_hour": pd.to_datetime(hourly["time"]),
        "temp_ext": hourly["temperature_2m"],
    })
    # Rendre timezone-aware
    df["ts_hour"] = df["ts_hour"].dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
    df.dropna(subset=["ts_hour"], inplace=True)
    return df


# ──────────────────────────────────────────────
# 3. MERGE & COMPUTE DELTA
# ──────────────────────────────────────────────

def merge_and_compute(df_int: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne intérieur + extérieur sur ts_hour.
    Ajoute colonne delta = temp_int - temp_ext.
    """
    df = pd.merge(df_int, df_ext, on="ts_hour", how="inner")
    df["delta"] = df["temp_int"] - df["temp_ext"]
    df.sort_values("ts_hour", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ──────────────────────────────────────────────
# 4. DIAGNOSTICS
# ──────────────────────────────────────────────

def compute_diagnostics(
    df: pd.DataFrame,
    comfort_min: float = 18.0,
    comfort_max: float = 22.0,
) -> dict:
    """Calcule tous les KPIs et génère le verdict + actions."""
    n = len(df)
    ti = df["temp_int"]
    te = df["temp_ext"]

    # Période
    start = df["ts_hour"].min()
    end = df["ts_hour"].max()

    # Stats intérieur
    hourly_var = ti.diff().abs().mean()
    stability_score = round(ti.std() + hourly_var, 2)

    # Bandes de confort
    pct_under = (ti < comfort_min).mean() * 100
    pct_over = (ti > comfort_max).mean() * 100
    pct_comfort = 100 - pct_under - pct_over

    # Heuristiques cohérence
    issues = []
    actions = []

    mild_outdoor = (te > 10).mean()
    cold_outdoor = (te < 5).mean()
    often_over = (ti > comfort_max).mean()
    often_under = (ti < comfort_min).mean()

    if mild_outdoor > 0.4 and often_over > 0.3:
        issues.append("Surchauffe fréquente malgré extérieur doux")
        actions.append("📉 Réduire la consigne ou ajuster le planning de chauffage aux périodes douces")

    if cold_outdoor > 0.3 and often_under > 0.25:
        issues.append("Sous-chauffe fréquente lors des périodes froides")
        actions.append("🌡️ Augmenter la consigne ou vérifier l'isolation / la puissance du chauffage")

    if stability_score > 3.0:
        issues.append(f"Instabilité thermique élevée (score {stability_score})")
        actions.append("🔧 Vérifier le régulateur/thermostat — variations importantes détectées")

    if not actions:
        actions.append("✅ Confort thermique satisfaisant sur la période")
        actions.append("🗓️ Pensez à réviser le planning saisonnier pour anticiper les changements météo")
        actions.append("📊 Continuez à surveiller la stabilité thermique nocturne")

    # Verdict
    if pct_comfort >= 80 and stability_score < 3.0:
        verdict = "✅ OK"
        verdict_detail = "Confort thermique maîtrisé sur la période analysée."
    else:
        verdict = "⚠️ Points à améliorer"
        details = []
        if pct_comfort < 80:
            details.append(f"seulement {pct_comfort:.1f}% du temps dans la plage de confort")
        if stability_score >= 3.0:
            details.append(f"instabilité thermique (score {stability_score})")
        verdict_detail = "Problèmes détectés : " + ", ".join(details) + "."

    # Day vs Night split
    df["hour_of_day"] = df["ts_hour"].dt.hour
    is_night = (df["hour_of_day"] >= 22) | (df["hour_of_day"] < 6)
    day_mean = df.loc[~is_night, "temp_int"].mean()
    night_mean = df.loc[is_night, "temp_int"].mean()

    return {
        "period_start": str(start),
        "period_end": str(end),
        "nb_points": n,
        "temp_int_min": round(float(ti.min()), 2),
        "temp_int_mean": round(float(ti.mean()), 2),
        "temp_int_median": round(float(ti.median()), 2),
        "temp_int_max": round(float(ti.max()), 2),
        "temp_int_std": round(float(ti.std()), 2),
        "stability_score": stability_score,
        "hourly_variation_mean": round(float(hourly_var), 2),
        "pct_underheating": round(pct_under, 1),
        "pct_overheating": round(pct_over, 1),
        "pct_comfort": round(pct_comfort, 1),
        "comfort_min": comfort_min,
        "comfort_max": comfort_max,
        "day_mean_temp": round(float(day_mean), 2) if not np.isnan(day_mean) else None,
        "night_mean_temp": round(float(night_mean), 2) if not np.isnan(night_mean) else None,
        "issues": issues,
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "actions": actions,
    }


# ──────────────────────────────────────────────
# 5. VISUALISATIONS
# ──────────────────────────────────────────────

def make_charts(df: pd.DataFrame, diag: dict) -> tuple:
    """Génère les 3 figures matplotlib."""
    comfort_min = diag["comfort_min"]
    comfort_max = diag["comfort_max"]
    times = df["ts_hour"]

    # ── Chart 1: temp_int vs temp_ext ──
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(times, df["temp_int"], label="Intérieur", color="#E84545", linewidth=1.5)
    ax1.plot(times, df["temp_ext"], label="Extérieur", color="#4A90D9", linewidth=1.2, alpha=0.8)
    ax1.axhline(comfort_min, color="orange", linestyle="--", linewidth=0.8, alpha=0.7, label=f"Min confort {comfort_min}°C")
    ax1.axhline(comfort_max, color="red", linestyle="--", linewidth=0.8, alpha=0.7, label=f"Max confort {comfort_max}°C")

    # Anomalies
    under = df[df["temp_int"] < comfort_min]
    over = df[df["temp_int"] > comfort_max]
    if not under.empty:
        ax1.scatter(under["ts_hour"], under["temp_int"], color="blue", s=15, zorder=5, label="Sous-chauffe", alpha=0.6)
    if not over.empty:
        ax1.scatter(over["ts_hour"], over["temp_int"], color="darkred", s=15, zorder=5, label="Surchauffe", alpha=0.6)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    plt.xticks(rotation=45)
    ax1.set_ylabel("°C")
    ax1.set_title("Température intérieure vs extérieure")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()

    # ── Chart 2: delta ──
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.fill_between(times, df["delta"], alpha=0.5, color="#5C6BC0", label="Δ int - ext")
    ax2.plot(times, df["delta"], color="#3949AB", linewidth=1)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    plt.xticks(rotation=45)
    ax2.set_ylabel("ΔT (°C)")
    ax2.set_title("Écart thermique (intérieur - extérieur)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()

    # ── Chart 3: histogramme temp_int ──
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.hist(df["temp_int"], bins=30, color="#66BB6A", edgecolor="white", linewidth=0.5)
    ax3.axvline(comfort_min, color="orange", linestyle="--", linewidth=1.5, label=f"Min {comfort_min}°C")
    ax3.axvline(comfort_max, color="red", linestyle="--", linewidth=1.5, label=f"Max {comfort_max}°C")
    ax3.set_xlabel("Température intérieure (°C)")
    ax3.set_ylabel("Heures")
    ax3.set_title("Distribution de la température intérieure")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()

    return fig1, fig2, fig3


# ──────────────────────────────────────────────
# 6. EXPORT
# ──────────────────────────────────────────────

def build_export(df: pd.DataFrame, diag: dict) -> tuple[bytes, bytes]:
    """Retourne (json_bytes, csv_bytes)."""
    json_bytes = json.dumps(diag, ensure_ascii=False, indent=2, default=str).encode("utf-8")

    df_export = df.copy()
    df_export["ts_hour"] = df_export["ts_hour"].astype(str)
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")

    return json_bytes, csv_bytes


# ──────────────────────────────────────────────
# 7. UI STREAMLIT
# ──────────────────────────────────────────────

def render_streamlit_ui():
    st.set_page_config(
        page_title="🏠 Diagnostic Chauffage",
        page_icon="🌡️",
        layout="wide",
    )

    # CSS minimal
    st.markdown("""
    <style>
    .kpi-box {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        border: 1px solid #313244;
    }
    .kpi-val { font-size: 2rem; font-weight: bold; color: #cdd6f4; }
    .kpi-label { font-size: 0.85rem; color: #a6adc8; margin-top: 4px; }
    .verdict-ok { background: #1a3a2a; border-left: 4px solid #40a02b; padding: 12px 16px; border-radius: 6px; }
    .verdict-warn { background: #3a2a1a; border-left: 4px solid #fe640b; padding: 12px 16px; border-radius: 6px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏠 Diagnostic Cohérence Chauffage")
    st.caption("Analyse votre historique Home Assistant × météo Open-Meteo pour diagnostiquer votre chauffage.")

    # ── SIDEBAR ──
    with st.sidebar:
        st.header("⚙️ Paramètres")

        uploaded_file = st.file_uploader(
            "📂 Export CSV Home Assistant",
            type=["csv"],
            help="Exporté depuis Historique HA"
        )

        st.markdown("---")
        location_mode = st.radio("📍 Localisation", ["Ville + Pays", "Coordonnées GPS"])

        if location_mode == "Ville + Pays":
            city = st.text_input("Ville", value="Gien")
            country = st.text_input("Code pays (ex: FR)", value="FR")
            lat_manual, lon_manual = None, None
        else:
            city, country = None, None
            lat_manual = st.number_input("Latitude", value=47.69, format="%.4f")
            lon_manual = st.number_input("Longitude", value=2.63, format="%.4f")

        tz = st.selectbox(
            "🕐 Fuseau horaire",
            ["Europe/Paris", "Europe/London", "Europe/Berlin", "UTC"],
            index=0,
        )

        st.markdown("---")
        st.subheader("🎯 Plage de confort cible")
        comfort_min = st.slider("Température min", 14.0, 20.0, 18.0, 0.5)
        comfort_max = st.slider("Température max", 20.0, 26.0, 22.0, 0.5)

        st.markdown("---")
        show_night = st.toggle("🌙 Afficher comparaison Jour / Nuit", value=True)

        run_btn = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)

    # ── MAIN ──
    if not run_btn:
        st.info("👈 Configurez les paramètres dans la barre latérale, puis cliquez sur **Lancer l'analyse**.")
        with st.expander("ℹ️ Comment exporter le CSV depuis Home Assistant ?"):
            st.markdown("""
1. Aller dans **Historique** (sidebar HA)
2. Sélectionner votre capteur de température
3. Choisir la période souhaitée
4. Cliquer **Télécharger les données** (icône ⬇️)
5. Uploader le fichier `.csv` ici

**Colonnes attendues** : `entity_id`, `state`, `last_changed`
            """)
        return

    if not uploaded_file:
        st.error("⚠️ Veuillez uploader un fichier CSV Home Assistant.")
        return

    # ── PIPELINE ──
    with st.spinner("📊 Chargement et nettoyage du CSV..."):
        try:
            df_int = load_and_clean_csv(uploaded_file, tz=tz)
        except Exception as e:
            st.error(f"❌ Erreur CSV : {e}")
            return

    st.success(f"✅ CSV chargé — {len(df_int)} points horaires • {df_int['ts_hour'].min().strftime('%d/%m/%Y')} → {df_int['ts_hour'].max().strftime('%d/%m/%Y')}")

    # Géocodage si nécessaire
    with st.spinner("🌍 Localisation..."):
        try:
            if location_mode == "Ville + Pays":
                lat, lon = geocode_city(city, country)
                st.caption(f"📍 Coordonnées trouvées : {lat:.4f}, {lon:.4f}")
            else:
                lat, lon = lat_manual, lon_manual
        except Exception as e:
            st.error(f"❌ Erreur géocodage : {e}")
            return

    # Météo
    start_date = df_int["ts_hour"].min().strftime("%Y-%m-%d")
    end_date = df_int["ts_hour"].max().strftime("%Y-%m-%d")

    with st.spinner("🌤️ Récupération météo Open-Meteo..."):
        try:
            df_ext = get_outdoor_hourly(lat, lon, start_date, end_date, tz)
        except Exception as e:
            st.error(f"❌ Erreur API météo : {e}. Vérifiez votre connexion ou réessayez.")
            return

    # Merge
    df_final = merge_and_compute(df_int, df_ext)
    if df_final.empty:
        st.error("❌ Aucune donnée après fusion intérieur/extérieur. Vérifiez les dates.")
        return

    # Diagnostics
    diag = compute_diagnostics(df_final, comfort_min=comfort_min, comfort_max=comfort_max)

    # ── AFFICHAGE RÉSULTATS ──
    st.markdown("---")
    st.header("📈 Résultats de l'analyse")

    # KPIs
    cols = st.columns(6)
    kpis = [
        ("🌡️ Moy. int.", f"{diag['temp_int_mean']}°C"),
        ("📉 Min", f"{diag['temp_int_min']}°C"),
        ("📈 Max", f"{diag['temp_int_max']}°C"),
        ("📊 Stabilité", str(diag["stability_score"])),
        ("🟢 Confort", f"{diag['pct_comfort']}%"),
        ("🔵 Nb points", str(diag["nb_points"])),
    ]
    for col, (label, val) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-val">{val}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # Jour / Nuit
    if show_night and diag["day_mean_temp"] and diag["night_mean_temp"]:
        c1, c2 = st.columns(2)
        c1.metric("☀️ Temp. moyenne jour (6h-22h)", f"{diag['day_mean_temp']}°C")
        c2.metric("🌙 Temp. moyenne nuit (22h-6h)", f"{diag['night_mean_temp']}°C",
                  delta=f"{diag['night_mean_temp'] - diag['day_mean_temp']:.1f}°C vs jour")

    # Bandes de confort
    st.markdown("---")
    st.subheader("🎯 Répartition du temps en plage de confort")
    c1, c2, c3 = st.columns(3)
    c1.metric("🔵 Sous-chauffe (<{:.0f}°C)".format(comfort_min), f"{diag['pct_underheating']}%")
    c2.metric("✅ Confort ({:.0f}–{:.0f}°C)".format(comfort_min, comfort_max), f"{diag['pct_comfort']}%")
    c3.metric("🔴 Surchauffe (>{:.0f}°C)".format(comfort_max), f"{diag['pct_overheating']}%")

    # Charts
    st.markdown("---")
    st.subheader("📉 Visualisations")
    fig1, fig2, fig3 = make_charts(df_final, diag)

    st.pyplot(fig1)
    st.pyplot(fig2)
    col_hist, _ = st.columns([1, 1])
    with col_hist:
        st.pyplot(fig3)

    plt.close("all")

    # ── PRESCRIPTION ──
    st.markdown("---")
    st.subheader("🩺 Prescription")

    verdict_class = "verdict-ok" if "OK" in diag["verdict"] else "verdict-warn"
    st.markdown(f"""
    <div class="{verdict_class}">
        <b>{diag['verdict']}</b> — {diag['verdict_detail']}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    if diag["issues"]:
        st.markdown("**Points à améliorer :**")
        for issue in diag["issues"]:
            st.markdown(f"- ⚠️ {issue}")

    st.markdown("**Actions rapides :**")
    for action in diag["actions"]:
        st.markdown(f"- {action}")

    # ── EXPORT ──
    st.markdown("---")
    st.subheader("📥 Télécharger le rapport")
    json_bytes, csv_bytes = build_export(df_final, diag)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Rapport JSON (KPIs + verdict)",
            data=json_bytes,
            file_name="diagnostic_chauffage.json",
            mime="application/json",
        )
    with col_dl2:
        st.download_button(
            "⬇️ Dataset enrichi CSV",
            data=csv_bytes,
            file_name="dataset_enrichi.csv",
            mime="text/csv",
        )


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    render_streamlit_ui()
