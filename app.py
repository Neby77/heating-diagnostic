"""
Heating Coherence Diagnostic - MVP
Analyse la cohérence du chauffage via capteur intérieur HA + météo Open-Meteo
"""

import base64
import io
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    """Génère les 3 figures Plotly interactives."""
    comfort_min = diag["comfort_min"]
    comfort_max = diag["comfort_max"]
    times = df["ts_hour"]

    chart_layout = dict(
        template="plotly_white",
        paper_bgcolor="rgba(255,255,255,0.75)",
        plot_bgcolor="rgba(255,255,255,0.6)",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=40, b=40),
        font=dict(color="#1a1a2e"),
    )

    # ── Chart 1: temp_int vs temp_ext ──
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=df["temp_int"], name="Intérieur", line=dict(color="#E84545", width=2),
                              hovertemplate="%{y:.1f}°C"))
    fig1.add_trace(go.Scatter(x=times, y=df["temp_ext"], name="Extérieur", line=dict(color="#4A90D9", width=1.5),
                              opacity=0.8, hovertemplate="%{y:.1f}°C"))
    fig1.add_hline(y=comfort_min, line_dash="dash", line_color="orange", opacity=0.7,
                   annotation_text=f"Min {comfort_min}°C", annotation_position="top left")
    fig1.add_hline(y=comfort_max, line_dash="dash", line_color="red", opacity=0.7,
                   annotation_text=f"Max {comfort_max}°C", annotation_position="top left")

    under = df[df["temp_int"] < comfort_min]
    over = df[df["temp_int"] > comfort_max]
    if not under.empty:
        fig1.add_trace(go.Scatter(x=under["ts_hour"], y=under["temp_int"], mode="markers", name="Sous-chauffe",
                                  marker=dict(color="blue", size=5, opacity=0.6), hovertemplate="%{y:.1f}°C"))
    if not over.empty:
        fig1.add_trace(go.Scatter(x=over["ts_hour"], y=over["temp_int"], mode="markers", name="Surchauffe",
                                  marker=dict(color="darkred", size=5, opacity=0.6), hovertemplate="%{y:.1f}°C"))

    fig1.update_layout(**chart_layout, title="Température intérieure vs extérieure",
                       yaxis_title="°C", height=400, xaxis=dict(tickformat="%d/%m %Hh"))

    # ── Chart 2: delta ──
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=df["delta"], fill="tozeroy", name="Δ int - ext",
                              line=dict(color="#3949AB", width=1.5), fillcolor="rgba(92,107,192,0.4)",
                              hovertemplate="%{y:.1f}°C"))
    fig2.add_hline(y=0, line_color="black", line_width=0.8)
    fig2.update_layout(**chart_layout, title="Écart thermique (intérieur - extérieur)",
                       yaxis_title="ΔT (°C)", height=300, xaxis=dict(tickformat="%d/%m %Hh"))

    # ── Chart 3: histogramme temp_int ──
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(x=df["temp_int"], nbinsx=30, marker_color="#66BB6A",
                                hovertemplate="Temp: %{x:.1f}°C<br>Count: %{y}"))
    fig3.add_vline(x=comfort_min, line_dash="dash", line_color="orange", line_width=2,
                   annotation_text=f"Min {comfort_min}°C")
    fig3.add_vline(x=comfort_max, line_dash="dash", line_color="red", line_width=2,
                   annotation_text=f"Max {comfort_max}°C")
    fig3.update_layout(**chart_layout, title="Distribution de la température intérieure",
                       xaxis_title="Température intérieure (°C)", yaxis_title="Heures", height=400)

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

    # Background image + CSS
    bg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ImageFond.png")
    bg_css = ""
    if os.path.exists(bg_path):
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
        bg_css = (
            '[data-testid="stAppViewContainer"] {'
            f'  background-image: linear-gradient(rgba(255,255,255,0.7), rgba(255,255,255,0.7)), url("data:image/png;base64,{bg_b64}");'
            '  background-size: cover;'
            '  background-position: center;'
            '  background-attachment: fixed;'
            '}'
            '[data-testid="stSidebar"] {'
            '  background: rgba(255, 255, 255, 0.88);'
            '}'
            '[data-testid="stHeader"] {'
            '  background: transparent;'
            '}'
            'html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * {'
            '  color: #1a1a2e;'
            '}'
            'h1, h2, h3, h4, h5, h6, .stMarkdown p, .stMarkdown li, label, .stRadio label span,'
            '[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaption"] {'
            '  color: #1a1a2e !important;'
            '}'
            '[data-testid="stMetricDelta"] { color: #555 !important; }'
        )

    st.markdown(
        "<style>"
        + bg_css
        + ".kpi-box {"
        "  background: rgba(255, 255, 255, 0.85);"
        "  border-radius: 10px;"
        "  padding: 16px 20px;"
        "  text-align: center;"
        "  border: 1px solid #ddd;"
        "  backdrop-filter: blur(4px);"
        "}"
        ".kpi-val { font-size: 2rem; font-weight: bold; color: #1a1a2e !important; }"
        ".kpi-label { font-size: 0.85rem; color: #555 !important; margin-top: 4px; }"
        ".verdict-ok { background: rgba(220, 255, 220, 0.85); border-left: 4px solid #40a02b; padding: 12px 16px; border-radius: 6px; color: #1a3a2a !important; }"
        ".verdict-warn { background: rgba(255, 235, 210, 0.85); border-left: 4px solid #fe640b; padding: 12px 16px; border-radius: 6px; color: #3a2a1a !important; }"
        "</style>",
        unsafe_allow_html=True,
    )

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

    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    col_hist, _ = st.columns([1, 1])
    with col_hist:
        st.plotly_chart(fig3, use_container_width=True)

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
