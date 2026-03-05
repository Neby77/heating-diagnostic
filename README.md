# 🏠 Heating Coherence Diagnostic

Analyse la cohérence du chauffage via capteur intérieur Home Assistant + météo Open-Meteo.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Utilisation

1. Exporter le CSV depuis l'historique Home Assistant (colonnes : `entity_id`, `state`, `last_changed`)
2. Uploader le fichier dans l'app
3. Configurer la localisation et la plage de confort
4. Lancer l'analyse
