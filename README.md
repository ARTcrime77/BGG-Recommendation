# BGG ML-Empfehlungssystem

Ein Machine Learning-basiertes Empfehlungssystem für Brettspiele, das BoardGameGeek (BGG) Daten verwendet.

## 🎯 Features

- **Personalisierte Empfehlungen** basierend auf Ihrer BGG-Sammlung und Spielstatistiken
- **Machine Learning** mit k-Nearest Neighbors und Feature Engineering
- **Echte BGG Top 500** durch Web-Scraping (mit Fallback)
- **Intelligentes Caching** zur Reduzierung von API-Calls
- **Umfassende Features**: Kategorien, Mechaniken, Autoren, Illustratoren, Verlage
- **Duplikat-Erkennung** auf mehreren Ebenen

## 📁 Projektstruktur

```
bgg-recommender/
├── config.py              # Konfiguration und Parameter
├── data_loader.py          # Datenlade- und Cache-Funktionen
├── ml_engine.py           # Machine Learning Engine
├── main.py                # Hauptklasse und Programm
├── requirements.txt       # Python Dependencies
├── README.md             # Diese Datei
└── bgg_cache/            # Cache-Verzeichnis (wird automatisch erstellt)
    ├── top500_games.json      # BGG Top 500 Cache
    └── game_details.json     # Spieldetails Cache
```

## 🚀 Installation

### 1. Virtuelle Umgebung erstellen (macOS/Linux)
```bash
# Projektordner erstellen
mkdir bgg-recommender && cd bgg-recommender

# Virtuelle Umgebung erstellen und aktivieren
python3 -m venv BGG_Reco
source BGG_Reco/bin/activate

# Dependencies installieren
pip install -r requirements.txt
```

### 2. Code-Dateien erstellen
Erstellen Sie die Dateien mit dem bereitgestellten Code:
- `config.py`
- `data_loader.py`
- `ml_engine.py`
- `main.py`
- `requirements.txt`

### 3. Benutzername anpassen
In `main.py`, Zeile 196:
```python
username = "IHR_BGG_NUTZERNAME"  # Hier Ihren BGG-Nutzernamen eingeben
```

## 📊 Verwendung

```bash
# Programm starten
python main.py
```

### Erste Ausführung
- System lädt BGG Top 500 (kann 10-15 Minuten dauern)
- Lädt Ihre BGG-Sammlung und Spielstatistiken
- Erstellt Feature-Matrix und trainiert ML-Modell
- Generiert personalisierte Empfehlungen

### Folgeausführungen
- Fragt ob Cache verwendet werden soll
- Deutlich schneller (30-60 Sekunden)

## 🔧 Konfiguration

In `config.py` können Sie anpassen:

```python
# Cache-Einstellungen
CACHE_MAX_AGE_DAYS = 7              # Cache-Alter in Tagen

# ML-Parameter
MIN_FEATURE_FREQUENCY = 2           # Min. Häufigkeit für Autoren/Verlage
MAX_NEIGHBORS = 20                  # k-NN Nachbarn
SIMILARITY_METRIC = 'cosine'        # Ähnlichkeits-Metrik

# Gewichtung
RATING_WEIGHT_MULTIPLIER = 2        # Gewichtung für Bewertungen
DEFAULT_NUM_RECOMMENDATIONS = 10    # Anzahl Empfehlungen
```

## 🤖 Machine Learning Features

### Numerische Features (7)
1. `avg_rating` - Durchschnittsbewertung
2. `complexity` - Komplexität (1-5)
3. `min_players` - Min. Spieleranzahl
4. `max_players` - Max. Spieleranzahl
5. `log(playing_time)` - Log-transformierte Spielzeit
6. `game_age` - Alter des Spiels
7. `capped_age` - Gekapptes Alter (max. 25 Jahre)

### Kategorische Features (One-Hot Encoded)
- **Kategorien** (Strategy, Family, Thematic, etc.)
- **Mechaniken** (Engine Building, Worker Placement, etc.)
- **Autoren** (nur häufige mit ≥2 Spielen)
- **Illustratoren** (nur häufige mit ≥2 Spielen)
- **Verlage** (nur häufige mit ≥2 Spielen)

### Gewichtungsformel
```python
Gewicht = (Bewertung - 5) × 2 + log(Spielanzahl + 1)
```

## 📝 Beispiel-Ausgabe

```
🎯 Top 10 ML-Empfehlungen aus den BGG Top 500:
======================================================================

 1. Terraforming Mars (BGG Rang #3) - 2016
    ⭐ Rating: 8.4
    🧩 Komplexität: 3.2/5
    📂 Kategorien: Economic, Science Fiction, Territory Building
    ⚙️ Mechaniken: Card Drafting, Hand Management, Tile Placement
    ✍️ Autoren: Jacob Fryxelius
    🎨 Illustratoren: Isaac Fryxelius
    🎯 Ähnlichkeit: 87.3%
```

## 🔍 Problembehandlung

### "Keine Sammlung gefunden"
- Überprüfen Sie den BGG-Nutzernamen
- Stellen Sie sicher, dass die Sammlung öffentlich ist

### "Scraping-Fehler"
- System verwendet automatisch Fallback-Liste
- Funktioniert auch ohne Live-Scraping

### "Keine Empfehlungen"
- Benötigt mindestens ein paar bewertete Spiele
- Spiele müssen Details in der BGG-API haben

## 🛠️ Erweiterungsmöglichkeiten

1. **Mehr ML-Algorithmen**: Collaborative Filtering, Matrix Factorization
2. **Deep Learning**: Neural Networks für komplexere Muster
3. **Hybrid-Ansätze**: Kombination verschiedener Techniken
4. **Web-Interface**: Flask/Django Frontend
5. **Echte Top 500**: Vollständiges Web-Scraping
6. **Mehr Features**: Themen, Altersempfehlungen, Spielzeit-Präferenzen

## 📄 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Respektieren Sie die BGG-Nutzungsbedingungen und verwenden Sie angemessene Delays bei API-Calls.

## 🤝 Beitragen

Verbesserungen und Erweiterungen sind willkommen! Erstellen Sie einen Pull Request oder öffnen Sie ein Issue.

---

**Viel Spaß beim Entdecken neuer Brettspiele! 🎲**