# BGG ML-Empfehlungssystem

Ein Machine Learning-basiertes Empfehlungssystem für Brettspiele, das BoardGameGeek (BGG) Daten verwendet.

## 🎯 Features

- **Personalisierte Empfehlungen** basierend auf Ihrer BGG-Sammlung und Spielstatistiken
- **Machine Learning** mit k-Nearest Neighbors und erweiterten Feature Engineering
- **Non-lineare Bewertungsgewichtung** - höhere Bewertungen haben exponentiell mehr Einfluss
- **Garantierte Empfehlungen** - mindestens 10 Empfehlungen auch bei umfangreichen Sammlungen
- **Adaptive Filterung** - automatische Erweiterung der Suchkriterien bei Bedarf
- **Echte BGG Top 1000** durch Web-Scraping (mit Fallback)
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
    ├── top_games.json        # BGG Top 1000 Cache
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
In `main.py`, Zeile 258:
```python
username = "IHR_BGG_NUTZERNAME"  # Hier Ihren BGG-Nutzernamen eingeben
```

## 📊 Verwendung

```bash
# Programm starten
python main.py
```

### Erste Ausführung
- System lädt BGG Top 1000 (kann 15-20 Minuten dauern)
- Lädt Ihre BGG-Sammlung und Spielstatistiken
- Erstellt erweiterte Feature-Matrix und trainiert ML-Modell
- Generiert mindestens 10 personalisierte Empfehlungen

### Folgeausführungen
- Fragt ob Cache verwendet werden soll
- Deutlich schneller (30-60 Sekunden)
- Automatische Erweiterung der Suche falls zu wenig Empfehlungen

## 🔧 Konfiguration

In `config.py` können Sie anpassen:

```python
# Cache-Einstellungen
CACHE_MAX_AGE_DAYS = 7              # Cache-Alter in Tagen
TARGET_TOP_GAMES = 1000             # Ziel-Anzahl Top-Spiele

# ML-Parameter
MIN_FEATURE_FREQUENCY = 2           # Min. Häufigkeit für Autoren/Verlage
MAX_NEIGHBORS = 20                  # k-NN Nachbarn
SIMILARITY_METRIC = 'cosine'        # Ähnlichkeits-Metrik

# Non-lineare Bewertungsgewichtung
RATING_WEIGHTING = {
    'use_nonlinear': True,          # Aktiviere non-lineare Gewichtung
    'exponent': 2.5,                # Potenz für exponentielles Wachstum
    'threshold': 6.0,               # Schwellwert für verstärkte Gewichtung
    'amplification_factor': 1.5,    # Verstärkungsfaktor für hohe Bewertungen
    'min_rating': 5.0               # Mindestbewertung für Gewichtung
}

# Grundeinstellungen
DEFAULT_NUM_RECOMMENDATIONS = 20    # Anzahl Empfehlungen
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

### Erweiterte Gewichtungsformel

**Non-lineare Bewertungsgewichtung:**
```python
# Exponentieller Ansatz mit Sigmoid-Smoothing
normalized_rating = (rating - 5.0) / 5.0
exponential_weight = rating^2.5
threshold_bonus = (rating - 6.0) × 1.5  # für Bewertungen ≥ 6.0
sigmoid_factor = 2 / (1 + e^(-3 × normalized_rating)) - 1

final_weight = 0.7 × exponential_weight + 0.3 × sigmoid_factor + threshold_bonus
```

**Ergebnis:** Bewertung 10.0 hat **5.17x mehr Gewicht** als linear äquivalent

**Zusätzliche Faktoren:**
- Spielhäufigkeit: `log(Spielanzahl + 1)`
- Aktualität: Zeitverfall über 12 Monate
- Konsistenz: Gleichmäßige Spielverteilung über Zeit

## 🆕 Neueste Verbesserungen

### Garantierte Empfehlungen (v2.1)
- **Mindestens 10 Empfehlungen** auch bei umfangreichen Sammlungen
- **Adaptive Suche** - automatische Erweiterung der Nachbarn-Suche
- **Zwei-Pass-Filterung** - strikt, dann entspannt bei Bedarf
- **Intelligente Fallbacks** - verhindert leere Empfehlungslisten

### Non-lineare Bewertungsgewichtung (v2.2)
- **Exponentielles Wachstum** - hohe Bewertungen haben deutlich mehr Einfluss
- **Schwellwert-Verstärkung** - Bonus für Bewertungen ≥ 6.0
- **Sigmoid-Smoothing** - sanfte Übergänge zwischen Gewichtungsstufen
- **Konfigurierbare Parameter** - vollständig anpassbar

| Rating | Linear | Non-Linear | Verstärkung |
|--------|--------|------------|-------------|
| 6.0    | 1.0    | 0.10       | 0.10x       |
| 7.0    | 2.0    | 2.56       | 1.28x       |
| 8.0    | 3.0    | 7.53       | 2.51x       |
| 9.0    | 4.0    | 15.20      | 3.80x       |
| 10.0   | 5.0    | 25.86      | **5.17x**   |

## 📝 Beispiel-Ausgabe

```
🎯 Top 20 ML-Empfehlungen aus den BGG Top 1000:
======================================================================

 1. Terraforming Mars (BGG Rang #3) - 2016
    ⭐ Rating: 8.4
    🧩 Komplexität: 3.2/5
    📂 Kategorien: Economic, Science Fiction, Territory Building
    ⚙️ Mechaniken: Card Drafting, Hand Management, Tile Placement
    ✍️ Autoren: Jacob Fryxelius
    🎨 Illustratoren: Isaac Fryxelius
    🎯 Ähnlichkeit: 87.3%

 2. Scythe (BGG Rang #9) - 2016
    ⭐ Rating: 8.3
    🧩 Komplexität: 3.4/5
    📂 Kategorien: Strategy, Economic, Fighting
    ⚙️ Mechaniken: Area Control, Variable Player Powers
    🎯 Ähnlichkeit: 84.7%
    
...mindestens 10 Empfehlungen garantiert
```

## 🔍 Problembehandlung

### "Keine Sammlung gefunden"
- Überprüfen Sie den BGG-Nutzernamen
- Stellen Sie sicher, dass die Sammlung öffentlich ist

### "Scraping-Fehler"
- System verwendet automatisch Fallback-Liste
- Funktioniert auch ohne Live-Scraping

### "Wenige Empfehlungen"
- System garantiert mindestens 10 Empfehlungen
- Automatische Erweiterung der Suchkriterien
- Bei sehr umfangreichen Sammlungen werden Filter gelockert

## 🛠️ Erweiterungsmöglichkeiten

1. **Mehr ML-Algorithmen**: Collaborative Filtering, Matrix Factorization
2. **Deep Learning**: Neural Networks für komplexere Muster
3. **Hybrid-Ansätze**: Kombination verschiedener Techniken
4. **Web-Interface**: Flask/Django Frontend
5. **Erweiterte Bewertungsmodelle**: Weitere non-lineare Ansätze
6. **Mehr Features**: Themen, Altersempfehlungen, Spielzeit-Präferenzen
7. **Adaptive Learning**: Lernende Parameter basierend auf Nutzerverhalten

## 📄 Lizenz

Dieses Projekt ist für Bildungszwecke erstellt. Respektieren Sie die BGG-Nutzungsbedingungen und verwenden Sie angemessene Delays bei API-Calls.

## 🤝 Beitragen

Verbesserungen und Erweiterungen sind willkommen! Erstellen Sie einen Pull Request oder öffnen Sie ein Issue.

---

**Viel Spaß beim Entdecken neuer Brettspiele! 🎲**