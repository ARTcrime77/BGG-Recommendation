"""
Konfigurationsdatei für das BGG ML-Empfehlungssystem
"""

import os

# Cache-Einstellungen
CACHE_DIR = "bgg_cache"
TOP_GAMES_FILE = os.path.join(CACHE_DIR, "top_games.json")
GAME_DETAILS_FILE = os.path.join(CACHE_DIR, "game_details.json")
CACHE_MAX_AGE_DAYS = 7

# BGG API-Einstellungen
BGG_API_BASE_URL = "https://boardgamegeek.com/xmlapi2"
BGG_BROWSE_URL = "https://boardgamegeek.com/browse/boardgame/page/"
API_DELAY = 1.5  # Sekunden zwischen API-Calls
BATCH_SIZE = 20  # Anzahl Spiele pro API-Request

# Web-Scraping Einstellungen
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
SCRAPING_DELAY = 2  # Sekunden zwischen Web-Requests
TARGET_TOP_GAMES = 1000  # Ziel: gewünschte Anzahl eindeutiger Spiele
MAX_SCRAPING_PAGES = 10  # Maximal 10 Seiten (1000 Spiele) scrapen

# ML-Einstellungen
MIN_FEATURE_FREQUENCY = 2  # Mindestanzahl Spiele für Features (Autoren, etc.)
MAX_NEIGHBORS = 20  # Maximale Nachbarn für k-NN
SIMILARITY_METRIC = 'cosine'

# Gewichtungsparameter
RATING_WEIGHT_MULTIPLIER = 2  # Faktor für Bewertungen über 5
PLAY_COUNT_LOG_BASE = 1  # Basis für Logarithmus der Spielanzahl

# Erweiterte Gewichtungsparameter (NEU)
USER_PREFERENCE_WEIGHTS = {
    'rating_weight': 5.0,           # Bewertungen stärker gewichten
    'play_count_weight': 0.5,       # Spielhäufigkeit
    'recency_weight': 1.5,          # Neuere Bewertungen bevorzugen
    'consistency_weight': 0.5,      # Konsistenz der Bewertungen
    'complexity_match_weight': 1.2, # Komplexitäts-Präferenz
    'time_preference_weight': 0.7   # Spielzeit-Präferenz
}

# Zeitbasierte Gewichtung
RECENCY_DECAY_MONTHS = 12    # Nach wie vielen Monaten Gewichtung um 50% reduziert
MIN_PLAYS_FOR_CONSISTENCY = 3  # Mindest-Spiele für Konsistenz-Berechnung
RECENT_THRESHOLD_MONTHS = 6   # Spiele der letzten 6 Monate als "recent" betrachten

# Ausgabe-Einstellungen
DEFAULT_NUM_RECOMMENDATIONS = 10
DEBUG_SHOW_SIMILARITY_DETAILS = True
SHOW_PROGRESS_EVERY = 40  # Progress-Update alle X Spiele