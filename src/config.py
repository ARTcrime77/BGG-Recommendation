"""
Konfigurationsdatei für das BGG ML-Empfehlungssystem
"""

import os

# Cache-Einstellungen
CACHE_DIR = "bgg_cache"
TOP500_FILE = os.path.join(CACHE_DIR, "top500_games.json")
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
TARGET_TOP_GAMES = 1000  # Ziel: 500 eindeutige Spiele
MAX_SCRAPING_PAGES = 15  # Maximal 15 Seiten (1500 Spiele) scrapen

# ML-Einstellungen
MIN_FEATURE_FREQUENCY = 2  # Mindestanzahl Spiele für Features (Autoren, etc.)
MAX_NEIGHBORS = 10  # Maximale Nachbarn für k-NN
SIMILARITY_METRIC = 'cosine'

# Gewichtungsparameter
RATING_WEIGHT_MULTIPLIER = 2  # Faktor für Bewertungen über 7
PLAY_COUNT_LOG_BASE = 1  # Basis für Logarithmus der Spielanzahl

# Ausgabe-Einstellungen
DEFAULT_NUM_RECOMMENDATIONS = 10
DEBUG_SHOW_SIMILARITY_DETAILS = True
SHOW_PROGRESS_EVERY = 40  # Progress-Update alle X Spiele