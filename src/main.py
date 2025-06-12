"""
Hauptklasse für das BGG ML-Empfehlungssystem
"""

import pandas as pd
import os
from datetime import datetime

from config import *
from data_loader import BGGDataLoader
from ml_engine import BGGMLEngine


class BGGRecommender:
    def __init__(self, username):
        self.username = username
        self.data_loader = BGGDataLoader()
        self.ml_engine = BGGMLEngine()
        
        # Datenstrukturen
        self.collection_data = None
        self.plays_data = None
        self.game_details = {}
        self.top_games_data = None
    
    def load_user_data(self):
        """Lädt alle Nutzerdaten (Sammlung und Spielstatistiken)"""
        print(f"\n👤 Lade Nutzerdaten für {self.username}...")
        
        # Sammlung laden
        collection_games = self.data_loader.fetch_user_collection(self.username)
        if not collection_games:
            print("❌ Keine Sammlung gefunden.")
            return False
        
        self.collection_data = collection_games
        
        # Spielstatistiken laden
        plays_list = self.data_loader.fetch_user_plays(self.username)
        if plays_list:
            self.plays_data = plays_list
        
        # Details für Nutzer-Spiele laden
        user_game_ids = [game['id'] for game in self.collection_data]
        print(f"\n🔍 Lade Details für {len(user_game_ids)} Spiele aus Ihrer Sammlung...")
        
        user_game_details = self.data_loader.fetch_game_details(user_game_ids)
        self.game_details.update(user_game_details)
        
        return True
    
    def load_top_games_data(self):
        """Lädt Top-Spiele-Daten und stellt sicher, dass 500 eindeutige verfügbar sind"""
        print("\n🎯 Lade Top 500+ Spiele...")
        
        # Lade Top 500+ Liste (automatisch erweitert auf 500 eindeutige)
        top_games_list = self.data_loader.load_top500_games()
        
        if not top_games_list:
            print("❌ Keine Top-Spiele gefunden")
            return False
        
        print(f"✓ {len(top_games_list)} eindeutige Top-Spiele für ML-Training verfügbar")
        
        # Lade Cache für Spieldetails
        cached_details = self.data_loader.load_game_details_cache()
        self.game_details.update(cached_details)
        
        # Finde Spiele, für die wir noch keine Details haben
        top_game_ids = [game['id'] for game in top_games_list]
        missing_ids = [gid for gid in top_game_ids if gid not in self.game_details]
        
        if missing_ids:
            should_fetch = True
            if cached_details:
                print(f"📊 {len(missing_ids)} Spiele brauchen noch Details")
                should_fetch = self.data_loader.ask_user_update_choice("Fehlende Spieldetails")
            
            if should_fetch:
                print(f"🔍 Lade Details für {len(missing_ids)} Spiele...")
                new_details = self.data_loader.fetch_game_details(missing_ids)
                self.game_details.update(new_details)
                self.data_loader.save_game_details_cache(self.game_details)
        
        # Erstelle DataFrame mit vollständigen Details
        self.top_games_data = self._create_games_dataframe(top_games_list)
        
        return True
    
    def _create_games_dataframe(self, games_list):
        """Erstellt DataFrame für Spiele mit vollständigen Details"""
        final_games = []
        seen_ids = set()
        
        for game in games_list:
            game_id = game['id']
            if game_id in self.game_details and game_id not in seen_ids:
                seen_ids.add(game_id)
                details = self.game_details[game_id]
                final_games.append({
                    'rank': game['rank'],
                    'id': game_id,
                    'name': details['name'],
                    'categories': details['categories'],
                    'mechanics': details['mechanics'],
                    'designers': details.get('designers', []),
                    'artists': details.get('artists', []),
                    'publishers': details.get('publishers', []),
                    'year_published': details.get('year_published', 2000),
                    'avg_rating': details['avg_rating'],
                    'complexity': details['complexity'],
                    'min_players': details.get('min_players', 2),
                    'max_players': details.get('max_players', 4),
                    'playing_time': details.get('playing_time', 60)
                })
        
        games_df = pd.DataFrame(final_games)
        
        # Prüfe auf Duplikate im DataFrame
        duplicates = games_df.duplicated(subset=['id'], keep='first')
        if duplicates.any():
            duplicate_count = duplicates.sum()
            print(f"⚠️  {duplicate_count} Duplikate im DataFrame gefunden - werden entfernt...")
            games_df = games_df.drop_duplicates(subset=['id'], keep='first')
        
        print(f"✓ {len(games_df)} eindeutige Top-Spiele mit vollständigen Details verfügbar")
        
        # Debug: Zeige erste paar Spiele
        if DEBUG_SHOW_SIMILARITY_DETAILS:
            print("📋 Erste 5 Spiele in der Liste:")
            for i, (_, game) in enumerate(games_df.head().iterrows()):
                print(f"   {i+1}. {game['name']} (ID: {game['id']}, Rang: {game['rank']})")
        
        return games_df
    
    def train_ml_model(self):
        """Trainiert das Machine Learning Modell"""
        print("\n🤖 Trainiere Machine Learning Modell...")
        
        if not self.ml_engine.create_feature_matrix(self.top_games_data):
            print("❌ Fehler beim Erstellen der Feature-Matrix")
            return False
        
        if not self.ml_engine.train_model():
            print("❌ Fehler beim Trainieren des ML-Modells")
            return False
        
        return True
    
    def generate_recommendations(self, num_recommendations=DEFAULT_NUM_RECOMMENDATIONS):
        """Generiert personalisierte Empfehlungen"""
        print(f"\n🎯 Generiere {num_recommendations} personalisierte Empfehlungen...")
        
        # Erstelle Nutzerpräferenzen
        user_preferences = self.ml_engine.create_user_preferences_vector(
            self.collection_data, self.plays_data, self.game_details
        )
        
        if user_preferences is None:
            print("❌ Nicht genügend Nutzerdaten für Empfehlungen")
            return []
        
        # Besessene Spiele-IDs
        owned_game_ids = {game['id'] for game in self.collection_data}
        
        # Generiere Empfehlungen
        recommendations = self.ml_engine.generate_recommendations(
            user_preferences, self.top_games_data, owned_game_ids, num_recommendations
        )
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """Zeigt Empfehlungen formatiert an"""
        if recommendations:
            print(f"\n🎯 Top {len(recommendations)} ML-Empfehlungen aus den BGG Top 500:")
            print("=" * 70)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i:2d}. {rec['name']} (BGG Rang #{rec['rank']}) - {rec['year_published']}")
                print(f"    ⭐ Rating: {rec['avg_rating']:.1f}")
                print(f"    🧩 Komplexität: {rec['complexity']:.1f}/5")
                print(f"    📂 Kategorien: {', '.join(rec['categories'])}")
                print(f"    ⚙️  Mechaniken: {', '.join(rec['mechanics'])}")
                if rec['designers']:
                    print(f"    ✍️  Autoren: {', '.join(rec['designers'])}")
                if rec['artists']:
                    print(f"    🎨 Illustratoren: {', '.join(rec['artists'])}")
                print(f"    🎯 Ähnlichkeit: {rec['similarity_score']:.1%}")
        else:
            print("❌ Keine Empfehlungen gefunden.")
    
    def show_cache_info(self):
        """Zeigt Cache-Informationen an"""
        print(f"\n📁 Cache-Info:")
        if os.path.exists(TOP500_FILE):
            cache_time = datetime.fromtimestamp(os.path.getmtime(TOP500_FILE))
            print(f"   Top 500: {cache_time.strftime('%d.%m.%Y %H:%M')}")
        if os.path.exists(GAME_DETAILS_FILE):
            cache_time = datetime.fromtimestamp(os.path.getmtime(GAME_DETAILS_FILE))
            print(f"   Spieldetails: {cache_time.strftime('%d.%m.%Y %H:%M')}")
    
    def run_analysis(self):
        """Führt die komplette ML-Analyse durch"""
        print("🤖 BGG ML-Empfehlungssystem mit 500 eindeutigen Top-Spielen")
        print("=" * 65)
        
        # 1. Nutzerdaten laden
        if not self.load_user_data():
            return
        
        # 2. Top-Spiele laden (automatisch auf 500 eindeutige erweitert)
        if not self.load_top_games_data():
            return
        
        # 3. ML-Modell trainieren
        if not self.train_ml_model():
            return
        
        # 4. Empfehlungen generieren
        recommendations = self.generate_recommendations()
        
        # 5. Ergebnisse anzeigen
        self.display_recommendations(recommendations)
        
        # 6. Cache-Info
        self.show_cache_info()


def main():
    """Hauptfunktion"""
    print("🎲 BGG ML-Empfehlungssystem")
    print("Benötigt: pip install scikit-learn beautifulsoup4 pandas numpy")
    print("=" * 50)
    
    username = "Artcrime77"  # Hier können Sie den Nutzernamen ändern
    
    recommender = BGGRecommender(username)
    recommender.run_analysis()


if __name__ == "__main__":
    main()