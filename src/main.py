# -*- coding: utf-8 -*-
"""
Hauptklasse für das BGG ML-Empfehlungssystem
"""

import pandas as pd
import os
import json
import sys
from datetime import datetime
import matplotlib
# Verwende Agg Backend für maximale Kompatibilität und Dateierzeugung
matplotlib.use('Agg')

from config import (
    DEBUG_SHOW_SIMILARITY_DETAILS,
    DEFAULT_NUM_RECOMMENDATIONS,
    TOP_GAMES_FILE,
    GAME_DETAILS_FILE,
    TARGET_TOP_GAMES,
    ENABLE_VISUALIZATIONS,
    SAVE_PLOTS_AS_FILES,
    SHOW_PLOTS_GUI
)
from data_loader import BGGDataLoader
from ml_engine import BGGMLEngine
from visualizer import BGGVisualizer
from report_generator import ReportGenerator
from cache_manager import cache_manager


class BGGRecommender:
    def __init__(self, username):
        self.username = username
        self.data_loader = BGGDataLoader()
        self.ml_engine = BGGMLEngine()
        self.visualizer = BGGVisualizer()
        
        # Datenstrukturen
        self.collection_data = None
        self.plays_data = None
        self.game_details = {}
        self.top_games_data = None
        self.user_preferences = None
    
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
        
        # Lade bereits gecachte Details
        cached_details = self.data_loader.load_game_details_cache()
        self.game_details.update(cached_details)
        
        # Finde fehlende User-Spiel-Details
        missing_user_ids = [gid for gid in user_game_ids if gid not in self.game_details]
        
        if missing_user_ids:
            print(f"🔍 Lade Details für {len(missing_user_ids)} neue Spiele aus Ihrer Sammlung...")
            user_game_details = self.data_loader.fetch_game_details(missing_user_ids)
            self.game_details.update(user_game_details)
            # Speichere aktualisierte Details im Cache
            self.data_loader.save_game_details_cache(self.game_details)
        else:
            print("✓ Alle Spiele aus Ihrer Sammlung bereits im Cache")
        
        return True
    
    def load_top_games_data(self):
        """Lädt Top-Spiele-Daten und stellt sicher, dass TARGET_TOP_GAMES eindeutige verfügbar sind"""
        print(f"\n🎯 Lade Top {TARGET_TOP_GAMES}+ Spiele...")
        
        # Lade Top-Spiele Liste (automatisch erweitert auf TARGET_TOP_GAMES eindeutige)
        top_games_list = self.data_loader.load_top_games()
        
        if not top_games_list:
            print("❌ Keine Top-Spiele gefunden")
            return False
        
        print(f"✓ {len(top_games_list)} eindeutige Top-Spiele für ML-Training verfügbar")
        
        # Cache bereits in load_user_data() geladen
        print(f"📊 {len(self.game_details)} Spieldetails bereits im Speicher")
        
        # Finde Spiele, für die wir noch keine Details haben
        top_game_ids = [game['id'] for game in top_games_list]
        missing_ids = [gid for gid in top_game_ids if gid not in self.game_details]
        
        if missing_ids:
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
    
    def generate_recommendations(
            self, num_recommendations=DEFAULT_NUM_RECOMMENDATIONS
    ):
        """Generiert personalisierte Empfehlungen"""
        print(f"\n🎯 Generiere {num_recommendations} personalisierte Empfehlungen...")
        
        # Erstelle Nutzerpräferenzen
        self.user_preferences = self.ml_engine.create_user_preferences_vector(
            self.collection_data, self.plays_data, self.game_details
        )
        
        if self.user_preferences is None:
            print("❌ Nicht genügend Nutzerdaten für Empfehlungen")
            return []
        
        # Besessene Spiele-IDs
        owned_game_ids = {game['id'] for game in self.collection_data}
        
        # Generiere Empfehlungen
        recommendations = self.ml_engine.generate_recommendations(
            self.user_preferences, self.top_games_data, owned_game_ids,
            num_recommendations
        )
        
        return recommendations
    
    def display_recommendations(self, recommendations):
        """Zeigt Empfehlungen formatiert an"""
        if recommendations:
            print(f"\n🎯 Top {len(recommendations)} ML-Empfehlungen aus den BGG Top {TARGET_TOP_GAMES}:")
            print("=" * 70)
            
            for i, rec in enumerate(recommendations, 1):
                rank_info = f"BGG Rang #{rec['rank']}"
                year_info = rec['year_published']
                name_line = f"\n{i:2d}. {rec['name']} ({rank_info}) - {year_info}"
                print(name_line)
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
        if os.path.exists(TOP_GAMES_FILE):
            cache_time = datetime.fromtimestamp(os.path.getmtime(TOP_GAMES_FILE))
            print(f"   Top {TARGET_TOP_GAMES}: {cache_time.strftime('%d.%m.%Y %H:%M')}")
        if os.path.exists(GAME_DETAILS_FILE):
            cache_time = datetime.fromtimestamp(os.path.getmtime(GAME_DETAILS_FILE))
            print(f"   Spieldetails: {cache_time.strftime('%d.%m.%Y %H:%M')}")

    def create_visualizations(self, recommendations):
        """Erstellt alle Visualisierungen der Empfehlungsergebnisse"""
        if not recommendations:
            print("⚠️ Keine Empfehlungen für Visualisierung vorhanden.")
            return {}
        
        print("\n📊 Erstelle Visualisierungen...")
        save_paths = {}
        
        try:
            # Konvertiere Empfehlungen zu DataFrame
            recommendations_df = pd.DataFrame(recommendations)
            
            # Konvertiere Top-Games-Daten für Visualisierung
            games_data = self.top_games_data.copy()
            games_data['rating'] = games_data['avg_rating']  # Rename für Visualizer
            
            # Hole Feature-Matrix und Feature-Namen vom ML-Engine
            feature_matrix = self.ml_engine.feature_matrix
            feature_names = self.ml_engine.feature_names
            
            # Konfiguriere Speicherpfade basierend auf Einstellungen
            if SAVE_PLOTS_AS_FILES:
                plots_dir = "bgg_cache/plots"
                os.makedirs(plots_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_paths = {
                    'similarity': os.path.join(plots_dir, f"similarity_{timestamp}.png"),
                    'ratings': os.path.join(plots_dir, f"ratings_{timestamp}.png"),
                    'features': os.path.join(plots_dir, f"features_{timestamp}.png"),
                    'user_prefs': os.path.join(plots_dir, f"user_prefs_{timestamp}.png"),
                    'era_complexity': os.path.join(plots_dir, f"era_complexity_{timestamp}.png"),
                    'creators': os.path.join(plots_dir, f"creators_{timestamp}.png"),
                    'collection_stats': os.path.join(plots_dir, f"collection_stats_{timestamp}.png")
                }
            
            # Konfiguriere GUI-Anzeige
            original_show_gui = SHOW_PLOTS_GUI
            
            # Einzelne Plots erstellen
            print("📊 Erstelle Sammlungs-Statistiken...")
            self.visualizer.plot_user_collection_stats(
                self.collection_data,
                self.plays_data,
                save_paths.get('collection_stats'),
                show_gui=original_show_gui
            )

            print("📊 Erstelle Nutzerprofil-Analyse...")
            self.visualizer.plot_user_preferences(
                self.user_preferences,
                save_paths.get('user_prefs'),
                show_gui=original_show_gui
            )

            print("📊 Erstelle Ära/Komplexitäts-Heatmap...")
            self.visualizer.plot_era_complexity_heatmap(
                recommendations_df,
                save_paths.get('era_complexity'),
                show_gui=original_show_gui
            )

            print("📊 Erstelle Creator-Treemap...")
            self.visualizer.plot_creator_treemap(
                recommendations_df,
                save_paths.get('creators'),
                show_gui=original_show_gui
            )

            print("📊 Erstelle Ähnlichkeitsdiagramm...")
            self.visualizer.plot_recommendation_similarity(
                recommendations_df, 
                self.collection_data, 
                save_paths.get('similarity'), 
                show_gui=original_show_gui
            )
            
            print("📊 Erstelle Bewertungsverteilungsdiagramm...")
            self.visualizer.plot_rating_distribution(
                games_data, 
                self.collection_data, 
                save_paths.get('ratings'),
                show_gui=original_show_gui
            )
            
            print("📊 Erstelle Feature-Analyse...")
            self.visualizer.plot_feature_analysis(
                feature_matrix, 
                feature_names, 
                games_data, 
                save_paths.get('features'),
                show_gui=original_show_gui
            )
            
            if SAVE_PLOTS_AS_FILES:
                print(f"💾 Plots gespeichert in: {plots_dir}")
            if not original_show_gui:
                print("📊 GUI-Anzeige deaktiviert (SHOW_PLOTS_GUI = False)")
            
            print("✅ Visualisierungen erfolgreich erstellt!")
            
        except Exception as e:
            print(f"⚠️ Fehler beim Erstellen der Visualisierungen: {e}")
            print("💡 Stellen Sie sicher, dass matplotlib und seaborn installiert sind:")
            print("   pip install matplotlib seaborn")
            
        return save_paths
    
    def run_analysis(self):
        """Führt die komplette ML-Analyse durch"""
        print(f"🤖 BGG ML-Empfehlungssystem mit {TARGET_TOP_GAMES} eindeutigen Top-Spielen")
        print("=" * 65)
        
        # 1. Nutzerdaten laden
        if not self.load_user_data():
            return
        
        # 2. Top-Spiele laden (automatisch auf TARGET_TOP_GAMES eindeutige erweitert)
        if not self.load_top_games_data():
            return
        
        # 3. ML-Modell trainieren
        if not self.train_ml_model():
            return
        
        # 4. Empfehlungen generieren
        recommendations = self.generate_recommendations()
        
        # 5. Ergebnisse anzeigen
        self.display_recommendations(recommendations)
        
        # 6. Empfehlungen speichern
        self.save_recommendations(recommendations)
        
        # 7. Visualisierungen erstellen (falls aktiviert)
        plot_paths = {}
        if ENABLE_VISUALIZATIONS:
            plot_paths = self.create_visualizations(recommendations)
        else:
            print("📊 Visualisierungen deaktiviert (ENABLE_VISUALIZATIONS = False)")
        
        # 8. HTML-Report erstellen
        if plot_paths:
            print("\n📝 Erstelle HTML-Report...")
            ReportGenerator.create_html_report(
                recommendations,
                self.user_preferences,
                plot_paths,
                save_dir=os.getcwd() 
            )

        # 9. Cache-Info
        self.show_cache_info()
    
    def save_recommendations(self, recommendations):
        """Speichert Empfehlungen mit CacheManager für spätere Visualisierung"""
        if not recommendations:
            return
        
        # Zusätzliche Metadaten sammeln
        metadata = {
            "username": self.username,
            "num_recommendations": len(recommendations),
            "games_data_summary": {
                "total_games": len(self.top_games_data) if self.top_games_data is not None else 0,
                "feature_count": len(self.ml_engine.feature_names) if hasattr(self.ml_engine, 'feature_names') else 0
            }
        }
        
        try:
            cache_manager.save_json_cache('recommendations', recommendations, metadata)
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern der Empfehlungen: {e}")
    
    def load_recommendations(self):
        """Lädt gespeicherte Empfehlungen mit CacheManager"""
        try:
            cache_result = cache_manager.load_json_cache('recommendations')
            
            if cache_result:
                recommendations, metadata = cache_result
                
                # Zeige Info über geladene Daten
                timestamp = metadata.get('timestamp')
                if timestamp:
                    cache_time = datetime.fromisoformat(timestamp)
                    print(f"📁 Lade gespeicherte Empfehlungen vom {cache_time.strftime('%d.%m.%Y %H:%M')}")
                
                username = metadata.get('metadata', {}).get('username', 'Unbekannt')
                num_recs = metadata.get('metadata', {}).get('num_recommendations', len(recommendations))
                print(f"   Nutzer: {username}")
                print(f"   Anzahl Empfehlungen: {num_recs}")
                
                return recommendations
            else:
                print("❌ Keine gespeicherten Empfehlungen gefunden")
                return None
                
        except Exception as e:
            print(f"⚠️ Fehler beim Laden der Empfehlungen: {e}")
            return None
    
    def visualize_saved_recommendations(self):
        """Erstellt Visualisierungen für gespeicherte Empfehlungen"""
        print("📊 Lade gespeicherte Empfehlungen für Visualisierung...")
        
        recommendations = self.load_recommendations()
        if not recommendations:
            return False
        
        # Versuche Nutzerdaten zu laden (für Sammlungs-Stats)
        print("🔍 Lade Nutzerdaten für Visualisierung...")
        self.load_user_data()

        # Prüfe ob wir die nötigen Daten haben
        if not hasattr(self, 'top_games_data') or self.top_games_data is None:
            print("⚠️ Top-Games-Daten nicht verfügbar. Lade Basisdaten...")
            if not self.load_top_games_data():
                print("❌ Konnte Top-Games-Daten nicht laden")
                return False
        
        if not hasattr(self.ml_engine, 'feature_names') or not self.ml_engine.feature_names:
            print("⚠️ Feature-Daten nicht verfügbar. Trainiere ML-Modell...")
            if not self.train_ml_model():
                print("❌ Konnte ML-Modell nicht trainieren")
                return False
        
        # Berechne Nutzerpräferenzen wenn nötig
        if self.user_preferences is None and self.collection_data:
            print("👤 Berechne Nutzerpräferenzen für Visualisierung...")
            self.user_preferences = self.ml_engine.create_user_preferences_vector(
                self.collection_data, self.plays_data, self.game_details
            )

        # Erstelle Visualisierungen
        self.create_visualizations(recommendations)
        return True


def main():
    """Hauptfunktion"""
    print("🎲 BGG ML-Empfehlungssystem")
    print("Benötigt: pip install scikit-learn beautifulsoup4 pandas numpy matplotlib seaborn")
    print("=" * 50)
    
    # Kommandozeilen-Argumente prüfen
    batch_mode = "--batch" in sys.argv
    if batch_mode:
        cache_manager.set_batch_mode(True)

    if "--visualize-only" in sys.argv:
        if not ENABLE_VISUALIZATIONS:
            print("❌ Visualisierungen sind deaktiviert (ENABLE_VISUALIZATIONS = False)")
            print("💡 Aktivieren Sie Visualisierungen in config.py")
            return
            
        print("📊 Nur-Visualisierung-Modus")
        username = "Artcrime77"  # Hier können Sie den Nutzernamen ändern
        recommender = BGGRecommender(username)
        
        if recommender.visualize_saved_recommendations():
            print("✅ Visualisierungen erfolgreich erstellt!")
        else:
            print("❌ Konnte Visualisierungen nicht erstellen.")
            print("💡 Führen Sie zuerst eine vollständige Analyse durch:")
            print("   python src/main.py")
        return
    
    # Standard-Modus: Vollständige Analyse
    username = "Artcrime77"  # Hier können Sie den Nutzernamen ändern
    
    recommender = BGGRecommender(username)
    recommender.run_analysis()
    
    print("\n💡 Tipp: Verwenden Sie 'python src/main.py --visualize-only' um nur die Visualisierungen anzuzeigen.")


if __name__ == "__main__":
    main()