# -*- coding: utf-8 -*-
"""
Offline BGG Recommender

Erweitert das BGG-Empfehlungssystem um vollständige Offline-Funktionalität
ohne Abhängigkeit von BGG APIs oder Internetverbindung.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from ml_engine import BGGMLEngine
from offline_data import offline_manager
from config import DEFAULT_NUM_RECOMMENDATIONS


class OfflineBGGRecommender:
    """
    Offline-Version des BGG-Empfehlungssystems
    
    Verwendet statische Daten und Beispiel-Nutzerprofile für Empfehlungen
    ohne Internetverbindung.
    """
    
    def __init__(self, username: str = None):
        self.username = username or "OfflineUser"
        self.ml_engine = BGGMLEngine()
        self.offline_mode = True
        
        # Offline-Daten laden
        self.games_database = offline_manager.get_offline_games_database()
        self.sample_profiles = offline_manager.get_sample_user_profiles()
        
        print(f"🔌 Offline-Modus aktiviert")
        print(f"   📚 {len(self.games_database)} Spiele in Offline-Datenbank")
        print(f"   👥 {len(self.sample_profiles)} Beispiel-Nutzerprofile verfügbar")
    
    def list_sample_users(self) -> List[str]:
        """
        Listet verfügbare Beispiel-Nutzer auf
        
        Returns:
            Liste der verfügbaren Nutzernamen
        """
        return [profile['username'] for profile in self.sample_profiles]
    
    def select_sample_user(self, username: str) -> bool:
        """
        Wählt einen Beispiel-Nutzer aus
        
        Args:
            username: Name des Beispiel-Nutzers
            
        Returns:
            True wenn Nutzer gefunden und ausgewählt
        """
        for profile in self.sample_profiles:
            if profile['username'] == username:
                self.username = username
                self.current_profile = profile
                print(f"👤 Beispiel-Nutzer ausgewählt: {username}")
                print(f"   📖 {profile['description']}")
                print(f"   🎮 {len(profile['collection'])} Spiele in Sammlung")
                return True
        
        print(f"❌ Beispiel-Nutzer '{username}' nicht gefunden")
        return False
    
    def create_games_dataframe(self) -> pd.DataFrame:
        """
        Erstellt DataFrame aus Offline-Spieldatenbank
        
        Returns:
            DataFrame mit Spielinformationen
        """
        print("📊 Erstelle DataFrame aus Offline-Datenbank...")
        
        # Konvertiere zu DataFrame
        games_df = pd.DataFrame(self.games_database)
        
        # Stelle sicher, dass alle nötigen Spalten vorhanden sind
        required_columns = [
            'id', 'name', 'rank', 'avg_rating', 'complexity',
            'categories', 'mechanics', 'designers', 'year',
            'min_players', 'max_players', 'playing_time'
        ]
        
        for col in required_columns:
            if col not in games_df.columns:
                # Standardwerte für fehlende Spalten
                if col == 'year':
                    games_df[col] = games_df.get('year_published', 2020)
                elif col in ['categories', 'mechanics', 'designers']:
                    games_df[col] = games_df[col].fillna([])
                elif col in ['min_players', 'max_players']:
                    games_df[col] = games_df[col].fillna(2)
                elif col == 'playing_time':
                    games_df[col] = games_df[col].fillna(60)
                else:
                    games_df[col] = games_df[col].fillna(0)
        
        # Umbenenne year zu year_published falls nötig (ML-Engine erwartet year_published)
        if 'year' in games_df.columns and 'year_published' not in games_df.columns:
            games_df['year_published'] = games_df['year']
        
        print(f"✓ DataFrame mit {len(games_df)} Spielen erstellt")
        return games_df
    
    def get_user_collection_data(self, username: str) -> List[Dict]:
        """
        Lädt Sammlung des Beispiel-Nutzers
        
        Args:
            username: Name des Beispiel-Nutzers
            
        Returns:
            Liste der Spiele in der Sammlung
        """
        for profile in self.sample_profiles:
            if profile['username'] == username:
                return profile['collection']
        
        return []
    
    def create_user_preferences_from_profile(self, username: str) -> Optional[Dict]:
        """
        Erstellt Nutzerpräferenzen aus Beispiel-Profil
        
        Args:
            username: Name des Beispiel-Nutzers
            
        Returns:
            Dictionary mit Nutzerpräferenzen oder None
        """
        profile = None
        for p in self.sample_profiles:
            if p['username'] == username:
                profile = p
                break
        
        if not profile:
            return None
        
        collection = profile['collection']
        preferences = profile.get('preferences', {})
        
        from collections import Counter
        
        # Erstelle Präferenz-Vektor basierend auf Profil
        user_preferences = {
            'avg_rating': np.mean([game['rating'] for game in collection]),
            'complexity': preferences.get('avg_complexity', 2.5),
            'min_players': 2,
            'max_players': 4,
            'playing_time': 90,
            'year_published': 2010,
            'categories': Counter(),
            'mechanics': Counter(),
            'designers': Counter(),
            'artists': Counter(),
            'publishers': Counter()
        }
        
        # Fülle kategorische Präferenzen
        favorite_categories = preferences.get('favorite_categories', [])
        for category in favorite_categories:
            user_preferences['categories'][category] = 0.8
        
        favorite_mechanics = preferences.get('favorite_mechanics', [])
        for mechanic in favorite_mechanics:
            user_preferences['mechanics'][mechanic] = 0.7
        
        # Zusätzliche Präferenz-Metriken
        user_preferences.update({
            'complexity_variance': 0.5,
            'time_variance': 900,
            'preferred_eras': Counter({'modern': 0.6, 'contemporary': 0.4}),
            'designer_loyalty': Counter(),
            'preference_strength': 0.7
        })
        
        return user_preferences
    
    def run_offline_analysis(self, username: str = None) -> List[Dict]:
        """
        Führt vollständige Offline-Analyse durch
        
        Args:
            username: Name des Beispiel-Nutzers (optional)
            
        Returns:
            Liste der Empfehlungen
        """
        if username:
            if not self.select_sample_user(username):
                return []
        else:
            # Wähle ersten verfügbaren Nutzer
            if self.sample_profiles:
                self.select_sample_user(self.sample_profiles[0]['username'])
            else:
                print("❌ Keine Beispiel-Nutzerprofile verfügbar")
                return []
        
        print(f"\n🤖 Offline BGG-Empfehlungsanalyse für {self.username}")
        print("=" * 60)
        
        # 1. Spiele-DataFrame erstellen
        games_df = self.create_games_dataframe()
        
        # 2. Feature-Matrix erstellen
        print("\n🔧 Erstelle Feature-Matrix...")
        if not self.ml_engine.create_feature_matrix(games_df):
            print("❌ Fehler beim Erstellen der Feature-Matrix")
            return []
        
        # 3. ML-Modell trainieren
        print("\n🤖 Trainiere ML-Modell...")
        if not self.ml_engine.train_model():
            print("❌ Fehler beim Trainieren des ML-Modells")
            return []
        
        # 4. Nutzerpräferenzen erstellen
        print(f"\n👤 Erstelle Präferenzen für {self.username}...")
        user_preferences = self.create_user_preferences_from_profile(self.username)
        
        if not user_preferences:
            print("❌ Konnte Nutzerpräferenzen nicht erstellen")
            return []
        
        # 5. Empfehlungen generieren
        print(f"\n🎯 Generiere Empfehlungen...")
        collection = self.get_user_collection_data(self.username)
        owned_game_ids = {game['id'] for game in collection}
        
        recommendations = self.ml_engine.generate_recommendations(
            user_preferences, games_df, owned_game_ids, DEFAULT_NUM_RECOMMENDATIONS
        )
        
        return recommendations
    
    def display_offline_recommendations(self, recommendations: List[Dict]):
        """
        Zeigt Offline-Empfehlungen formatiert an
        """
        if recommendations:
            print(f"\n🎯 Top {len(recommendations)} Offline-Empfehlungen:")
            print("=" * 50)
            
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i:2d}. {rec['name']} (Rang #{rec['rank']})")
                print(f"    ⭐ Rating: {rec['avg_rating']:.1f}")
                print(f"    🧩 Komplexität: {rec['complexity']:.1f}/5")
                print(f"    📂 Kategorien: {', '.join(rec['categories'][:3])}")
                print(f"    ⚙️  Mechaniken: {', '.join(rec['mechanics'][:3])}")
                print(f"    🎯 Ähnlichkeit: {rec['similarity_score']:.1%}")
        else:
            print("❌ Keine Offline-Empfehlungen gefunden.")
    
    def show_available_profiles(self):
        """
        Zeigt verfügbare Beispiel-Profile an
        """
        print("\n👥 Verfügbare Beispiel-Nutzerprofile:")
        print("=" * 40)
        
        for i, profile in enumerate(self.sample_profiles, 1):
            print(f"\n{i}. {profile['username']}")
            print(f"   📖 {profile['description']}")
            print(f"   🎮 {len(profile['collection'])} Spiele")
            
            prefs = profile.get('preferences', {})
            if 'favorite_categories' in prefs:
                cats = ', '.join(prefs['favorite_categories'][:3])
                print(f"   📂 Kategorien: {cats}")


def demo_offline_mode():
    """
    Demo-Funktion für Offline-Modus
    """
    print("🔌 BGG Offline-Empfehlungssystem Demo")
    print("=" * 50)
    
    # Prüfe Internetverbindung
    online = offline_manager.check_internet_connectivity()
    print(f"🌐 Internet-Verbindung: {'✅ Verfügbar' if online else '❌ Nicht verfügbar'}")
    
    if online:
        print("💡 Offline-Modus kann trotzdem für Demo verwendet werden")
    
    # Erstelle Offline-Recommender
    recommender = OfflineBGGRecommender()
    
    # Zeige verfügbare Profile
    recommender.show_available_profiles()
    
    # Führe Analyse für alle Profile durch
    for profile in recommender.sample_profiles:
        username = profile['username']
        print(f"\n" + "="*60)
        print(f"🎯 Analysiere Profil: {username}")
        print("="*60)
        
        recommendations = recommender.run_offline_analysis(username)
        recommender.display_offline_recommendations(recommendations[:5])  # Top 5


if __name__ == "__main__":
    demo_offline_mode()