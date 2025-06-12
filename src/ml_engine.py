"""
Machine Learning Engine für das BGG Empfehlungssystem
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from config import *


class BGGMLEngine:
    def __init__(self):
        self.feature_matrix = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_info = {}
    
    def create_feature_matrix(self, games_df):
        """Erstellt Feature-Matrix für Machine Learning"""
        if games_df is None or len(games_df) == 0:
            return False
        
        print("🔧 Erstelle Feature-Matrix...")
        
        # Prüfe auf Duplikate
        print(f"📊 Eingangsdaten: {len(games_df)} Spiele")
        unique_ids = games_df['id'].nunique()
        total_rows = len(games_df)
        
        if unique_ids != total_rows:
            print(f"⚠️  Duplikate entdeckt: {total_rows} Zeilen, aber nur {unique_ids} eindeutige IDs")
            games_df = games_df.drop_duplicates(subset=['id'], keep='first')
            print(f"✓ Nach Bereinigung: {len(games_df)} Spiele")
        
        # Alle Features sammeln
        all_categories = set()
        all_mechanics = set()
        all_designers = set()
        all_artists = set()
        all_publishers = set()
        
        for _, game in games_df.iterrows():
            all_categories.update(game['categories'])
            all_mechanics.update(game['mechanics'])
            all_designers.update(game['designers'])
            all_artists.update(game['artists'])
            all_publishers.update(game['publishers'])
        
        # Filter: Nur häufige Features (mindestens MIN_FEATURE_FREQUENCY Spiele)
        frequent_designers, frequent_artists, frequent_publishers = self._filter_frequent_features(
            games_df, all_designers, all_artists, all_publishers
        )
        
        # Feature-Info speichern
        self.feature_info = {
            'categories': sorted(all_categories),
            'mechanics': sorted(all_mechanics),
            'designers': sorted(frequent_designers),
            'artists': sorted(frequent_artists),
            'publishers': sorted(frequent_publishers)
        }
        
        print(f"📂 Features:")
        print(f"   {len(all_categories)} Kategorien")
        print(f"   {len(all_mechanics)} Mechaniken") 
        print(f"   {len(frequent_designers)} häufige Autoren (von {len(all_designers)} total)")
        print(f"   {len(frequent_artists)} häufige Illustratoren (von {len(all_artists)} total)")
        print(f"   {len(frequent_publishers)} häufige Verlage (von {len(all_publishers)} total)")
        
        # Feature-Matrix erstellen
        features = []
        current_year = 2025
        
        for _, game in games_df.iterrows():
            feature_vector = []
            
            # Numerische Features
            game_age = current_year - game['year_published']
            
            feature_vector.extend([
                game['avg_rating'],
                game['complexity'],
                game['min_players'],
                game['max_players'],
                np.log(game['playing_time'] + 1),  # Log-transform für Spielzeit
                game_age,  # Alter des Spiels
                min(game_age, 25)  # Gekapptes Alter für sehr alte Spiele
            ])
            
            # One-Hot Encoding für kategorische Features
            feature_vector.extend(self._encode_categorical_features(game, self.feature_info))
            
            features.append(feature_vector)
        
        self.feature_matrix = np.array(features)
        
        # Prüfe auf identische Feature-Vektoren
        unique_vectors = np.unique(self.feature_matrix, axis=0)
        if len(unique_vectors) != len(self.feature_matrix):
            duplicate_vectors = len(self.feature_matrix) - len(unique_vectors)
            print(f"⚠️  {duplicate_vectors} Spiele haben identische Feature-Vektoren")
        
        # Normalisierung
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
        
        self._print_feature_summary()
        
        return True
    
    def _filter_frequent_features(self, games_df, all_designers, all_artists, all_publishers):
        """Filtert Features nach Häufigkeit"""
        designer_counts = Counter()
        artist_counts = Counter()
        publisher_counts = Counter()
        
        for _, game in games_df.iterrows():
            for designer in game['designers']:
                designer_counts[designer] += 1
            for artist in game['artists']:
                artist_counts[artist] += 1
            for publisher in game['publishers']:
                publisher_counts[publisher] += 1
        
        frequent_designers = {name for name, count in designer_counts.items() 
                            if count >= MIN_FEATURE_FREQUENCY}
        frequent_artists = {name for name, count in artist_counts.items() 
                          if count >= MIN_FEATURE_FREQUENCY}
        frequent_publishers = {name for name, count in publisher_counts.items() 
                             if count >= MIN_FEATURE_FREQUENCY}
        
        return frequent_designers, frequent_artists, frequent_publishers
    
    def _encode_categorical_features(self, game, feature_info):
        """Erstellt One-Hot Encoding für kategorische Features"""
        encoding = []
        
        # Kategorien
        for category in feature_info['categories']:
            encoding.append(1 if category in game['categories'] else 0)
        
        # Mechaniken
        for mechanic in feature_info['mechanics']:
            encoding.append(1 if mechanic in game['mechanics'] else 0)
        
        # Designer
        for designer in feature_info['designers']:
            encoding.append(1 if designer in game['designers'] else 0)
        
        # Artists
        for artist in feature_info['artists']:
            encoding.append(1 if artist in game['artists'] else 0)
        
        # Publishers
        for publisher in feature_info['publishers']:
            encoding.append(1 if publisher in game['publishers'] else 0)
        
        return encoding
    
    def _print_feature_summary(self):
        """Gibt eine Zusammenfassung der Feature-Matrix aus"""
        print(f"✓ Feature-Matrix erstellt: {self.feature_matrix.shape}")
        print(f"   {self.feature_matrix.shape[0]} Spiele × {self.feature_matrix.shape[1]} Features")
        
        # Feature-Aufschlüsselung
        numeric_features = 7
        category_features = len(self.feature_info['categories'])
        mechanic_features = len(self.feature_info['mechanics'])
        designer_features = len(self.feature_info['designers'])
        artist_features = len(self.feature_info['artists'])
        publisher_features = len(self.feature_info['publishers'])
        
        print(f"   Aufschlüsselung: {numeric_features} numerisch + {category_features} Kategorien + "
              f"{mechanic_features} Mechaniken + {designer_features} Autoren + "
              f"{artist_features} Illustratoren + {publisher_features} Verlage")
    
    def train_model(self):
        """Trainiert das Machine Learning Modell"""
        if self.feature_matrix is None:
            return False
        
        print("🤖 Trainiere ML-Modell...")
        
        # k-NN für Ähnlichkeitssuche
        n_neighbors = min(MAX_NEIGHBORS, len(self.feature_matrix))
        
        self.ml_model = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=SIMILARITY_METRIC,
            algorithm='brute'
        )
        
        self.ml_model.fit(self.feature_matrix)
        print(f"✓ ML-Modell trainiert (k={n_neighbors}, metric={SIMILARITY_METRIC})")
        return True
    
    def create_user_preferences_vector(self, collection_data, plays_data, game_details):
        """Erstellt einen Präferenz-Vektor basierend auf Nutzerdaten"""
        if collection_data is None or len(collection_data) == 0:
            return None
        
        # Spielhäufigkeiten berechnen
        play_counts = defaultdict(int)
        if plays_data is not None:
            for play in plays_data:
                play_counts[play['game_id']] += play['quantity']
        
        # Gewichtete Präferenzen sammeln
        weighted_games = []
        
        for game in collection_data:
            game_id = game['id']
            rating = game['rating']
            play_count = play_counts.get(game_id, 0)
            
            # Gewichtung berechnen
            weight = 0
            if rating and rating > 7:
                weight += (rating - 5) * RATING_WEIGHT_MULTIPLIER
            if play_count > 0:
                weight += np.log(play_count + PLAY_COUNT_LOG_BASE)
            
            if weight > 0 and game_id in game_details:
                weighted_games.append((game_id, weight))
        
        if not weighted_games:
            return None
        
        return self._calculate_weighted_preferences(weighted_games, game_details)
    
    def _calculate_weighted_preferences(self, weighted_games, game_details):
        """Berechnet gewichtete Durchschnittspräferenzen"""
        total_weight = sum(weight for _, weight in weighted_games)
        
        # Sammle gewichtete Features
        current_year = 2025
        preferences = {
            'avg_rating': 0,
            'complexity': 0,
            'min_players': 0,
            'max_players': 0,
            'playing_time': 0,
            'year_published': 0,
            'categories': Counter(),
            'mechanics': Counter(),
            'designers': Counter(),
            'artists': Counter(),
            'publishers': Counter()
        }
        
        for game_id, weight in weighted_games:
            details = game_details[game_id]
            norm_weight = weight / total_weight
            
            preferences['avg_rating'] += details['avg_rating'] * norm_weight
            preferences['complexity'] += details['complexity'] * norm_weight
            preferences['min_players'] += details['min_players'] * norm_weight
            preferences['max_players'] += details['max_players'] * norm_weight
            preferences['playing_time'] += details['playing_time'] * norm_weight
            preferences['year_published'] += details.get('year_published', 2000) * norm_weight
            
            for category in details['categories']:
                preferences['categories'][category] += norm_weight
            for mechanic in details['mechanics']:
                preferences['mechanics'][mechanic] += norm_weight
            for designer in details.get('designers', []):
                preferences['designers'][designer] += norm_weight
            for artist in details.get('artists', []):
                preferences['artists'][artist] += norm_weight
            for publisher in details.get('publishers', []):
                preferences['publishers'][publisher] += norm_weight
        
        return preferences
    
    def generate_recommendations(self, user_preferences, games_df, owned_game_ids, num_recommendations=DEFAULT_NUM_RECOMMENDATIONS):
        """Generiert ML-basierte Empfehlungen"""
        if self.ml_model is None or games_df is None or user_preferences is None:
            return []
        
        # Erstelle Nutzer-Feature-Vektor
        user_vector = self._create_user_feature_vector(user_preferences)
        
        if user_vector is None:
            return []
        
        # Finde ähnliche Spiele
        max_neighbors = min(num_recommendations * 3, len(games_df))
        distances, indices = self.ml_model.kneighbors(user_vector, n_neighbors=max_neighbors)
        
        return self._filter_and_rank_recommendations(
            games_df, indices[0], distances[0], owned_game_ids, num_recommendations
        )
    
    def _create_user_feature_vector(self, user_prefs):
        """Erstellt Feature-Vektor für Nutzerpräferenzen"""
        current_year = 2025
        user_game_age = current_year - user_prefs['year_published']
        
        user_vector = []
        user_vector.extend([
            user_prefs['avg_rating'],
            user_prefs['complexity'],
            user_prefs['min_players'],
            user_prefs['max_players'],
            np.log(user_prefs['playing_time'] + 1),
            user_game_age,
            min(user_game_age, 25)
        ])
        
        # Kategorische Features
        for category in self.feature_info['categories']:
            user_vector.append(user_prefs['categories'].get(category, 0))
        
        for mechanic in self.feature_info['mechanics']:
            user_vector.append(user_prefs['mechanics'].get(mechanic, 0))
        
        for designer in self.feature_info['designers']:
            user_vector.append(user_prefs['designers'].get(designer, 0))
        
        for artist in self.feature_info['artists']:
            user_vector.append(user_prefs['artists'].get(artist, 0))
        
        for publisher in self.feature_info['publishers']:
            user_vector.append(user_prefs['publishers'].get(publisher, 0))
        
        user_vector = np.array(user_vector).reshape(1, -1)
        return self.scaler.transform(user_vector)
    
    def _filter_and_rank_recommendations(self, games_df, indices, distances, owned_game_ids, num_recommendations):
        """Filtert und rankt Empfehlungen"""
        seen_game_ids = set()
        recommendations = []
        
        if DEBUG_SHOW_SIMILARITY_DETAILS:
            print(f"🔍 Analysiere {len(indices)} ähnliche Spiele...")
        
        for i, idx in enumerate(indices):
            game = games_df.iloc[idx]
            game_id = game['id']
            
            # Debug-Info für erste paar Spiele
            if DEBUG_SHOW_SIMILARITY_DETAILS and i < 5:
                print(f"   {i+1}. {game['name']} (ID: {game_id}) - Ähnlichkeit: {1 - distances[i]:.3f}")
            
            # Überspringe bereits besessene und bereits empfohlene Spiele
            if game_id not in owned_game_ids and game_id not in seen_game_ids:
                seen_game_ids.add(game_id)
                
                recommendations.append({
                    'rank': game['rank'],
                    'id': game_id,
                    'name': game['name'],
                    'avg_rating': game['avg_rating'],
                    'complexity': game['complexity'],
                    'categories': game['categories'][:3],
                    'mechanics': game['mechanics'][:3],
                    'designers': game['designers'][:2],
                    'artists': game['artists'][:2],
                    'year_published': game['year_published'],
                    'similarity_score': 1 - distances[i]
                })
                
                if len(recommendations) >= num_recommendations:
                    break
        
        print(f"✓ {len(recommendations)} eindeutige Empfehlungen gefunden")
        return recommendations