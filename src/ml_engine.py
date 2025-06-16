"""
Machine Learning Engine für das BGG Empfehlungssystem
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from config import (
    MIN_FEATURE_FREQUENCY,
    MAX_NEIGHBORS,
    SIMILARITY_METRIC,
    DEFAULT_NUM_RECOMMENDATIONS,
    DEBUG_SHOW_SIMILARITY_DETAILS,
    WEIGHTS,
    PLAY_COUNT_LOG_BASE,
    RECENCY_DECAY_MONTHS,
    MIN_PLAYS_FOR_CONSISTENCY,
    RECENT_THRESHOLD_MONTHS,
    RATING_WEIGHTING,
    EXCLUDE_BGG_RATING_FROM_FEATURES
)


class BGGMLEngine:
    def __init__(self):
        self.feature_matrix = None
        self.ml_model = None
        self.scaler = StandardScaler()
        self.feature_info = {}
        self.feature_names = []
    
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
            msg = f"⚠️  Duplikate entdeckt: {total_rows} Zeilen, "
            msg += f"aber nur {unique_ids} eindeutige IDs"
            print(msg)
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
        frequent_designers, frequent_artists, frequent_publishers = (
            self._filter_frequent_features(
                games_df, all_designers, all_artists, all_publishers
            )
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
        designers_info = f"   {len(frequent_designers)} häufige Autoren "
        designers_info += f"(von {len(all_designers)} total)"
        print(designers_info)
        artists_info = f"   {len(frequent_artists)} häufige Illustratoren "
        artists_info += f"(von {len(all_artists)} total)"
        print(artists_info)
        publishers_info = f"   {len(frequent_publishers)} häufige Verlage "
        publishers_info += f"(von {len(all_publishers)} total)"
        print(publishers_info)
        
        # Feature-Namen erstellen
        self.feature_names = []
        
        # Numerische Feature-Namen
        if not EXCLUDE_BGG_RATING_FROM_FEATURES:
            self.feature_names.append('avg_rating')
        self.feature_names.extend([
            'complexity', 'min_players', 'max_players', 
            'log_playing_time', 'game_age', 'capped_game_age'
        ])
        
        # Kategorische Feature-Namen
        for category in self.feature_info['categories']:
            self.feature_names.append(f'category_{category}')
        for mechanic in self.feature_info['mechanics']:
            self.feature_names.append(f'mechanic_{mechanic}')
        for designer in self.feature_info['designers']:
            self.feature_names.append(f'designer_{designer}')
        for artist in self.feature_info['artists']:
            self.feature_names.append(f'artist_{artist}')
        for publisher in self.feature_info['publishers']:
            self.feature_names.append(f'publisher_{publisher}')

        # Feature-Matrix erstellen
        features = []
        current_year = 2025
        
        for _, game in games_df.iterrows():
            feature_vector = []
            
            # Numerische Features
            game_age = current_year - game['year_published']
            
            # Basis-Features (ohne BGG-Rating)
            numeric_features = [
                game['complexity'],
                game['min_players'],
                game['max_players'],
                np.log(game['playing_time'] + 1),  # Log-transform für Spielzeit
                game_age,  # Alter des Spiels
                min(game_age, 25)  # Gekapptes Alter für sehr alte Spiele
            ]
            
            # BGG-Rating optional hinzufügen
            # Hinweis: BGG-Rating bleibt als Qualitätsfilter aktiv, wird aber nicht für ML-Ähnlichkeit verwendet
            if not EXCLUDE_BGG_RATING_FROM_FEATURES:
                numeric_features.insert(0, game['avg_rating'])
            
            feature_vector.extend(numeric_features)
            
            # One-Hot Encoding für kategorische Features mit Gewichtung
            categorical_features = self._encode_categorical_features_weighted(
                game, self.feature_info
            )
            feature_vector.extend(categorical_features)
            
            features.append(feature_vector)
        
        self.feature_matrix = np.array(features)
        
        # Prüfe auf identische Feature-Vektoren
        unique_vectors = np.unique(self.feature_matrix, axis=0)
        if len(unique_vectors) != len(self.feature_matrix):
            duplicate_vectors = len(self.feature_matrix) - len(unique_vectors)
            msg = f"⚠️  {duplicate_vectors} Spiele haben identische Feature-Vektoren"
            print(msg)
        
        # Normalisierung
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
        
        self._print_feature_summary()
        
        return True
    
    def _filter_frequent_features(
            self, games_df, all_designers, all_artists, all_publishers
    ):
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
        
        frequent_designers = {
            name for name, count in designer_counts.items()
            if count >= MIN_FEATURE_FREQUENCY
        }
        frequent_artists = {
            name for name, count in artist_counts.items()
            if count >= MIN_FEATURE_FREQUENCY
        }
        frequent_publishers = {
            name for name, count in publisher_counts.items()
            if count >= MIN_FEATURE_FREQUENCY
        }
        
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
    
    def _encode_categorical_features_weighted(self, game, feature_info):
        """Erstellt gewichtetes One-Hot Encoding für kategorische Features"""
        encoding = []
        
        # Kategorien (gewichtet)
        for category in feature_info['categories']:
            value = WEIGHTS['features']['category_weight'] if category in game['categories'] else 0
            encoding.append(value)
        
        # Mechaniken (gewichtet)
        for mechanic in feature_info['mechanics']:
            value = WEIGHTS['features']['mechanic_weight'] if mechanic in game['mechanics'] else 0
            encoding.append(value)
        
        # Designer (gewichtet)
        for designer in feature_info['designers']:
            value = WEIGHTS['features']['designer_weight'] if designer in game['designers'] else 0
            encoding.append(value)
        
        # Artists (gewichtet)
        for artist in feature_info['artists']:
            value = WEIGHTS['features']['artist_weight'] if artist in game['artists'] else 0
            encoding.append(value)
        
        # Publishers (gewichtet)
        for publisher in feature_info['publishers']:
            value = WEIGHTS['features']['publisher_weight'] if publisher in game['publishers'] else 0
            encoding.append(value)
        
        return encoding
    
    def _print_feature_summary(self):
        """Gibt eine Zusammenfassung der Feature-Matrix aus"""
        print(f"✓ Feature-Matrix erstellt: {self.feature_matrix.shape}")
        print(f"   {self.feature_matrix.shape[0]} Spiele × {self.feature_matrix.shape[1]} Features")
        
        # Feature-Aufschlüsselung
        numeric_features = 6 if EXCLUDE_BGG_RATING_FROM_FEATURES else 7
        category_features = len(self.feature_info['categories'])
        mechanic_features = len(self.feature_info['mechanics'])
        designer_features = len(self.feature_info['designers'])
        artist_features = len(self.feature_info['artists'])
        publisher_features = len(self.feature_info['publishers'])
        
        breakdown = f"   Aufschlüsselung: {numeric_features} numerisch + "
        breakdown += f"{category_features} Kategorien (×{WEIGHTS['features']['category_weight']}) + "
        breakdown += f"{mechanic_features} Mechaniken (×{WEIGHTS['features']['mechanic_weight']}) + "
        breakdown += f"{designer_features} Autoren (×{WEIGHTS['features']['designer_weight']}) + "
        breakdown += f"{artist_features} Illustratoren (×{WEIGHTS['features']['artist_weight']}) + "
        breakdown += f"{publisher_features} Verlage (×{WEIGHTS['features']['publisher_weight']})"
        print(breakdown)
    
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
        msg = f"✓ ML-Modell trainiert (k={n_neighbors}, metric={SIMILARITY_METRIC})"
        print(msg)
        return True
    
    def create_user_preferences_vector(
            self, collection_data, plays_data, game_details
    ):
        """Erstellt einen erweiterten Präferenz-Vektor basierend auf Nutzerdaten"""
        if collection_data is None or len(collection_data) == 0:
            return None
        
        # Spielhäufigkeiten und zeitliche Daten berechnen
        play_counts, recent_plays, play_dates = (
            self._analyze_play_patterns(plays_data)
        )
        
        # Erweiterte gewichtete Präferenzen sammeln
        weighted_games = []
        
        for game in collection_data:
            game_id = game['id']
            rating = game['rating']
            play_count = play_counts.get(game_id, 0)
            
            if game_id in game_details:
                # Erweiterte Gewichtung berechnen
                weight = self._calculate_advanced_weight(
                    game, play_count, recent_plays.get(game_id, 0), 
                    play_dates.get(game_id, []), game_details[game_id]
                )
                
                if weight > 0:
                    weighted_games.append((game_id, weight))
        
        if not weighted_games:
            return None
        
        return self._calculate_weighted_preferences(weighted_games, game_details)
    
    def _analyze_play_patterns(self, plays_data):
        """Analysiert Spiel-Muster für erweiterte Gewichtung"""
        play_counts = defaultdict(int)
        recent_plays = defaultdict(int)
        play_dates = defaultdict(list)
        
        if plays_data is not None:
            from datetime import datetime, timedelta
            
            # Schwellwert für "recent" plays
            recent_threshold = datetime.now() - timedelta(days=30 * RECENT_THRESHOLD_MONTHS)
            
            for play in plays_data:
                game_id = play['game_id']
                quantity = play['quantity']
                
                play_counts[game_id] += quantity
                
                # Datum verarbeiten
                try:
                    if play['date']:
                        play_date = datetime.strptime(play['date'], '%Y-%m-%d')
                        play_dates[game_id].append(play_date)
                        
                        # Zähle recent plays
                        if play_date >= recent_threshold:
                            recent_plays[game_id] += quantity
                except:
                    # Fehlerhafte Datumsformate ignorieren
                    pass
        
        return play_counts, recent_plays, play_dates
    
    def _calculate_nonlinear_rating_weight(self, rating):
        """Berechnet non-lineares Gewicht basierend auf Spielbewertung"""
        rating_config = RATING_WEIGHTING
        
        if not rating_config['use_nonlinear']:
            # Fallback zu linearer Gewichtung
            return max(0, rating - rating_config['min_rating'])
        
        # Normalisiere Rating (0-10 Skala auf 0-1)
        normalized_rating = (rating - rating_config['min_rating']) / (10.0 - rating_config['min_rating'])
        normalized_rating = max(0, min(1, normalized_rating))  # Begrenze auf [0,1]
        
        # Exponentieller Gewichtungsterm
        exponential_weight = np.power(normalized_rating, rating_config['exponent'])
        
        # Verstärkung für sehr hohe Bewertungen (über threshold)
        if rating >= rating_config['threshold']:
            threshold_bonus = (rating - rating_config['threshold']) * rating_config['amplification_factor']
            exponential_weight += threshold_bonus
        
        # Sigmoid-ähnliche Transformation für sanftere Übergänge
        sigmoid_factor = 2.0 / (1.0 + np.exp(-3.0 * normalized_rating)) - 1.0
        
        # Kombiniere exponentiellen und sigmoid Ansatz
        final_weight = 0.7 * exponential_weight + 0.3 * sigmoid_factor
        
        # Skaliere zurück auf sinnvolle Werte (multipliziere mit der ursprünglichen Range)
        scaled_weight = final_weight * (rating - rating_config['min_rating'])
        
        return max(0, scaled_weight)
    
    def _calculate_advanced_weight(self, game, play_count, recent_plays, play_dates, game_details):
        """Berechnet erweiterte Gewichtung für ein Spiel"""
        weights = WEIGHTS['user_preferences']
        total_weight = 0
        
        # 1. Erweiterte Bewertungsgewichtung (non-linear)
        if game['rating'] and game['rating'] > RATING_WEIGHTING['min_rating']:
            rating_weight = self._calculate_nonlinear_rating_weight(game['rating']) * weights['rating_weight']
            total_weight += rating_weight
        
        # 2. Spielhäufigkeits-Gewicht
        if play_count > 0:
            play_weight = np.log(play_count + PLAY_COUNT_LOG_BASE) * weights['play_count_weight']
            total_weight += play_weight
        
        # 3. Neuheits-Gewicht (recent plays)
        if recent_plays > 0:
            recency_weight = np.log(recent_plays + 1) * weights['recency_weight']
            total_weight += recency_weight
        
        # 4. Konsistenz-Gewicht
        if len(play_dates) >= MIN_PLAYS_FOR_CONSISTENCY:
            consistency_score = self._calculate_consistency_score(play_dates, play_count)
            consistency_weight = consistency_score * weights['consistency_weight']
            total_weight += consistency_weight
        
        # 5. Zeitverfall für alte Bewertungen
        if play_dates:
            time_decay = self._calculate_time_decay(play_dates)
            total_weight *= time_decay
        
        return max(0, total_weight)  # Negative Gewichte vermeiden
    
    def _calculate_consistency_score(self, play_dates, total_plays):
        """Berechnet Konsistenz-Score basierend auf Spielverteilung über Zeit"""
        if not play_dates or len(play_dates) < 2:
            return 0.5  # Neutral score
        
        from datetime import datetime
        
        # Sortiere Daten
        sorted_dates = sorted(play_dates)
        
        # Berechne zeitliche Verteilung
        time_span = (sorted_dates[-1] - sorted_dates[0]).days
        
        if time_span == 0:
            return 0.8  # Alle an einem Tag gespielt
        
        # Je gleichmäßiger verteilt, desto höher der Score
        expected_interval = time_span / len(sorted_dates)
        actual_intervals = [(sorted_dates[i+1] - sorted_dates[i]).days 
                           for i in range(len(sorted_dates)-1)]
        
        if actual_intervals:
            # Standardabweichung der Intervalle (normalisiert)
            std_dev = np.std(actual_intervals)
            consistency = max(0, 1 - (std_dev / expected_interval))
            return min(1.0, consistency)
        
        return 0.5
    
    def _calculate_time_decay(self, play_dates):
        """Berechnet Zeitverfall-Faktor"""
        if not play_dates:
            return 1.0
        
        from datetime import datetime
        
        # Letztes Spieldatum
        last_play = max(play_dates)
        months_since_last_play = (datetime.now() - last_play).days / 30
        
        # Exponentieller Verfall
        decay_factor = np.exp(-months_since_last_play / RECENCY_DECAY_MONTHS)
        
        # Mindestens 10% der ursprünglichen Gewichtung
        return max(0.1, decay_factor)
    
    def _calculate_weighted_preferences(self, weighted_games, game_details):
        """Berechnet gewichtete Durchschnittspräferenzen mit erweiterten Metriken"""
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
            'publishers': Counter(),
            # Erweiterte Präferenzen (NEU)
            'complexity_variance': 0,
            'time_variance': 0,
            'preferred_eras': Counter(),
            'designer_loyalty': Counter(),
            'weight_distribution': []
        }
        
        complexity_values = []
        time_values = []
        era_values = []
        
        for game_id, weight in weighted_games:
            details = game_details[game_id]
            norm_weight = weight / total_weight
            
            # Standard-Präferenzen
            preferences['avg_rating'] += details['avg_rating'] * norm_weight
            preferences['complexity'] += details['complexity'] * norm_weight
            preferences['min_players'] += details['min_players'] * norm_weight
            preferences['max_players'] += details['max_players'] * norm_weight
            preferences['playing_time'] += details['playing_time'] * norm_weight
            preferences['year_published'] += details.get('year_published', 2000) * norm_weight
            
            # Sammle Werte für Varianz-Berechnung
            complexity_values.append(details['complexity'])
            time_values.append(details['playing_time'])
            era_values.append(details.get('year_published', 2000))
            preferences['weight_distribution'].append(weight)
            
            # Kategorische Features
            for category in details['categories']:
                preferences['categories'][category] += norm_weight
            for mechanic in details['mechanics']:
                preferences['mechanics'][mechanic] += norm_weight
            for designer in details.get('designers', []):
                preferences['designers'][designer] += norm_weight
                preferences['designer_loyalty'][designer] += norm_weight
            for artist in details.get('artists', []):
                preferences['artists'][artist] += norm_weight
            for publisher in details.get('publishers', []):
                preferences['publishers'][publisher] += norm_weight
            
            # Ära-Präferenzen
            year = details.get('year_published', 2000)
            era = self._categorize_era(year)
            preferences['preferred_eras'][era] += norm_weight
        
        # Berechne Varianzen für Diversitäts-Präferenzen
        if complexity_values:
            preferences['complexity_variance'] = np.var(complexity_values)
        if time_values:
            preferences['time_variance'] = np.var(time_values)
        
        # Berechne Präferenz-Stärke (wie stark ausgeprägt sind die Vorlieben?)
        preferences['preference_strength'] = self._calculate_preference_strength(preferences)
        
        return preferences
    
    def _categorize_era(self, year):
        """Kategorisiert Spiele nach Ära"""
        if year < 1995:
            return 'classic'
        elif year < 2005:
            return 'golden_age'
        elif year < 2015:
            return 'modern'
        else:
            return 'contemporary'
    
    def _calculate_preference_strength(self, preferences):
        """Berechnet wie stark ausgeprägt die Präferenzen sind"""
        # Berechne Konzentration der Top-Kategorien
        top_categories = preferences['categories'].most_common(3)
        total_category_weight = sum(preferences['categories'].values())
        
        if total_category_weight > 0:
            top_concentration = sum(weight for _, weight in top_categories) / total_category_weight
        else:
            top_concentration = 0
        
        # Berechne Komplexitäts-Konsistenz
        complexity_consistency = 1 / (1 + preferences['complexity_variance'])
        
        # Kombiniere zu Gesamtstärke
        strength = (top_concentration + complexity_consistency) / 2
        return min(1.0, strength)
    
    def generate_recommendations(self, user_preferences, games_df, owned_game_ids, num_recommendations=DEFAULT_NUM_RECOMMENDATIONS):
        """Generiert ML-basierte Empfehlungen mit erweiterten Präferenz-Matching"""
        if self.ml_model is None or games_df is None or user_preferences is None:
            return []
        
        # Erstelle Nutzer-Feature-Vektor
        user_vector = self._create_user_feature_vector(user_preferences)
        
        if user_vector is None:
            return []
        
        # Mindestens 10 Empfehlungen anstreben
        min_recommendations = 10
        target_recommendations = max(num_recommendations, min_recommendations)
        
        # Finde ähnliche Spiele mit erweiterten Parametern
        # Mehr Nachbarn bei starken Präferenzen, weniger bei schwachen
        preference_strength = user_preferences.get('preference_strength', 0.5)
        neighbor_multiplier = 2 + int(preference_strength * 2)  # 2-4x
        max_neighbors = min(target_recommendations * neighbor_multiplier, len(games_df))
        
        # Iterativ mehr Nachbarn suchen, bis genügend Empfehlungen gefunden
        attempt = 1
        max_attempts = 3
        
        while attempt <= max_attempts:
            current_neighbors = min(max_neighbors * attempt, len(games_df))
            distances, indices = self.ml_model.kneighbors(user_vector, n_neighbors=current_neighbors)
            
            # Erweiterte Filterung und Ranking
            recommendations = self._advanced_filter_and_rank_recommendations(
                games_df, indices[0], distances[0], owned_game_ids, 
                user_preferences, target_recommendations
            )
            
            # Prüfe ob genügend Empfehlungen gefunden wurden
            if len(recommendations) >= min_recommendations or attempt == max_attempts:
                if len(recommendations) < min_recommendations and attempt == max_attempts:
                    print(f"⚠️  Nur {len(recommendations)} Empfehlungen gefunden (weniger als {min_recommendations})")
                return recommendations[:num_recommendations]
            
            print(f"🔍 Versuch {attempt}: {len(recommendations)} Empfehlungen gefunden, erweitere Suche...")
            attempt += 1
        
        return []
    
    def _create_user_feature_vector(self, user_prefs):
        """Erstellt Feature-Vektor für Nutzerpräferenzen"""
        current_year = 2025
        user_game_age = current_year - user_prefs['year_published']
        
        user_vector = []
        
        # Basis-Features (ohne BGG-Rating)
        numeric_features = [
            user_prefs['complexity'],
            user_prefs['min_players'],
            user_prefs['max_players'],
            np.log(user_prefs['playing_time'] + 1),
            user_game_age,
            min(user_game_age, 25)
        ]
        
        # BGG-Rating optional hinzufügen
        if not EXCLUDE_BGG_RATING_FROM_FEATURES:
            numeric_features.insert(0, user_prefs['avg_rating'])
        
        user_vector.extend(numeric_features)
        
        # Kategorische Features (gewichtet)
        for category in self.feature_info['categories']:
            value = user_prefs['categories'].get(category, 0) * WEIGHTS['features']['category_weight']
            user_vector.append(value)
        
        for mechanic in self.feature_info['mechanics']:
            value = user_prefs['mechanics'].get(mechanic, 0) * WEIGHTS['features']['mechanic_weight']
            user_vector.append(value)
        
        for designer in self.feature_info['designers']:
            value = user_prefs['designers'].get(designer, 0) * WEIGHTS['features']['designer_weight']
            user_vector.append(value)
        
        for artist in self.feature_info['artists']:
            value = user_prefs['artists'].get(artist, 0) * WEIGHTS['features']['artist_weight']
            user_vector.append(value)
        
        for publisher in self.feature_info['publishers']:
            value = user_prefs['publishers'].get(publisher, 0) * WEIGHTS['features']['publisher_weight']
            user_vector.append(value)
        
        user_vector = np.array(user_vector).reshape(1, -1)
        return self.scaler.transform(user_vector)
    
    def _advanced_filter_and_rank_recommendations(self, games_df, indices, distances, 
                                                 owned_game_ids, user_preferences, num_recommendations):
        """Erweiterte Filterung und Ranking mit Präferenz-Matching"""
        seen_game_ids = set()
        recommendations = []
        filtered_out_count = 0
        
        if DEBUG_SHOW_SIMILARITY_DETAILS:
            print(f"🔍 Analysiere {len(indices)} ähnliche Spiele mit erweiterten Kriterien...")
        
        # Extrahiere Nutzer-Präferenzen für erweiterte Filterung
        complexity_range = self._get_complexity_range(user_preferences)
        preferred_time_range = self._get_time_range(user_preferences)
        preferred_eras = user_preferences.get('preferred_eras', Counter())
        
        # Erweitere Filter-Bereiche wenn zu wenige Empfehlungen gefunden werden
        min_recommendations = 10
        initial_filter_strict = True
        
        for pass_num in range(2):  # Zwei Durchgänge: strikt, dann entspannt
            if pass_num == 1 and len(recommendations) >= min_recommendations:
                break  # Genügend Empfehlungen im ersten Durchgang
            
            if pass_num == 1:
                # Entspannte Filter für zweiten Durchgang
                complexity_range = self._get_relaxed_complexity_range(user_preferences)
                preferred_time_range = self._get_relaxed_time_range(user_preferences)
                if DEBUG_SHOW_SIMILARITY_DETAILS:
                    print(f"🔄 Erweitere Filter-Kriterien für mehr Empfehlungen...")
            
            for i, idx in enumerate(indices):
                game = games_df.iloc[idx]
                game_id = game['id']
                
                # Basis-Filter: Bereits besessen oder duplikat
                if game_id in owned_game_ids or game_id in seen_game_ids:
                    continue
                
                # Erweiterte Filter (entspannt im zweiten Durchgang)
                if pass_num == 0 and not self._passes_advanced_filters(game, user_preferences, complexity_range, preferred_time_range):
                    filtered_out_count += 1
                    continue
                elif pass_num == 1 and not self._passes_relaxed_filters(game, user_preferences, complexity_range, preferred_time_range):
                    continue
                
                seen_game_ids.add(game_id)
                
                # Berechne erweiterten Ähnlichkeits-Score
                base_similarity = 1 - distances[i]
                enhanced_score = self._calculate_enhanced_similarity_score(
                    game, user_preferences, base_similarity
                )
                
                # Debug-Info für erste paar Spiele
                if DEBUG_SHOW_SIMILARITY_DETAILS and len(recommendations) < 5:
                    filter_info = "(entspannt)" if pass_num == 1 else ""
                    print(f"   {len(recommendations)+1}. {game['name']} (ID: {game_id}) {filter_info}")
                    print(f"      Basis-Ähnlichkeit: {base_similarity:.3f}")
                    print(f"      Erweiterte Ähnlichkeit: {enhanced_score:.3f}")
                
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
                    'similarity_score': enhanced_score,
                    'base_similarity': base_similarity,
                    'match_reasons': self._get_match_reasons(game, user_preferences)
                })
                
                if len(recommendations) >= num_recommendations:
                    break
            
            if len(recommendations) >= num_recommendations:
                break
        
        # Sortiere nach erweitertem Ähnlichkeits-Score
        recommendations.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        if filtered_out_count > 0 and DEBUG_SHOW_SIMILARITY_DETAILS:
            print(f"📊 {filtered_out_count} Spiele durch erweiterte Filter ausgeschlossen")
        
        print(f"✓ {len(recommendations)} erweiterte Empfehlungen gefunden")
        return recommendations
    
    def _get_complexity_range(self, user_preferences):
        """Bestimmt akzeptable Komplexitäts-Range basierend auf Nutzer-Präferenzen"""
        avg_complexity = user_preferences.get('complexity', 2.5)
        complexity_variance = user_preferences.get('complexity_variance', 0.5)
        
        # Bei niedriger Varianz: engerer Bereich, bei hoher: weiter
        range_width = 0.5 + complexity_variance
        
        return (max(1.0, avg_complexity - range_width), 
                min(5.0, avg_complexity + range_width))
    
    def _get_time_range(self, user_preferences):
        """Bestimmt akzeptable Spielzeit-Range"""
        avg_time = user_preferences.get('playing_time', 60)
        time_variance = user_preferences.get('time_variance', 900)  # 30min variance default
        
        # Logarithmische Skalierung für Spielzeit
        range_factor = np.sqrt(time_variance) / 10
        range_width = max(15, avg_time * 0.3 + range_factor)
        
        return (max(5, avg_time - range_width), avg_time + range_width)
    
    def _get_relaxed_complexity_range(self, user_preferences):
        """Bestimmt erweiterte akzeptable Komplexitäts-Range für entspannte Filter"""
        avg_complexity = user_preferences.get('complexity', 2.5)
        complexity_variance = user_preferences.get('complexity_variance', 0.5)
        
        # Erweitere Bereich um 50% für entspannte Filter
        range_width = (0.5 + complexity_variance) * 1.5
        
        return (max(1.0, avg_complexity - range_width), 
                min(5.0, avg_complexity + range_width))
    
    def _get_relaxed_time_range(self, user_preferences):
        """Bestimmt erweiterte akzeptable Spielzeit-Range für entspannte Filter"""
        avg_time = user_preferences.get('playing_time', 60)
        time_variance = user_preferences.get('time_variance', 900)
        
        # Erweitere Bereich um 100% für entspannte Filter
        range_factor = np.sqrt(time_variance) / 10
        range_width = max(15, avg_time * 0.3 + range_factor) * 2
        
        return (max(5, avg_time - range_width), avg_time + range_width)
    
    def _passes_advanced_filters(self, game, user_preferences, complexity_range, time_range):
        """Prüft ob Spiel erweiterte Filter passiert"""
        # Komplexitäts-Filter
        if not (complexity_range[0] <= game['complexity'] <= complexity_range[1]):
            return False
        
        # Spielzeit-Filter  
        if not (time_range[0] <= game['playing_time'] <= time_range[1]):
            return False
        
        # Weitere Filter können hier hinzugefügt werden
        return True
    
    def _passes_relaxed_filters(self, game, user_preferences, complexity_range, time_range):
        """Prüft ob Spiel entspannte Filter passiert (weniger strikt)"""
        # Nur grundlegende Filter für entspannte Suche
        # Komplexitäts-Filter (erweitert)
        if not (complexity_range[0] <= game['complexity'] <= complexity_range[1]):
            return False
        
        # Spielzeit-Filter (erweitert)
        if not (time_range[0] <= game['playing_time'] <= time_range[1]):
            return False
        
        # Sehr niedrige Bewertungen ausschließen
        if game['avg_rating'] < 5.0:
            return False
        
        return True
    
    def _calculate_enhanced_similarity_score(self, game, user_preferences, base_similarity):
        """Berechnet erweiterten Ähnlichkeits-Score"""
        enhanced_score = base_similarity
        
        # Designer-Loyalty Bonus
        for designer in game['designers']:
            if designer in user_preferences['designer_loyalty']:
                loyalty_strength = user_preferences['designer_loyalty'][designer]
                enhanced_score += loyalty_strength * 0.1  # Max 10% Bonus
        
        # Ära-Präferenz Bonus
        game_era = self._categorize_era(game['year_published'])
        if game_era in user_preferences.get('preferred_eras', {}):
            era_strength = user_preferences['preferred_eras'][game_era]
            enhanced_score += era_strength * 0.05  # Max 5% Bonus
        
        # Komplexitäts-Match Bonus
        complexity_diff = abs(game['complexity'] - user_preferences['complexity'])
        complexity_bonus = max(0, (1 - complexity_diff) * 0.1)  # Bis 10% Bonus
        enhanced_score += complexity_bonus
        
        return min(1.0, enhanced_score)  # Cap bei 1.0
    
    def _get_match_reasons(self, game, user_preferences):
        """Erstellt Liste von Gründen warum das Spiel passt"""
        reasons = []
        
        # Top Kategorien
        top_categories = [cat for cat, _ in user_preferences['categories'].most_common(3)]
        matching_categories = set(game['categories']) & set(top_categories)
        if matching_categories:
            reasons.append(f"Kategorie-Match: {', '.join(list(matching_categories)[:2])}")
        
        # Designer Match
        for designer in game['designers'][:2]:
            if designer in user_preferences['designer_loyalty']:
                reasons.append(f"Lieblings-Autor: {designer}")
                break
        
        # Komplexität Match
        complexity_diff = abs(game['complexity'] - user_preferences['complexity'])
        if complexity_diff <= 0.5:
            reasons.append(f"Passende Komplexität ({game['complexity']:.1f})")
        
        return reasons[:3]  # Max 3 Gründe