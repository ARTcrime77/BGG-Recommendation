"""
Unit tests for BGGMLEngine class
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd
import os
import sys
from collections import Counter

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ml_engine import BGGMLEngine
from config import MIN_FEATURE_FREQUENCY, MAX_NEIGHBORS, DEFAULT_NUM_RECOMMENDATIONS
try:
    from .test_fixtures import MOCK_GAME_DETAILS, MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA
except ImportError:
    from test_fixtures import MOCK_GAME_DETAILS, MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA


class TestBGGMLEngine(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment"""
        self.engine = BGGMLEngine()
        
        # Create test DataFrame
        self.test_games_df = pd.DataFrame([
            {
                'id': 174430,
                'name': 'Gloomhaven',
                'categories': ['Adventure', 'Exploration', 'Fantasy'],
                'mechanics': ['Action Points', 'Card Drafting', 'Cooperative'],
                'designers': ['Isaac Childres'],
                'artists': ['Artist 1'],
                'publishers': ['Publisher 1'],
                'year_published': 2017,
                'avg_rating': 8.8,
                'complexity': 3.9,
                'min_players': 1,
                'max_players': 4,
                'playing_time': 120,
                'rank': 1
            },
            {
                'id': 169786,
                'name': 'Scythe',
                'categories': ['Economic', 'Fighting', 'Science Fiction'],
                'mechanics': ['Area Control', 'Variable Powers', 'Worker Placement'],
                'designers': ['Jamey Stegmaier'],
                'artists': ['Artist 1'],  # Same artist for frequency test
                'publishers': ['Publisher 2'],
                'year_published': 2016,
                'avg_rating': 8.3,
                'complexity': 3.4,
                'min_players': 1,
                'max_players': 5,
                'playing_time': 115,
                'rank': 2
            },
            {
                'id': 167791,
                'name': 'Terraforming Mars',
                'categories': ['Economic', 'Environmental', 'Science Fiction'],
                'mechanics': ['Card Drafting', 'Hand Management', 'Tile Placement'],
                'designers': ['Jacob Fryxelius'],
                'artists': ['Artist 1'],  # Same artist for frequency test
                'publishers': ['Publisher 1'],  # Same publisher for frequency test
                'year_published': 2016,
                'avg_rating': 8.4,
                'complexity': 3.2,
                'min_players': 1,
                'max_players': 5,
                'playing_time': 120,
                'rank': 3
            }
        ])
    
    def test_init(self):
        """Test BGGMLEngine initialization"""
        self.assertIsNone(self.engine.feature_matrix)
        self.assertIsNone(self.engine.ml_model)
        self.assertEqual(self.engine.feature_info, {})
        self.assertIsNotNone(self.engine.scaler)
    
    def test_create_feature_matrix_success(self):
        """Test successful feature matrix creation"""
        result = self.engine.create_feature_matrix(self.test_games_df)
        
        self.assertTrue(result)
        self.assertIsNotNone(self.engine.feature_matrix)
        self.assertEqual(self.engine.feature_matrix.shape[0], 3)  # 3 games
        self.assertGreater(self.engine.feature_matrix.shape[1], 7)  # More than 7 numeric features
    
    def test_create_feature_matrix_empty_dataframe(self):
        """Test feature matrix creation with empty DataFrame"""
        empty_df = pd.DataFrame()
        result = self.engine.create_feature_matrix(empty_df)
        
        self.assertFalse(result)
    
    def test_create_feature_matrix_none_input(self):
        """Test feature matrix creation with None input"""
        result = self.engine.create_feature_matrix(None)
        
        self.assertFalse(result)
    
    def test_create_feature_matrix_removes_duplicates(self):
        """Test that feature matrix creation handles duplicate IDs"""
        # Add duplicate game with same ID
        df_with_duplicates = self.test_games_df.copy()
        duplicate_row = self.test_games_df.iloc[0].copy()
        duplicate_row['name'] = 'Gloomhaven Duplicate'
        df_with_duplicates = pd.concat([df_with_duplicates, duplicate_row.to_frame().T], ignore_index=True)
        
        result = self.engine.create_feature_matrix(df_with_duplicates)
        
        self.assertTrue(result)
        self.assertEqual(self.engine.feature_matrix.shape[0], 3)  # Should still be 3 after dedup
    
    def test_filter_frequent_features(self):
        """Test frequent feature filtering"""
        with patch('ml_engine.MIN_FEATURE_FREQUENCY', 2):
            frequent_designers, frequent_artists, frequent_publishers = self.engine._filter_frequent_features(
                self.test_games_df, 
                set(['Isaac Childres', 'Jamey Stegmaier', 'Jacob Fryxelius']),
                set(['Artist 1', 'Artist 2']),
                set(['Publisher 1', 'Publisher 2'])
            )
        
        # Artist 1 appears 3 times, so should be frequent
        self.assertIn('Artist 1', frequent_artists)
        # Artist 2 appears 0 times, so should not be frequent
        self.assertNotIn('Artist 2', frequent_artists)
        
        # Publisher 1 appears 2 times, so should be frequent with MIN_FREQUENCY = 2
        self.assertIn('Publisher 1', frequent_publishers)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        feature_info = {
            'categories': ['Adventure', 'Economic', 'Fantasy'],
            'mechanics': ['Action Points', 'Card Drafting'],
            'designers': ['Isaac Childres'],
            'artists': ['Artist 1'],
            'publishers': ['Publisher 1']
        }
        
        game = {
            'categories': ['Adventure', 'Fantasy'],
            'mechanics': ['Card Drafting'],
            'designers': ['Isaac Childres'],
            'artists': ['Artist 1'],
            'publishers': []
        }
        
        encoding = self.engine._encode_categorical_features(game, feature_info)
        
        # Should have 8 features total (3 cats + 2 mechs + 1 designer + 1 artist + 1 publisher)
        self.assertEqual(len(encoding), 8)
        # Adventure should be 1
        self.assertEqual(encoding[0], 1)
        # Economic should be 0
        self.assertEqual(encoding[1], 0)
        # Fantasy should be 1
        self.assertEqual(encoding[2], 1)
        # Card Drafting should be 1
        self.assertEqual(encoding[4], 1)
        # Publisher should be 0 (not in game publishers)
        self.assertEqual(encoding[7], 0)
    
    def test_print_feature_summary(self):
        """Test feature summary printing"""
        # Setup feature matrix and info
        self.engine.feature_matrix = np.random.rand(3, 15)
        self.engine.feature_info = {
            'categories': ['Cat1', 'Cat2'],
            'mechanics': ['Mech1', 'Mech2', 'Mech3'],
            'designers': ['Designer1'],
            'artists': ['Artist1'],
            'publishers': ['Pub1']
        }
        
        # Should not raise an exception
        with patch('builtins.print'):
            self.engine._print_feature_summary()
    
    def test_train_model_success(self):
        """Test successful ML model training"""
        # Setup feature matrix first
        self.engine.create_feature_matrix(self.test_games_df)
        
        result = self.engine.train_model()
        
        self.assertTrue(result)
        self.assertIsNotNone(self.engine.ml_model)
    
    def test_train_model_no_feature_matrix(self):
        """Test ML model training without feature matrix"""
        result = self.engine.train_model()
        
        self.assertFalse(result)
    
    def test_create_user_preferences_vector_success(self):
        """Test successful user preferences vector creation"""
        result = self.engine.create_user_preferences_vector(
            MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS
        )
        
        self.assertIsNotNone(result)
        self.assertIn('avg_rating', result)
        self.assertIn('categories', result)
        self.assertIn('mechanics', result)
    
    def test_create_user_preferences_vector_no_collection(self):
        """Test user preferences vector creation with no collection data"""
        result = self.engine.create_user_preferences_vector(
            None, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS
        )
        
        self.assertIsNone(result)
    
    def test_create_user_preferences_vector_empty_collection(self):
        """Test user preferences vector creation with empty collection"""
        result = self.engine.create_user_preferences_vector(
            [], MOCK_PLAYS_DATA, MOCK_GAME_DETAILS
        )
        
        self.assertIsNone(result)
    
    def test_calculate_weighted_preferences(self):
        """Test weighted preferences calculation"""
        weighted_games = [(174430, 0.6), (169786, 0.4)]
        
        result = self.engine._calculate_weighted_preferences(weighted_games, MOCK_GAME_DETAILS)
        
        self.assertIsNotNone(result)
        self.assertIn('avg_rating', result)
        self.assertIn('complexity', result)
        self.assertIsInstance(result['categories'], Counter)
        self.assertIsInstance(result['mechanics'], Counter)
    
    def test_create_user_feature_vector(self):
        """Test user feature vector creation"""
        # Setup feature info
        self.engine.feature_info = {
            'categories': ['Adventure', 'Economic'],
            'mechanics': ['Card Drafting', 'Worker Placement'],
            'designers': ['Isaac Childres'],
            'artists': ['Artist 1'],
            'publishers': ['Publisher 1']
        }
        
        # Mock scaler
        self.engine.scaler = Mock()
        self.engine.scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 0.5, 0.3, 0.2, 0.1, 0.0]])
        
        user_prefs = {
            'avg_rating': 8.0,
            'complexity': 3.0,
            'min_players': 2,
            'max_players': 4,
            'playing_time': 90,
            'year_published': 2018,
            'categories': Counter({'Adventure': 0.7, 'Economic': 0.3}),
            'mechanics': Counter({'Card Drafting': 0.6}),
            'designers': Counter({'Isaac Childres': 0.4}),
            'artists': Counter({'Artist 1': 0.3}),
            'publishers': Counter({'Publisher 1': 0.2})
        }
        
        result = self.engine._create_user_feature_vector(user_prefs)
        
        self.assertIsNotNone(result)
        self.engine.scaler.transform.assert_called_once()
    
    def test_generate_recommendations_success(self):
        """Test successful recommendation generation"""
        # Setup engine
        self.engine.create_feature_matrix(self.test_games_df)
        self.engine.train_model()
        
        # Mock user preferences
        user_prefs = {
            'avg_rating': 8.0,
            'complexity': 3.0,
            'min_players': 2,
            'max_players': 4,
            'playing_time': 90,
            'year_published': 2018,
            'categories': Counter({'Adventure': 0.7}),
            'mechanics': Counter({'Card Drafting': 0.6}),
            'designers': Counter({'Isaac Childres': 0.4}),
            'artists': Counter(),
            'publishers': Counter()
        }
        
        owned_game_ids = {174430}  # Own Gloomhaven
        
        recommendations = self.engine.generate_recommendations(
            user_prefs, self.test_games_df, owned_game_ids, 2
        )
        
        self.assertIsNotNone(recommendations)
        self.assertLessEqual(len(recommendations), 2)
        # Should not recommend owned games
        for rec in recommendations:
            self.assertNotIn(rec['id'], owned_game_ids)
    
    def test_generate_recommendations_no_model(self):
        """Test recommendation generation without trained model"""
        user_prefs = {'avg_rating': 8.0}
        
        recommendations = self.engine.generate_recommendations(
            user_prefs, self.test_games_df, set(), 5
        )
        
        self.assertEqual(recommendations, [])
    
    def test_generate_recommendations_no_preferences(self):
        """Test recommendation generation without user preferences"""
        self.engine.create_feature_matrix(self.test_games_df)
        self.engine.train_model()
        
        recommendations = self.engine.generate_recommendations(
            None, self.test_games_df, set(), 5
        )
        
        self.assertEqual(recommendations, [])
    
    def test_filter_and_rank_recommendations(self):
        """Test recommendation filtering and ranking"""
        indices = np.array([0, 1, 2])
        distances = np.array([0.1, 0.2, 0.3])
        owned_game_ids = {174430}  # Own first game
        
        with patch('ml_engine.DEBUG_SHOW_SIMILARITY_DETAILS', False):
            recommendations = self.engine._filter_and_rank_recommendations(
                self.test_games_df, indices, distances, owned_game_ids, 2
            )
        
        self.assertLessEqual(len(recommendations), 2)
        # Should not include owned game
        for rec in recommendations:
            self.assertNotIn(rec['id'], owned_game_ids)
    
    def test_filter_and_rank_recommendations_with_debug(self):
        """Test recommendation filtering with debug output"""
        indices = np.array([0, 1, 2])
        distances = np.array([0.1, 0.2, 0.3])
        owned_game_ids = set()
        
        with patch('ml_engine.DEBUG_SHOW_SIMILARITY_DETAILS', True):
            with patch('builtins.print'):
                recommendations = self.engine._filter_and_rank_recommendations(
                    self.test_games_df, indices, distances, owned_game_ids, 3
                )
        
        self.assertEqual(len(recommendations), 3)
        # Check similarity scores
        for i, rec in enumerate(recommendations):
            expected_similarity = 1 - distances[i]
            self.assertAlmostEqual(rec['similarity_score'], expected_similarity, places=3)


if __name__ == '__main__':
    unittest.main()