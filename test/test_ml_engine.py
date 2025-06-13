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
from config import MIN_FEATURE_FREQUENCY, MAX_NEIGHBORS, DEFAULT_NUM_RECOMMENDATIONS, EXCLUDE_BGG_RATING_FROM_FEATURES
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
        # Features include numeric (6 or 7) + categorical features
        expected_min_features = 6 if EXCLUDE_BGG_RATING_FROM_FEATURES else 7
        self.assertGreaterEqual(self.engine.feature_matrix.shape[1], expected_min_features)
    
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
        
        mock_user_prefs = {
            'complexity': 3.0, 
            'playing_time': 90,
            'designer_loyalty': Counter(),
            'preferred_eras': Counter(),
            'categories': Counter({'Adventure': 0.7}),
            'mechanics': Counter({'Card Drafting': 0.6}),
            'designers': Counter({'Isaac Childres': 0.4}),
            'artists': Counter(),
            'publishers': Counter()
        }
        with patch('ml_engine.DEBUG_SHOW_SIMILARITY_DETAILS', False):
            recommendations = self.engine._advanced_filter_and_rank_recommendations(
                self.test_games_df, indices, distances, owned_game_ids, mock_user_prefs, 2
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
        
        mock_user_prefs = {'complexity': 3.0, 'playing_time': 90}
        with patch('ml_engine.DEBUG_SHOW_SIMILARITY_DETAILS', True):
            with patch('builtins.print'):
                recommendations = self.engine._advanced_filter_and_rank_recommendations(
                    self.test_games_df, indices, distances, owned_game_ids, mock_user_prefs, 3
                )
        
        self.assertEqual(len(recommendations), 3)
        # Check similarity scores
        for i, rec in enumerate(recommendations):
            expected_similarity = 1 - distances[i]
            self.assertAlmostEqual(rec['similarity_score'], expected_similarity, places=3)

    def test_feature_matrix_with_bgg_rating_excluded(self):
        """Test feature matrix creation with BGG rating excluded"""
        with patch('ml_engine.EXCLUDE_BGG_RATING_FROM_FEATURES', True):
            engine = BGGMLEngine()
            result = engine.create_feature_matrix(self.test_games_df)
            
            self.assertTrue(result)
            # Should have 6 numeric features instead of 7 when BGG rating excluded
            # Plus categorical features (varies based on data)
            expected_min_features = 6  # numeric only
            self.assertGreaterEqual(engine.feature_matrix.shape[1], expected_min_features)
            
            # Feature matrix should not be None
            self.assertIsNotNone(engine.feature_matrix)

    def test_feature_matrix_with_bgg_rating_included(self):
        """Test feature matrix creation with BGG rating included"""
        with patch('ml_engine.EXCLUDE_BGG_RATING_FROM_FEATURES', False):
            engine = BGGMLEngine()
            result = engine.create_feature_matrix(self.test_games_df)
            
            self.assertTrue(result)
            # Should have 7 numeric features when BGG rating included
            expected_min_features = 7  # numeric only
            self.assertGreaterEqual(engine.feature_matrix.shape[1], expected_min_features)

    def test_feature_matrix_shape_difference(self):
        """Test that feature matrix has different shapes with/without BGG rating"""
        # Test with BGG rating excluded
        with patch('ml_engine.EXCLUDE_BGG_RATING_FROM_FEATURES', True):
            engine_excluded = BGGMLEngine()
            engine_excluded.create_feature_matrix(self.test_games_df)
            shape_excluded = engine_excluded.feature_matrix.shape
        
        # Test with BGG rating included
        with patch('ml_engine.EXCLUDE_BGG_RATING_FROM_FEATURES', False):
            engine_included = BGGMLEngine()
            engine_included.create_feature_matrix(self.test_games_df)
            shape_included = engine_included.feature_matrix.shape
        
        # Should have 1 less feature when excluded
        self.assertEqual(shape_included[1] - shape_excluded[1], 1)
        # Same number of games
        self.assertEqual(shape_included[0], shape_excluded[0])

    def test_user_feature_vector_with_bgg_rating_excluded(self):
        """Test user feature vector creation with BGG rating excluded"""
        import config
        original_setting = config.EXCLUDE_BGG_RATING_FROM_FEATURES
        
        try:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = True
            
            engine = BGGMLEngine()
            engine.create_feature_matrix(self.test_games_df)  # Setup feature_info
            
            # Mock scaler
            engine.scaler = Mock()
            # Expected: 6 numeric + 8 categorical features (based on test data)
            engine.scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]])
            
            user_prefs = {
                'avg_rating': 8.0,  # Still calculated but not used in vector
                'complexity': 3.0,
                'min_players': 2,
                'max_players': 4,
                'playing_time': 90,
                'year_published': 2018,
                'categories': Counter({'Adventure': 0.7}),
                'mechanics': Counter({'Card Drafting': 0.6}),
                'designers': Counter(),
                'artists': Counter(),
                'publishers': Counter()
            }
            
            result = engine._create_user_feature_vector(user_prefs)
            
            self.assertIsNotNone(result)
            engine.scaler.transform.assert_called_once()
            
            # Verify the input to scaler.transform doesn't include avg_rating at position 0
            call_args = engine.scaler.transform.call_args[0][0]
            expected_first_feature = 3.0  # complexity should be first, not avg_rating
            self.assertEqual(call_args[0, 0], expected_first_feature)
            
        finally:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = original_setting

    def test_user_feature_vector_with_bgg_rating_included(self):
        """Test user feature vector creation with BGG rating included"""
        import config
        original_setting = config.EXCLUDE_BGG_RATING_FROM_FEATURES
        
        try:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = False
            
            engine = BGGMLEngine()
            engine.create_feature_matrix(self.test_games_df)  # Setup feature_info
            
            # Mock scaler
            engine.scaler = Mock()
            # Expected: 7 numeric + categorical features
            engine.scaler.transform.return_value = np.array([[1, 2, 3, 4, 5, 6, 7, 0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0]])
            
            user_prefs = {
                'avg_rating': 8.0,
                'complexity': 3.0,
                'min_players': 2,
                'max_players': 4,
                'playing_time': 90,
                'year_published': 2018,
                'categories': Counter({'Adventure': 0.7}),
                'mechanics': Counter({'Card Drafting': 0.6}),
                'designers': Counter(),
                'artists': Counter(),
                'publishers': Counter()
            }
            
            result = engine._create_user_feature_vector(user_prefs)
            
            self.assertIsNotNone(result)
            engine.scaler.transform.assert_called_once()
            
            # Verify the input to scaler.transform includes avg_rating at position 0
            call_args = engine.scaler.transform.call_args[0][0]
            expected_first_feature = 8.0  # avg_rating should be first
            self.assertEqual(call_args[0, 0], expected_first_feature)
            
        finally:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = original_setting

    def test_feature_count_reporting(self):
        """Test that feature count reporting is accurate with/without BGG rating"""
        import config
        original_setting = config.EXCLUDE_BGG_RATING_FROM_FEATURES
        
        try:
            # Test with BGG rating excluded
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = True
            engine_excluded = BGGMLEngine()
            
            # Capture print output
            with patch('builtins.print') as mock_print:
                engine_excluded.create_feature_matrix(self.test_games_df)
            
            # Find the call that contains "6 numerisch" 
            numeric_feature_calls = [call for call in mock_print.call_args_list 
                                   if len(call[0]) > 0 and "numerisch" in str(call[0][0])]
            self.assertTrue(any("6 numerisch" in str(call[0][0]) for call in numeric_feature_calls))
            
            # Test with BGG rating included
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = False
            engine_included = BGGMLEngine()
            
            with patch('builtins.print') as mock_print:
                engine_included.create_feature_matrix(self.test_games_df)
            
            # Find the call that contains "7 numerisch"
            numeric_feature_calls = [call for call in mock_print.call_args_list 
                                   if len(call[0]) > 0 and "numerisch" in str(call[0][0])]
            self.assertTrue(any("7 numerisch" in str(call[0][0]) for call in numeric_feature_calls))
            
        finally:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = original_setting

    def test_recommendations_still_work_without_bgg_rating(self):
        """Test that recommendation generation still works when BGG rating excluded from features"""
        import config
        original_setting = config.EXCLUDE_BGG_RATING_FROM_FEATURES
        
        try:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = True
            
            engine = BGGMLEngine()
            engine.create_feature_matrix(self.test_games_df)
            engine.train_model()
            
            user_prefs = {
                'avg_rating': 8.0,  # Still calculated for preferences
                'complexity': 3.0,
                'min_players': 2,
                'max_players': 4,
                'playing_time': 90,
                'year_published': 2018,
                'categories': Counter({'Adventure': 0.7}),
                'mechanics': Counter({'Card Drafting': 0.6}),
                'designers': Counter(),
                'artists': Counter(),
                'publishers': Counter()
            }
            
            owned_game_ids = {174430}  # Own Gloomhaven
            
            recommendations = engine.generate_recommendations(
                user_prefs, self.test_games_df, owned_game_ids, 2
            )
            
            # Should still generate recommendations
            self.assertIsNotNone(recommendations)
            # Should not recommend owned games
            for rec in recommendations:
                self.assertNotIn(rec['id'], owned_game_ids)
                
        finally:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = original_setting

    def test_bgg_rating_still_used_as_quality_filter(self):
        """Test that BGG rating is still used as quality filter even when excluded from features"""
        import config
        original_setting = config.EXCLUDE_BGG_RATING_FROM_FEATURES
        
        try:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = True
            
            # Create test data with low-rated game
            test_df = self.test_games_df.copy()
            low_rated_game = {
                'id': 999999,
                'name': 'Bad Game',
                'categories': ['Adventure'],
                'mechanics': ['Action Points'],
                'designers': ['Bad Designer'],
                'artists': ['Bad Artist'],
                'publishers': ['Bad Publisher'],
                'year_published': 2020,
                'avg_rating': 4.0,  # Below 5.0 threshold
                'complexity': 2.0,
                'min_players': 2,
                'max_players': 4,
                'playing_time': 60,
                'rank': 9999
            }
            test_df = pd.concat([test_df, pd.DataFrame([low_rated_game])], ignore_index=True)
            
            engine = BGGMLEngine()
            engine.create_feature_matrix(test_df)
            engine.train_model()
            
            # Mock user preferences
            user_prefs = {
                'avg_rating': 8.0,
                'complexity': 2.0,  # Similar to bad game
                'min_players': 2,
                'max_players': 4,
                'playing_time': 60,  # Similar to bad game
                'year_published': 2020,
                'categories': Counter({'Adventure': 1.0}),  # Same category
                'mechanics': Counter({'Action Points': 1.0}),  # Same mechanic
                'designers': Counter(),
                'artists': Counter(),
                'publishers': Counter()
            }
            
            owned_game_ids = set()
            
            recommendations = engine.generate_recommendations(
                user_prefs, test_df, owned_game_ids, 10
            )
            
            # Bad game should be filtered out despite similar features
            bad_game_recommended = any(rec['id'] == 999999 for rec in recommendations)
            self.assertFalse(bad_game_recommended, "Low-rated game should be filtered out by quality filter")
            
        finally:
            config.EXCLUDE_BGG_RATING_FROM_FEATURES = original_setting


if __name__ == '__main__':
    unittest.main()