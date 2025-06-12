"""
Integration tests for BGG ML Recommendation System
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil
import json

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import BGGRecommender
from data_loader import BGGDataLoader
from ml_engine import BGGMLEngine
try:
    from .test_fixtures import (
        MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS,
        MOCK_TOP_GAMES, get_mock_collection_response, get_mock_game_details_response
    )
except ImportError:
    from test_fixtures import (
        MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS,
        MOCK_TOP_GAMES, get_mock_collection_response, get_mock_game_details_response
    )


class TestBGGIntegration(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment"""
        self.test_cache_dir = tempfile.mkdtemp()
        
        # Create comprehensive test data
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
                'artists': ['Artist 2'],
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
                'artists': ['Artist 3'],
                'publishers': ['Publisher 3'],
                'year_published': 2016,
                'avg_rating': 8.4,
                'complexity': 3.2,
                'min_players': 1,
                'max_players': 5,
                'playing_time': 120,
                'rank': 3
            },
            {
                'id': 220308,
                'name': 'Gaia Project',
                'categories': ['Economic', 'Science Fiction', 'Space Exploration'],
                'mechanics': ['Tile Placement', 'Variable Player Powers'],
                'designers': ['Helge Ostertag'],
                'artists': ['Artist 4'],
                'publishers': ['Publisher 4'],
                'year_published': 2017,
                'avg_rating': 8.2,
                'complexity': 4.4,
                'min_players': 1,
                'max_players': 4,
                'playing_time': 150,
                'rank': 4
            },
            {
                'id': 173346,
                'name': '7 Wonders Duel',
                'categories': ['Ancient', 'Card Game', 'Civilization'],
                'mechanics': ['Card Drafting', 'Set Collection'],
                'designers': ['Antoine Bauza'],
                'artists': ['Artist 5'],
                'publishers': ['Publisher 5'],
                'year_published': 2015,
                'avg_rating': 8.1,
                'complexity': 2.2,
                'min_players': 2,
                'max_players': 2,
                'playing_time': 30,
                'rank': 5
            }
        ])
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_cache_dir') and os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_data_loader_to_ml_engine_integration(self):
        """Test integration between data loader and ML engine"""
        # Create ML engine and test feature matrix creation
        ml_engine = BGGMLEngine()
        
        # Test feature matrix creation with test data
        result = ml_engine.create_feature_matrix(self.test_games_df)
        
        self.assertTrue(result)
        self.assertIsNotNone(ml_engine.feature_matrix)
        self.assertEqual(ml_engine.feature_matrix.shape[0], 5)  # 5 games
        
        # Test model training
        train_result = ml_engine.train_model()
        self.assertTrue(train_result)
        self.assertIsNotNone(ml_engine.ml_model)
    
    def test_ml_engine_recommendation_flow(self):
        """Test complete ML engine recommendation flow"""
        ml_engine = BGGMLEngine()
        
        # Create feature matrix and train model
        ml_engine.create_feature_matrix(self.test_games_df)
        ml_engine.train_model()
        
        # Create user preferences
        collection_data = [
            {'id': 174430, 'rating': 9.0},  # Likes Gloomhaven
            {'id': 169786, 'rating': 8.0}   # Likes Scythe
        ]
        
        plays_data = [
            {'game_id': 174430, 'quantity': 5},
            {'game_id': 169786, 'quantity': 3}
        ]
        
        game_details = {
            174430: self.test_games_df[self.test_games_df['id'] == 174430].iloc[0].to_dict(),
            169786: self.test_games_df[self.test_games_df['id'] == 169786].iloc[0].to_dict()
        }
        
        user_preferences = ml_engine.create_user_preferences_vector(
            collection_data, plays_data, game_details
        )
        
        self.assertIsNotNone(user_preferences)
        
        # Generate recommendations (excluding owned games)
        owned_game_ids = {174430, 169786}
        recommendations = ml_engine.generate_recommendations(
            user_preferences, self.test_games_df, owned_game_ids, 3
        )
        
        self.assertIsNotNone(recommendations)
        self.assertLessEqual(len(recommendations), 3)
        
        # Verify no owned games in recommendations
        for rec in recommendations:
            self.assertNotIn(rec['id'], owned_game_ids)
    
    def test_full_system_integration_with_mocks(self):
        """Test full system integration with mocked external dependencies"""
        recommender = BGGRecommender('testuser')
        
        # Mock external API calls
        recommender.data_loader.fetch_user_collection = Mock(return_value=MOCK_COLLECTION_DATA)
        recommender.data_loader.fetch_user_plays = Mock(return_value=MOCK_PLAYS_DATA)
        recommender.data_loader.load_top500_games = Mock(return_value=MOCK_TOP_GAMES)
        recommender.data_loader.load_game_details_cache = Mock(return_value=MOCK_GAME_DETAILS)
        recommender.data_loader.fetch_game_details = Mock(return_value={})
        
        # Mock DataFrame creation to return our test data
        with patch.object(recommender, '_create_games_dataframe', return_value=self.test_games_df):
            # Test complete flow
            user_data_result = recommender.load_user_data()
            self.assertTrue(user_data_result)
            
            top_games_result = recommender.load_top_games_data()
            self.assertTrue(top_games_result)
            
            ml_training_result = recommender.train_ml_model()
            self.assertTrue(ml_training_result)
            
            recommendations = recommender.generate_recommendations(3)
            self.assertIsNotNone(recommendations)
    
    def test_cache_integration(self):
        """Test cache system integration"""
        data_loader = BGGDataLoader()
        
        # Test saving and loading game details cache
        test_cache_file = os.path.join(self.test_cache_dir, 'test_game_details.json')
        
        with patch('data_loader.GAME_DETAILS_FILE', test_cache_file):
            # Save cache
            data_loader.save_game_details_cache(MOCK_GAME_DETAILS)
            
            # Verify file exists
            self.assertTrue(os.path.exists(test_cache_file))
            
            # Load cache
            loaded_details = data_loader.load_game_details_cache()
            
            # Verify loaded data matches original
            self.assertEqual(len(loaded_details), len(MOCK_GAME_DETAILS))
            for game_id in MOCK_GAME_DETAILS:
                self.assertIn(game_id, loaded_details)
    
    def test_duplicate_handling_integration(self):
        """Test duplicate handling across the system"""
        # Create data with duplicates
        games_with_duplicates = [
            {'id': 1, 'name': 'Game 1', 'rank': 1},
            {'id': 2, 'name': 'Game 2', 'rank': 2},
            {'id': 1, 'name': 'Game 1 Duplicate', 'rank': 3},  # Duplicate ID
            {'id': 3, 'name': 'Game 3', 'rank': 4}
        ]
        
        data_loader = BGGDataLoader()
        
        # Test duplicate removal
        unique_games = data_loader.remove_duplicates_from_games(games_with_duplicates)
        self.assertEqual(len(unique_games), 3)
        
        # Test DataFrame duplicate handling
        df_with_duplicates = pd.DataFrame([
            {'id': 1, 'name': 'Game 1', 'categories': ['Strategy'], 'mechanics': ['Engine Building'],
             'designers': ['Designer 1'], 'artists': ['Artist 1'], 'publishers': ['Publisher 1'],
             'year_published': 2020, 'avg_rating': 8.0, 'complexity': 3.0,
             'min_players': 2, 'max_players': 4, 'playing_time': 90, 'rank': 1},
            {'id': 1, 'name': 'Game 1 Duplicate', 'categories': ['Strategy'], 'mechanics': ['Engine Building'],
             'designers': ['Designer 1'], 'artists': ['Artist 1'], 'publishers': ['Publisher 1'],
             'year_published': 2020, 'avg_rating': 8.0, 'complexity': 3.0,
             'min_players': 2, 'max_players': 4, 'playing_time': 90, 'rank': 1}
        ])
        
        ml_engine = BGGMLEngine()
        result = ml_engine.create_feature_matrix(df_with_duplicates)
        
        self.assertTrue(result)
        # Should have only 1 game after duplicate removal
        self.assertEqual(ml_engine.feature_matrix.shape[0], 1)
    
    def test_error_handling_integration(self):
        """Test error handling across system components"""
        recommender = BGGRecommender('testuser')
        
        # Test with API failures
        recommender.data_loader.fetch_user_collection = Mock(return_value=None)
        
        result = recommender.load_user_data()
        self.assertFalse(result)
        
        # Test with empty top games
        recommender.data_loader.load_top500_games = Mock(return_value=None)
        
        result = recommender.load_top_games_data()
        self.assertFalse(result)
        
        # Test ML engine with no data
        ml_engine = BGGMLEngine()
        result = ml_engine.create_feature_matrix(None)
        self.assertFalse(result)
        
        result = ml_engine.train_model()
        self.assertFalse(result)
    
    def test_feature_engineering_integration(self):
        """Test feature engineering pipeline integration"""
        ml_engine = BGGMLEngine()
        
        # Test with comprehensive game data
        result = ml_engine.create_feature_matrix(self.test_games_df)
        self.assertTrue(result)
        
        # Verify feature dimensions
        n_games = len(self.test_games_df)
        n_features = ml_engine.feature_matrix.shape[1]
        
        self.assertEqual(ml_engine.feature_matrix.shape[0], n_games)
        self.assertGreater(n_features, 7)  # Should have more than 7 numeric features
        
        # Verify feature info structure
        self.assertIn('categories', ml_engine.feature_info)
        self.assertIn('mechanics', ml_engine.feature_info)
        self.assertIn('designers', ml_engine.feature_info)
        self.assertIn('artists', ml_engine.feature_info)
        self.assertIn('publishers', ml_engine.feature_info)
        
        # Test normalization
        feature_means = np.mean(ml_engine.feature_matrix, axis=0)
        feature_stds = np.std(ml_engine.feature_matrix, axis=0)
        
        # Features should be approximately normalized (mean ~0, std ~1)
        self.assertTrue(np.allclose(feature_means, 0, atol=1e-10))
        # Note: std might not be exactly 1 due to sklearn's StandardScaler behavior
    
    def test_recommendation_quality_integration(self):
        """Test recommendation quality and relevance"""
        ml_engine = BGGMLEngine()
        
        # Setup and train model
        ml_engine.create_feature_matrix(self.test_games_df)
        ml_engine.train_model()
        
        # Create user who likes economic and science fiction games
        collection_data = [
            {'id': 169786, 'rating': 9.0},   # Scythe (Economic, Science Fiction)
            {'id': 167791, 'rating': 8.5}    # Terraforming Mars (Economic, Science Fiction)
        ]
        
        plays_data = [
            {'game_id': 169786, 'quantity': 10},
            {'game_id': 167791, 'quantity': 8}
        ]
        
        game_details = {
            169786: self.test_games_df[self.test_games_df['id'] == 169786].iloc[0].to_dict(),
            167791: self.test_games_df[self.test_games_df['id'] == 167791].iloc[0].to_dict()
        }
        
        user_preferences = ml_engine.create_user_preferences_vector(
            collection_data, plays_data, game_details
        )
        
        owned_game_ids = {169786, 167791}
        recommendations = ml_engine.generate_recommendations(
            user_preferences, self.test_games_df, owned_game_ids, 2
        )
        
        # Should get recommendations
        self.assertGreater(len(recommendations), 0)
        
        # Gaia Project should be recommended (also Economic + Science Fiction)
        recommended_ids = [rec['id'] for rec in recommendations]
        self.assertIn(220308, recommended_ids)  # Gaia Project ID
        
        # Verify similarity scores are reasonable
        for rec in recommendations:
            self.assertGreater(rec['similarity_score'], 0)
            self.assertLessEqual(rec['similarity_score'], 1)


if __name__ == '__main__':
    unittest.main()