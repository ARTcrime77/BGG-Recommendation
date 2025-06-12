"""
Unit tests for BGGRecommender class (main.py)
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import BGGRecommender
try:
    from .test_fixtures import (
        MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS,
        MOCK_TOP_GAMES
    )
except ImportError:
    from test_fixtures import (
        MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS,
        MOCK_TOP_GAMES
    )


class TestBGGRecommender(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment"""
        self.recommender = BGGRecommender('testuser')
    
    def test_init(self):
        """Test BGGRecommender initialization"""
        self.assertEqual(self.recommender.username, 'testuser')
        self.assertIsNotNone(self.recommender.data_loader)
        self.assertIsNotNone(self.recommender.ml_engine)
        self.assertIsNone(self.recommender.collection_data)
        self.assertIsNone(self.recommender.plays_data)
        self.assertEqual(self.recommender.game_details, {})
        self.assertIsNone(self.recommender.top_games_data)
    
    def test_load_user_data_success(self):
        """Test successful user data loading"""
        # Mock the data loader methods
        self.recommender.data_loader.fetch_user_collection = Mock(return_value=MOCK_COLLECTION_DATA)
        self.recommender.data_loader.fetch_user_plays = Mock(return_value=MOCK_PLAYS_DATA)
        self.recommender.data_loader.fetch_game_details = Mock(return_value=MOCK_GAME_DETAILS)
        
        result = self.recommender.load_user_data()
        
        self.assertTrue(result)
        self.assertEqual(self.recommender.collection_data, MOCK_COLLECTION_DATA)
        self.assertEqual(self.recommender.plays_data, MOCK_PLAYS_DATA)
        self.assertEqual(len(self.recommender.game_details), len(MOCK_GAME_DETAILS))
    
    def test_load_user_data_no_collection(self):
        """Test user data loading when no collection is found"""
        self.recommender.data_loader.fetch_user_collection = Mock(return_value=None)
        
        result = self.recommender.load_user_data()
        
        self.assertFalse(result)
        self.assertIsNone(self.recommender.collection_data)
    
    def test_load_user_data_no_plays(self):
        """Test user data loading when no plays are found"""
        self.recommender.data_loader.fetch_user_collection = Mock(return_value=MOCK_COLLECTION_DATA)
        self.recommender.data_loader.fetch_user_plays = Mock(return_value=None)
        self.recommender.data_loader.fetch_game_details = Mock(return_value=MOCK_GAME_DETAILS)
        
        result = self.recommender.load_user_data()
        
        self.assertTrue(result)
        self.assertEqual(self.recommender.collection_data, MOCK_COLLECTION_DATA)
        self.assertIsNone(self.recommender.plays_data)
    
    def test_load_top_games_data_success(self):
        """Test successful top games data loading"""
        self.recommender.data_loader.load_top500_games = Mock(return_value=MOCK_TOP_GAMES)
        self.recommender.data_loader.load_game_details_cache = Mock(return_value=MOCK_GAME_DETAILS)
        self.recommender.data_loader.fetch_game_details = Mock(return_value={})
        self.recommender.data_loader.ask_user_update_choice = Mock(return_value=False)
        
        with patch.object(self.recommender, '_create_games_dataframe') as mock_create_df:
            mock_df = pd.DataFrame(MOCK_TOP_GAMES)
            mock_create_df.return_value = mock_df
            
            result = self.recommender.load_top_games_data()
        
        self.assertTrue(result)
        self.assertIsNotNone(self.recommender.top_games_data)
    
    def test_load_top_games_data_no_games(self):
        """Test top games data loading when no games are found"""
        self.recommender.data_loader.load_top500_games = Mock(return_value=None)
        
        result = self.recommender.load_top_games_data()
        
        self.assertFalse(result)
    
    def test_load_top_games_data_missing_details(self):
        """Test top games data loading with missing game details"""
        self.recommender.data_loader.load_top500_games = Mock(return_value=MOCK_TOP_GAMES)
        self.recommender.data_loader.load_game_details_cache = Mock(return_value={})
        self.recommender.data_loader.ask_user_update_choice = Mock(return_value=True)
        self.recommender.data_loader.fetch_game_details = Mock(return_value=MOCK_GAME_DETAILS)
        self.recommender.data_loader.save_game_details_cache = Mock()
        
        with patch.object(self.recommender, '_create_games_dataframe') as mock_create_df:
            mock_df = pd.DataFrame(MOCK_TOP_GAMES)
            mock_create_df.return_value = mock_df
            
            result = self.recommender.load_top_games_data()
        
        self.assertTrue(result)
        self.recommender.data_loader.fetch_game_details.assert_called_once()
        self.recommender.data_loader.save_game_details_cache.assert_called_once()
    
    def test_create_games_dataframe(self):
        """Test games DataFrame creation"""
        # Setup game details in recommender
        self.recommender.game_details = MOCK_GAME_DETAILS
        
        games_df = self.recommender._create_games_dataframe(MOCK_TOP_GAMES)
        
        self.assertIsInstance(games_df, pd.DataFrame)
        self.assertGreater(len(games_df), 0)
        self.assertIn('id', games_df.columns)
        self.assertIn('name', games_df.columns)
        self.assertIn('categories', games_df.columns)
    
    def test_create_games_dataframe_with_duplicates(self):
        """Test games DataFrame creation with duplicate IDs"""
        # Add duplicate game to test data
        games_with_duplicates = MOCK_TOP_GAMES + [{'rank': 6, 'id': 174430, 'name': 'Duplicate Game'}]
        self.recommender.game_details = MOCK_GAME_DETAILS
        
        with patch('main.DEBUG_SHOW_SIMILARITY_DETAILS', False):
            games_df = self.recommender._create_games_dataframe(games_with_duplicates)
        
        # Should remove duplicates
        unique_ids = games_df['id'].nunique()
        self.assertEqual(unique_ids, len(games_df))
    
    def test_train_ml_model_success(self):
        """Test successful ML model training"""
        # Mock the ML engine methods
        self.recommender.ml_engine.create_feature_matrix = Mock(return_value=True)
        self.recommender.ml_engine.train_model = Mock(return_value=True)
        
        result = self.recommender.train_ml_model()
        
        self.assertTrue(result)
        self.recommender.ml_engine.create_feature_matrix.assert_called_once()
        self.recommender.ml_engine.train_model.assert_called_once()
    
    def test_train_ml_model_feature_matrix_failure(self):
        """Test ML model training with feature matrix creation failure"""
        self.recommender.ml_engine.create_feature_matrix = Mock(return_value=False)
        
        result = self.recommender.train_ml_model()
        
        self.assertFalse(result)
    
    def test_train_ml_model_training_failure(self):
        """Test ML model training with model training failure"""
        self.recommender.ml_engine.create_feature_matrix = Mock(return_value=True)
        self.recommender.ml_engine.train_model = Mock(return_value=False)
        
        result = self.recommender.train_ml_model()
        
        self.assertFalse(result)
    
    def test_generate_recommendations_success(self):
        """Test successful recommendation generation"""
        # Setup test data
        self.recommender.collection_data = MOCK_COLLECTION_DATA
        self.recommender.plays_data = MOCK_PLAYS_DATA
        self.recommender.game_details = MOCK_GAME_DETAILS
        
        mock_recommendations = [
            {
                'id': 220308,
                'name': 'Gaia Project',
                'rank': 4,
                'avg_rating': 8.2,
                'complexity': 4.4,
                'categories': ['Economic', 'Science Fiction'],
                'mechanics': ['Variable Player Powers'],
                'designers': ['Helge Ostertag'],
                'artists': ['Dennis Lohausen'],
                'year_published': 2017,
                'similarity_score': 0.85
            }
        ]
        
        # Mock ML engine methods
        self.recommender.ml_engine.create_user_preferences_vector = Mock(return_value={'avg_rating': 8.0})
        self.recommender.ml_engine.generate_recommendations = Mock(return_value=mock_recommendations)
        
        recommendations = self.recommender.generate_recommendations(5)
        
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0]['name'], 'Gaia Project')
    
    def test_generate_recommendations_no_preferences(self):
        """Test recommendation generation when user preferences cannot be created"""
        self.recommender.collection_data = MOCK_COLLECTION_DATA
        self.recommender.plays_data = MOCK_PLAYS_DATA
        self.recommender.game_details = MOCK_GAME_DETAILS
        
        # Mock ML engine to return None preferences
        self.recommender.ml_engine.create_user_preferences_vector = Mock(return_value=None)
        
        recommendations = self.recommender.generate_recommendations(5)
        
        self.assertEqual(recommendations, [])
    
    def test_display_recommendations_with_recommendations(self):
        """Test displaying recommendations when recommendations exist"""
        mock_recommendations = [
            {
                'id': 220308,
                'name': 'Gaia Project',
                'rank': 4,
                'avg_rating': 8.2,
                'complexity': 4.4,
                'categories': ['Economic', 'Science Fiction'],
                'mechanics': ['Variable Player Powers'],
                'designers': ['Helge Ostertag'],
                'artists': ['Dennis Lohausen'],
                'year_published': 2017,
                'similarity_score': 0.85
            }
        ]
        
        with patch('builtins.print') as mock_print:
            self.recommender.display_recommendations(mock_recommendations)
            
            # Should print recommendation details
            mock_print.assert_called()
            # Check that the game name appears in one of the print calls
            printed_text = ''.join(str(call) for call in mock_print.call_args_list)
            self.assertIn('Gaia Project', printed_text)
    
    def test_display_recommendations_empty(self):
        """Test displaying recommendations when no recommendations exist"""
        with patch('builtins.print') as mock_print:
            self.recommender.display_recommendations([])
            
            # Should print "no recommendations found" message
            mock_print.assert_called()
            printed_text = str(mock_print.call_args_list[-1])
            self.assertIn('Keine Empfehlungen', printed_text)
    
    @patch('os.path.exists')
    @patch('os.path.getmtime')
    def test_show_cache_info(self, mock_getmtime, mock_exists):
        """Test cache information display"""
        mock_exists.return_value = True
        mock_getmtime.return_value = 1640995200  # Fixed timestamp
        
        with patch('builtins.print') as mock_print:
            self.recommender.show_cache_info()
            
            mock_print.assert_called()
            # Should show cache information
            printed_text = ''.join(str(call) for call in mock_print.call_args_list)
            self.assertIn('Cache-Info', printed_text)
    
    def test_run_analysis_success(self):
        """Test successful complete analysis run"""
        # Mock all the required methods
        self.recommender.load_user_data = Mock(return_value=True)
        self.recommender.load_top_games_data = Mock(return_value=True)
        self.recommender.train_ml_model = Mock(return_value=True)
        self.recommender.generate_recommendations = Mock(return_value=[{'name': 'Test Game'}])
        self.recommender.display_recommendations = Mock()
        self.recommender.show_cache_info = Mock()
        
        with patch('builtins.print'):
            self.recommender.run_analysis()
        
        # Verify all steps were called
        self.recommender.load_user_data.assert_called_once()
        self.recommender.load_top_games_data.assert_called_once()
        self.recommender.train_ml_model.assert_called_once()
        self.recommender.generate_recommendations.assert_called_once()
        self.recommender.display_recommendations.assert_called_once()
        self.recommender.show_cache_info.assert_called_once()
    
    def test_run_analysis_user_data_failure(self):
        """Test analysis run when user data loading fails"""
        self.recommender.load_user_data = Mock(return_value=False)
        
        with patch('builtins.print'):
            self.recommender.run_analysis()
        
        # Should stop after user data failure
        self.recommender.load_user_data.assert_called_once()
    
    def test_run_analysis_top_games_failure(self):
        """Test analysis run when top games loading fails"""
        self.recommender.load_user_data = Mock(return_value=True)
        self.recommender.load_top_games_data = Mock(return_value=False)
        
        with patch('builtins.print'):
            self.recommender.run_analysis()
        
        # Should stop after top games failure
        self.recommender.load_user_data.assert_called_once()
        self.recommender.load_top_games_data.assert_called_once()
    
    def test_run_analysis_ml_training_failure(self):
        """Test analysis run when ML training fails"""
        self.recommender.load_user_data = Mock(return_value=True)
        self.recommender.load_top_games_data = Mock(return_value=True)
        self.recommender.train_ml_model = Mock(return_value=False)
        
        with patch('builtins.print'):
            self.recommender.run_analysis()
        
        # Should stop after ML training failure
        self.recommender.load_user_data.assert_called_once()
        self.recommender.load_top_games_data.assert_called_once()
        self.recommender.train_ml_model.assert_called_once()


class TestMainFunction(unittest.TestCase):
    
    @patch('main.BGGRecommender')
    def test_main_function(self, mock_recommender_class):
        """Test main function execution"""
        mock_recommender = Mock()
        mock_recommender_class.return_value = mock_recommender
        
        with patch('builtins.print'):
            from main import main
            main()
        
        # Should create recommender with hardcoded username
        mock_recommender_class.assert_called_once_with("Artcrime77")
        mock_recommender.run_analysis.assert_called_once()


if __name__ == '__main__':
    unittest.main()