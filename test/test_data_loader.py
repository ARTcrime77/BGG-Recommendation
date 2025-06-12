"""
Unit tests for BGGDataLoader class
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
import os
import json
import tempfile
import shutil
from datetime import datetime, timedelta
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import BGGDataLoader
from config import CACHE_DIR, TOP500_FILE, GAME_DETAILS_FILE
from test_fixtures import (
    MOCK_COLLECTION_DATA, MOCK_PLAYS_DATA, MOCK_GAME_DETAILS,
    MOCK_TOP_GAMES, MOCK_TOP500_CACHE, MOCK_GAME_DETAILS_CACHE,
    get_mock_collection_response, get_mock_game_details_response,
    get_mock_plays_response, get_mock_scraping_response
)


class TestBGGDataLoader(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment"""
        self.loader = BGGDataLoader()
        # Create temporary directory for cache tests
        self.test_cache_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'test_cache_dir') and os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def test_init_creates_cache_directory(self):
        """Test that BGGDataLoader creates cache directory on initialization"""
        with patch('os.path.exists', return_value=False) as mock_exists:
            with patch('os.makedirs') as mock_makedirs:
                loader = BGGDataLoader()
                mock_exists.assert_called_with(CACHE_DIR)
                mock_makedirs.assert_called_with(CACHE_DIR)
    
    def test_should_update_cache_file_missing(self):
        """Test cache update check when file doesn't exist"""
        with patch('os.path.exists', return_value=False):
            result = self.loader.should_update_cache('nonexistent_file.json')
            self.assertTrue(result)
    
    def test_should_update_cache_file_old(self):
        """Test cache update check when file is old"""
        old_time = datetime.now() - timedelta(days=10)
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getmtime', return_value=old_time.timestamp()):
                result = self.loader.should_update_cache('old_file.json', max_age_days=7)
                self.assertTrue(result)
    
    def test_should_update_cache_file_fresh(self):
        """Test cache update check when file is fresh"""
        fresh_time = datetime.now() - timedelta(days=3)
        with patch('os.path.exists', return_value=True):
            with patch('os.path.getmtime', return_value=fresh_time.timestamp()):
                result = self.loader.should_update_cache('fresh_file.json', max_age_days=7)
                self.assertFalse(result)
    
    @patch('builtins.input')
    def test_ask_user_update_choice_yes(self, mock_input):
        """Test user choice prompt for cache update - yes"""
        mock_input.return_value = 'j'
        result = self.loader.ask_user_update_choice('Test Cache')
        self.assertTrue(result)
    
    @patch('builtins.input')
    def test_ask_user_update_choice_no(self, mock_input):
        """Test user choice prompt for cache update - no"""
        mock_input.return_value = 'n'
        result = self.loader.ask_user_update_choice('Test Cache')
        self.assertFalse(result)
    
    @patch('builtins.input')
    def test_ask_user_update_choice_invalid_then_valid(self, mock_input):
        """Test user choice prompt with invalid input then valid"""
        mock_input.side_effect = ['invalid', 'j']
        result = self.loader.ask_user_update_choice('Test Cache')
        self.assertTrue(result)
        self.assertEqual(mock_input.call_count, 2)
    
    def test_remove_duplicates_from_games(self):
        """Test duplicate removal from games list"""
        games_with_duplicates = [
            {'id': 1, 'name': 'Game 1'},
            {'id': 2, 'name': 'Game 2'},
            {'id': 1, 'name': 'Game 1 Duplicate'},
            {'id': 3, 'name': 'Game 3'}
        ]
        
        unique_games = self.loader.remove_duplicates_from_games(games_with_duplicates)
        
        self.assertEqual(len(unique_games), 3)
        game_ids = [game['id'] for game in unique_games]
        self.assertEqual(game_ids, [1, 2, 3])
    
    def test_remove_duplicates_no_duplicates(self):
        """Test duplicate removal when no duplicates exist"""
        games_no_duplicates = [
            {'id': 1, 'name': 'Game 1'},
            {'id': 2, 'name': 'Game 2'},
            {'id': 3, 'name': 'Game 3'}
        ]
        
        unique_games = self.loader.remove_duplicates_from_games(games_no_duplicates)
        
        self.assertEqual(len(unique_games), 3)
        self.assertEqual(unique_games, games_no_duplicates)
    
    @patch('requests.get')
    def test_fetch_user_collection_success(self, mock_get):
        """Test successful user collection fetch"""
        mock_get.return_value = get_mock_collection_response()
        
        collection = self.loader.fetch_user_collection('testuser')
        
        self.assertIsNotNone(collection)
        self.assertEqual(len(collection), 2)
        self.assertEqual(collection[0]['id'], 174430)
        self.assertEqual(collection[0]['name'], 'Gloomhaven')
        self.assertEqual(collection[0]['rating'], 8.5)
    
    @patch('requests.get')
    def test_fetch_user_collection_error(self, mock_get):
        """Test user collection fetch with API error"""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        collection = self.loader.fetch_user_collection('nonexistentuser')
        
        self.assertIsNone(collection)
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_fetch_user_plays_success(self, mock_sleep, mock_get):
        """Test successful user plays fetch"""
        mock_get.return_value = get_mock_plays_response()
        
        plays = self.loader.fetch_user_plays('testuser', pages=1)
        
        self.assertIsNotNone(plays)
        self.assertEqual(len(plays), 2)
        self.assertEqual(plays[0]['game_id'], 174430)
        self.assertEqual(plays[0]['quantity'], 1)
    
    @patch('requests.get')
    @patch('time.sleep')
    def test_fetch_game_details_success(self, mock_sleep, mock_get):
        """Test successful game details fetch"""
        mock_get.return_value = get_mock_game_details_response()
        
        game_details = self.loader.fetch_game_details([174430])
        
        self.assertIsNotNone(game_details)
        self.assertIn(174430, game_details)
        game = game_details[174430]
        self.assertEqual(game['name'], 'Gloomhaven')
        self.assertEqual(game['avg_rating'], 8.77729)
        self.assertIn('Adventure', game['categories'])
    
    def test_fetch_game_details_empty_list(self):
        """Test game details fetch with empty game list"""
        game_details = self.loader.fetch_game_details([])
        self.assertEqual(game_details, {})
    
    def test_extract_int_value_valid(self):
        """Test integer value extraction with valid element"""
        mock_elem = Mock()
        mock_elem.get.return_value = '5'
        
        result = self.loader._extract_int_value(mock_elem, 2)
        self.assertEqual(result, 5)
    
    def test_extract_int_value_invalid(self):
        """Test integer value extraction with invalid element"""
        mock_elem = Mock()
        mock_elem.get.return_value = 'invalid'
        
        result = self.loader._extract_int_value(mock_elem, 2)
        self.assertEqual(result, 2)  # Should return default
    
    def test_extract_int_value_none(self):
        """Test integer value extraction with None element"""
        result = self.loader._extract_int_value(None, 2)
        self.assertEqual(result, 2)  # Should return default
    
    def test_save_top_games_cache(self):
        """Test saving top games cache"""
        test_file = os.path.join(self.test_cache_dir, 'test_top500.json')
        
        with patch('data_loader.TOP500_FILE', test_file):
            self.loader.save_top_games_cache(MOCK_TOP_GAMES, 5, 5)
        
        self.assertTrue(os.path.exists(test_file))
        
        with open(test_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        self.assertIn('timestamp', cache_data)
        self.assertIn('games', cache_data)
        self.assertEqual(len(cache_data['games']), len(MOCK_TOP_GAMES))
    
    def test_save_game_details_cache(self):
        """Test saving game details cache"""
        test_file = os.path.join(self.test_cache_dir, 'test_details.json')
        
        with patch('data_loader.GAME_DETAILS_FILE', test_file):
            self.loader.save_game_details_cache(MOCK_GAME_DETAILS)
        
        self.assertTrue(os.path.exists(test_file))
        
        with open(test_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        self.assertIn('timestamp', cache_data)
        self.assertIn('details', cache_data)
        self.assertEqual(len(cache_data['details']), len(MOCK_GAME_DETAILS))
    
    def test_load_game_details_cache_success(self):
        """Test loading game details cache successfully"""
        test_file = os.path.join(self.test_cache_dir, 'test_details.json')
        
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(MOCK_GAME_DETAILS_CACHE, f)
        
        with patch('data_loader.GAME_DETAILS_FILE', test_file):
            details = self.loader.load_game_details_cache()
        
        self.assertEqual(len(details), len(MOCK_GAME_DETAILS))
        self.assertIn(174430, details)
    
    def test_load_game_details_cache_missing_file(self):
        """Test loading game details cache when file doesn't exist"""
        with patch('data_loader.GAME_DETAILS_FILE', '/nonexistent/file.json'):
            details = self.loader.load_game_details_cache()
        
        self.assertEqual(details, {})
    
    @patch('requests.get')
    def test_scrape_bgg_top_games_success(self, mock_get):
        """Test successful BGG top games scraping"""
        mock_get.return_value = get_mock_scraping_response()
        
        with patch.object(self.loader, 'save_top_games_cache'):
            top_games = self.loader.scrape_bgg_top_games()
        
        self.assertIsNotNone(top_games)
        self.assertGreater(len(top_games), 0)
    
    @patch('requests.get')
    def test_scrape_bgg_top_games_failure(self, mock_get):
        """Test BGG top games scraping failure fallback"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        with patch.object(self.loader, 'get_fallback_top_games') as mock_fallback:
            mock_fallback.return_value = MOCK_TOP_GAMES
            top_games = self.loader.scrape_bgg_top_games()
        
        mock_fallback.assert_called_once()
        self.assertEqual(top_games, MOCK_TOP_GAMES)
    
    def test_get_fallback_top_games(self):
        """Test fallback top games generation"""
        with patch.object(self.loader, 'save_top_games_cache'):
            fallback_games = self.loader.get_fallback_top_games()
        
        self.assertIsNotNone(fallback_games)
        self.assertGreaterEqual(len(fallback_games), 50)  # Should have at least 50 games
        self.assertEqual(fallback_games[0]['rank'], 1)
        self.assertEqual(fallback_games[0]['name'], 'Gloomhaven')
    
    def test_generate_additional_games(self):
        """Test generation of additional games"""
        additional_games = self.loader.generate_additional_games(100, 50)
        
        self.assertEqual(len(additional_games), 50)
        self.assertEqual(additional_games[0]['rank'], 101)
        self.assertEqual(additional_games[-1]['rank'], 150)


if __name__ == '__main__':
    unittest.main()