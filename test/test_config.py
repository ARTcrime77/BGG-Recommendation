"""
Tests for configuration module
"""

import unittest
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config


class TestConfig(unittest.TestCase):
    
    def test_cache_settings(self):
        """Test cache configuration settings"""
        self.assertEqual(config.CACHE_DIR, "bgg_cache")
        self.assertTrue(config.TOP500_FILE.endswith("top500_games.json"))
        self.assertTrue(config.GAME_DETAILS_FILE.endswith("game_details.json"))
        self.assertIsInstance(config.CACHE_MAX_AGE_DAYS, int)
        self.assertGreater(config.CACHE_MAX_AGE_DAYS, 0)
    
    def test_api_settings(self):
        """Test BGG API configuration settings"""
        self.assertTrue(config.BGG_API_BASE_URL.startswith("https://"))
        self.assertTrue(config.BGG_BROWSE_URL.startswith("https://"))
        self.assertIsInstance(config.API_DELAY, (int, float))
        self.assertGreater(config.API_DELAY, 0)
        self.assertIsInstance(config.BATCH_SIZE, int)
        self.assertGreater(config.BATCH_SIZE, 0)
    
    def test_scraping_settings(self):
        """Test web scraping configuration settings"""
        self.assertIsInstance(config.USER_AGENT, str)
        self.assertGreater(len(config.USER_AGENT), 10)
        self.assertIsInstance(config.SCRAPING_DELAY, (int, float))
        self.assertGreater(config.SCRAPING_DELAY, 0)
        self.assertIsInstance(config.TARGET_TOP_GAMES, int)
        self.assertGreater(config.TARGET_TOP_GAMES, 100)
        self.assertIsInstance(config.MAX_SCRAPING_PAGES, int)
        self.assertGreater(config.MAX_SCRAPING_PAGES, 0)
    
    def test_ml_settings(self):
        """Test ML configuration settings"""
        self.assertIsInstance(config.MIN_FEATURE_FREQUENCY, int)
        self.assertGreater(config.MIN_FEATURE_FREQUENCY, 0)
        self.assertIsInstance(config.MAX_NEIGHBORS, int)
        self.assertGreater(config.MAX_NEIGHBORS, 0)
        self.assertIsInstance(config.SIMILARITY_METRIC, str)
        self.assertIn(config.SIMILARITY_METRIC, ['cosine', 'euclidean', 'manhattan'])
    
    def test_weight_settings(self):
        """Test weighting configuration settings"""
        self.assertIsInstance(config.RATING_WEIGHT_MULTIPLIER, (int, float))
        self.assertGreater(config.RATING_WEIGHT_MULTIPLIER, 0)
        self.assertIsInstance(config.PLAY_COUNT_LOG_BASE, (int, float))
        self.assertGreaterEqual(config.PLAY_COUNT_LOG_BASE, 0)
    
    def test_output_settings(self):
        """Test output configuration settings"""
        self.assertIsInstance(config.DEFAULT_NUM_RECOMMENDATIONS, int)
        self.assertGreater(config.DEFAULT_NUM_RECOMMENDATIONS, 0)
        self.assertIsInstance(config.DEBUG_SHOW_SIMILARITY_DETAILS, bool)
        self.assertIsInstance(config.SHOW_PROGRESS_EVERY, int)
        self.assertGreater(config.SHOW_PROGRESS_EVERY, 0)
    
    def test_file_paths(self):
        """Test that file path configurations are properly constructed"""
        # Test that cache file paths are within cache directory
        self.assertTrue(config.TOP500_FILE.startswith(config.CACHE_DIR))
        self.assertTrue(config.GAME_DETAILS_FILE.startswith(config.CACHE_DIR))
        
        # Test file extensions
        self.assertTrue(config.TOP500_FILE.endswith('.json'))
        self.assertTrue(config.GAME_DETAILS_FILE.endswith('.json'))
    
    def test_reasonable_values(self):
        """Test that configuration values are within reasonable ranges"""
        # Cache age should be reasonable (not too short or too long)
        self.assertGreaterEqual(config.CACHE_MAX_AGE_DAYS, 1)
        self.assertLessEqual(config.CACHE_MAX_AGE_DAYS, 30)
        
        # API delay should be respectful
        self.assertGreaterEqual(config.API_DELAY, 0.5)
        self.assertLessEqual(config.API_DELAY, 5.0)
        
        # Batch size should be reasonable
        self.assertGreaterEqual(config.BATCH_SIZE, 1)
        self.assertLessEqual(config.BATCH_SIZE, 100)
        
        # Target games should be reasonable
        self.assertGreaterEqual(config.TARGET_TOP_GAMES, 100)
        self.assertLessEqual(config.TARGET_TOP_GAMES, 5000)
        
        # ML parameters should be reasonable
        self.assertGreaterEqual(config.MIN_FEATURE_FREQUENCY, 1)
        self.assertLessEqual(config.MIN_FEATURE_FREQUENCY, 10)
        
        self.assertGreaterEqual(config.MAX_NEIGHBORS, 5)
        self.assertLessEqual(config.MAX_NEIGHBORS, 100)
        
        # Recommendations should be reasonable
        self.assertGreaterEqual(config.DEFAULT_NUM_RECOMMENDATIONS, 1)
        self.assertLessEqual(config.DEFAULT_NUM_RECOMMENDATIONS, 50)


if __name__ == '__main__':
    unittest.main()