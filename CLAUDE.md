# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a German-language BoardGameGeek (BGG) Machine Learning recommendation system that generates personalized board game recommendations using k-Nearest Neighbors algorithm.

## Commands

### Running the Application
```bash
# Activate virtual environment
source BGG_Reco/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the main application
python src/main.py
```

### Development Setup
```bash
# Create virtual environment
python3 -m venv BGG_Reco

# Install dependencies
pip install numpy pandas scikit-learn requests beautifulsoup4
```

## Architecture

### Core Components

**Data Flow:**
1. `BGGDataLoader` scrapes/fetches BGG Top games and user data
2. `BGGMLEngine` processes game features and trains k-NN model
3. `BGGRecommender` orchestrates the entire recommendation pipeline

**Key Classes:**
- `BGGRecommender` (main.py) - Main orchestrator class that coordinates data loading, ML training, and recommendation generation
- `BGGDataLoader` (data_loader.py) - Handles BGG API calls, web scraping, and caching
- `BGGMLEngine` (ml_engine.py) - Feature engineering and machine learning pipeline

### Feature Engineering Strategy

The ML engine creates a comprehensive feature matrix with:
- **Numerical features (7)**: rating, complexity, player counts, log-transformed playing time, game age
- **Categorical features (one-hot encoded)**: categories, mechanics, designers, artists, publishers
- **Frequency filtering**: Only includes designers/artists/publishers appearing in ≥2 games

### Caching System

**Cache Files:**
- `bgg_cache/top500_games.json` - Top games list with metadata
- `bgg_cache/game_details.json` - Detailed game information

**Cache Logic:**
- Checks cache age (default 7 days) before fetching new data
- Interactive prompts for cache updates
- Automatic fallback to hardcoded game list if scraping fails

### Configuration

All parameters are centralized in `config.py`:
- API delays and batch sizes
- ML parameters (k-NN neighbors, similarity metric)
- Feature frequency thresholds
- Caching settings

### User Interaction Flow

1. Load user's BGG collection and play statistics
2. Scrape/load BGG Top games (target: 1000 to get 500 unique)
3. Fetch detailed game information via BGG API
4. Create feature matrix and train k-NN model
5. Generate user preference vector from collection/ratings
6. Find similar games and filter out owned games
7. Present ranked recommendations

### BGG API Integration

- Uses BGG XML API v2 for collection and game details
- Implements rate limiting (1.5s delays) and batch processing
- Handles API errors and provides fallback data
- Web scraping for Top games list with duplicate detection

### Important Notes

- The system targets German users (German language in UI)
- Username is hardcoded in `main.py:236` - change "Artcrime77" to desired BGG username
- Requires public BGG collection and ratings for recommendations
- Automatically handles duplicate games across multiple data sources