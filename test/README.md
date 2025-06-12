# BGG ML Recommendation System - Test Suite

This directory contains comprehensive unit and integration tests for the BGG ML Recommendation System.

## Test Structure

```
test/
├── __init__.py                 # Test package initialization
├── test_fixtures.py           # Mock data and test fixtures
├── test_config.py              # Configuration tests
├── test_data_loader.py         # BGGDataLoader unit tests
├── test_ml_engine.py           # BGGMLEngine unit tests  
├── test_main.py                # BGGRecommender unit tests
├── test_integration.py         # Integration tests
├── run_tests.py                # Test runner script
└── README.md                   # This file
```

## Running Tests

### Using the Test Runner

```bash
# Run all tests
python test/run_tests.py

# Run only unit tests
python test/run_tests.py --unit

# Run only integration tests  
python test/run_tests.py --integration

# Run specific test module
python test/run_tests.py --module test_data_loader

# Run with coverage report (requires coverage package)
python test/run_tests.py --coverage
```

### Using unittest directly

```bash
# Run all tests
python -m unittest discover test

# Run specific test file
python -m unittest test.test_data_loader

# Run specific test class
python -m unittest test.test_data_loader.TestBGGDataLoader

# Run specific test method
python -m unittest test.test_data_loader.TestBGGDataLoader.test_fetch_user_collection_success
```

### Using pytest (if installed)

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest test/test_data_loader.py

# Run tests with specific marker
pytest -m unit
```

## Test Categories

### Unit Tests
- **test_config.py**: Configuration validation tests
- **test_data_loader.py**: BGGDataLoader class tests
- **test_ml_engine.py**: BGGMLEngine class tests  
- **test_main.py**: BGGRecommender class tests

### Integration Tests
- **test_integration.py**: End-to-end system tests

### Test Fixtures
- **test_fixtures.py**: Mock data, API responses, and test utilities

## Test Coverage

The test suite covers:

### BGGDataLoader
- Cache management (creation, validation, loading)
- BGG API interactions (collection, plays, game details)
- Web scraping functionality
- Error handling and fallbacks
- Duplicate detection and removal

### BGGMLEngine  
- Feature matrix creation and normalization
- Model training and validation
- User preference vector generation
- Recommendation generation and filtering
- Categorical feature encoding

### BGGRecommender
- Complete workflow orchestration
- Data loading pipeline
- ML model training
- Recommendation display
- Error handling

### Integration Tests
- Full system end-to-end flow
- Component interaction validation
- Cache system integration
- Error propagation
- Data quality validation

## Mock Data

Test fixtures include:
- Sample BGG user collections
- Mock game details and metadata
- Simulated API responses (XML)
- Test game datasets for ML training

## Dependencies

Required for testing:
```bash
# Core testing (included in Python)
unittest
unittest.mock

# Optional for enhanced testing
pytest              # Alternative test runner
pytest-cov         # Coverage reporting
coverage           # Coverage analysis
```

## Best Practices

1. **Isolation**: Each test is independent and doesn't rely on external services
2. **Mocking**: External APIs and file operations are mocked
3. **Comprehensive**: Tests cover both success and failure scenarios  
4. **Fast**: Unit tests run quickly without network calls
5. **Realistic**: Test data mirrors real BGG API responses