#!/usr/bin/env python3
"""
Test runner for BGG ML Recommendation System
"""

import unittest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all tests in the test directory"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_specific_test_module(module_name):
    """Run tests from a specific module"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'test.{module_name}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_unit_tests():
    """Run only unit tests (excluding integration tests)"""
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    
    # Load specific unit test modules
    test_modules = [
        'test_data_loader',
        'test_ml_engine', 
        'test_main'
    ]
    
    suite = unittest.TestSuite()
    for module in test_modules:
        module_suite = loader.loadTestsFromName(f'test.{module}')
        suite.addTest(module_suite)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def run_integration_tests():
    """Run only integration tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName('test.test_integration')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

def main():
    """Main test runner function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BGG ML Recommendation System Test Runner')
    parser.add_argument('--module', '-m', help='Run tests from specific module (e.g., test_data_loader)')
    parser.add_argument('--unit', '-u', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', '-i', action='store_true', help='Run only integration tests')
    parser.add_argument('--coverage', '-c', action='store_true', help='Run with coverage report')
    
    args = parser.parse_args()
    
    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage()
            cov.start()
            
            success = run_tests_based_on_args(args)
            
            cov.stop()
            cov.save()
            
            print("\n" + "="*50)
            print("COVERAGE REPORT")
            print("="*50)
            cov.report()
            
            return success
        except ImportError:
            print("Coverage module not installed. Install with: pip install coverage")
            print("Running tests without coverage...")
            return run_tests_based_on_args(args)
    else:
        return run_tests_based_on_args(args)

def run_tests_based_on_args(args):
    """Run tests based on command line arguments"""
    if args.module:
        print(f"Running tests from module: {args.module}")
        return run_specific_test_module(args.module)
    elif args.unit:
        print("Running unit tests...")
        return run_unit_tests()
    elif args.integration:
        print("Running integration tests...")
        return run_integration_tests()
    else:
        print("Running all tests...")
        return run_all_tests()

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)