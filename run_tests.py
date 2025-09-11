#!/usr/bin/env python3
"""
Test runner script for the Vector DB API project
"""

import sys
import os
import subprocess
from pathlib import Path

def run_tests(test_path=None, verbose=False, coverage=False, markers=None):
    """Run tests with pytest"""
    
    # Add src to Python path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Build pytest command
    cmd = ['python3', '-m', 'pytest']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src/vector_db_api', '--cov-report=html', '--cov-report=term'])
    
    # Add test path if specified
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append('tests/')
    
    # Add marker filters if specified
    if markers:
        cmd.extend(['-m', ' or '.join(markers)])
    
    # Add additional pytest options
    cmd.extend([
        '--tb=short',  # Shorter traceback format
        '--strict-markers',  # Strict marker checking
        '--disable-warnings'  # Disable warnings for cleaner output
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=True)
        print("=" * 60)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("=" * 60)
        print(f"❌ Tests failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ pytest not found. Please install pytest: pip install pytest")
        return False

def main():
    """Main function to handle command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tests for Vector DB API')
    parser.add_argument('--path', '-p', help='Specific test path to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--coverage', '-c', action='store_true', help='Run with coverage')
    parser.add_argument('--unit', action='store_true', help='Run only unit tests')
    parser.add_argument('--integration', action='store_true', help='Run only integration tests')
    parser.add_argument('--cohere', action='store_true', help='Run Cohere integration tests (requires COHERE_API_KEY)')
    parser.add_argument('--external', action='store_true', help='Run tests that require external services')
    
    args = parser.parse_args()
    
    # Determine test path and markers
    test_path = args.path
    markers = []
    
    if args.unit:
        test_path = 'tests/unit/'
        markers.append('unit')
    elif args.integration:
        test_path = 'tests/integration/'
        markers.append('integration')
    elif args.cohere:
        test_path = 'tests/integration/test_cohere_embeddings.py'
        markers.append('cohere')
    elif args.external:
        markers.append('external')
    
    # Run tests
    success = run_tests(
        test_path=test_path,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=markers
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
