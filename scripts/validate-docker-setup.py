#!/usr/bin/env python3
"""
Docker setup validation script
Validates that all Docker files are properly configured
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - NOT FOUND")
        return False

def check_file_content(filepath, required_content, description):
    """Check if file contains required content"""
    if not os.path.exists(filepath):
        print(f"❌ {description}: {filepath} - FILE NOT FOUND")
        return False
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        for item in required_content:
            if item in content:
                print(f"✅ {description}: Contains '{item}'")
            else:
                print(f"❌ {description}: Missing '{item}'")
                return False
        return True
    except Exception as e:
        print(f"❌ {description}: Error reading file - {e}")
        return False

def validate_docker_setup():
    """Validate the complete Docker setup"""
    print("🐳 Validating Docker Setup for Vector DB API")
    print("=" * 50)
    
    all_good = True
    
    # Check required files
    required_files = [
        ("Dockerfile", "Dockerfile"),
        ("docker-compose.yml", "Docker Compose"),
        (".dockerignore", "Docker ignore file"),
        ("Makefile", "Makefile with build commands"),
        ("scripts/docker-scripts.sh", "Docker helper scripts"),
    ]
    
    print("\n📁 Checking Required Files:")
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Check Dockerfile content
    print("\n🔍 Checking Dockerfile Content:")
    dockerfile_checks = [
        ("FROM python:3.9-slim", "Base image"),
        ("WORKDIR /app", "Work directory"),
        ("COPY requirements.txt", "Requirements copy"),
        ("EXPOSE 8000", "Port exposure"),
        ("CMD [\"uvicorn\"", "Start command"),
    ]
    
    if not check_file_content("Dockerfile", [item[0] for item in dockerfile_checks], "Production Dockerfile"):
        all_good = False
    
    # Check docker-compose.yml content
    print("\n🔍 Checking Docker Compose Content:")
    compose_checks = [
        ("version: '3.8'", "Compose version"),
        ("vector-db-api:", "Service definition"),
        ("ports:", "Port mapping"),
        ("networks:", "Network configuration"),
    ]
    
    if not check_file_content("docker-compose.yml", [item[0] for item in compose_checks], "Docker Compose"):
        all_good = False
    
    
    # Check helper script permissions
    print("\n🔍 Checking Helper Script:")
    if os.path.exists("scripts/docker-scripts.sh"):
        if os.access("scripts/docker-scripts.sh", os.X_OK):
            print("✅ Docker helper script is executable")
        else:
            print("❌ Docker helper script is not executable")
            print("   Run: chmod +x scripts/docker-scripts.sh")
            all_good = False
    else:
        all_good = False
    
    # Check source structure
    print("\n🔍 Checking Source Structure:")
    source_checks = [
        ("src/vector_db_api/main.py", "Main application file"),
        ("src/vector_db_api/__init__.py", "Package init file"),
        ("requirements.txt", "Dependencies file"),
    ]
    
    for filepath, description in source_checks:
        if not check_file_exists(filepath, description):
            all_good = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 Docker setup validation PASSED!")
        print("\n📋 Next Steps:")
        print("1. Install Docker: See DOCKER-SETUP.md")
        print("2. Build image: make build")
        print("3. Start services: make run")
        print("4. Test API: curl http://localhost:8000/health")
        return True
    else:
        print("❌ Docker setup validation FAILED!")
        print("Please fix the issues above before proceeding.")
        return False

if __name__ == "__main__":
    success = validate_docker_setup()
    sys.exit(0 if success else 1)
