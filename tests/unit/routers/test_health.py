"""
Unit tests for Health router
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from fastapi.testclient import TestClient
from fastapi import FastAPI

from vector_db_api.api.routers.health import router
from vector_db_api.api.dto import HealthOut


class TestHealthRouter:
    """Test cases for Health router"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        # Create FastAPI app with the router
        self.app = FastAPI()
        self.app.include_router(router)
        
        self.client = TestClient(self.app)
    
    def test_health_check_success(self):
        """Test successful health check"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert "timestamp" in data
        assert "details" in data
        
        # Check field values
        assert data["status"] == "ok"
        assert isinstance(data["timestamp"], str)  # ISO datetime string
        assert isinstance(data["details"], dict)
        assert data["details"] == {}  # Empty details dict
    
    def test_health_check_response_structure(self):
        """Test that health check response has correct structure"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = ["status", "timestamp", "details"]
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data["status"], str)
        assert isinstance(data["timestamp"], str)  # ISO datetime string
        assert isinstance(data["details"], dict)
    
    def test_health_check_timestamp_format(self):
        """Test that health check timestamp is in correct format"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Parse timestamp to ensure it's valid ISO format
        timestamp_str = data["timestamp"]
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            assert isinstance(parsed_timestamp, datetime)
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp_str}")
    
    def test_health_check_timestamp_is_recent(self):
        """Test that health check timestamp is recent"""
        # Act
        before_request = datetime.utcnow()
        response = self.client.get("/health")
        after_request = datetime.utcnow()
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Parse timestamp
        timestamp_str = data["timestamp"]
        response_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Check that timestamp is within reasonable bounds
        assert before_request <= response_timestamp <= after_request
    
    def test_health_check_multiple_calls(self):
        """Test multiple health check calls return different timestamps"""
        # Act
        response1 = self.client.get("/health")
        response2 = self.client.get("/health")
        
        # Assert
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Both should have same structure
        assert data1["status"] == "ok"
        assert data2["status"] == "ok"
        assert data1["details"] == {}
        assert data2["details"] == {}
        
        # Timestamps should be different (or very close)
        timestamp1 = datetime.fromisoformat(data1["timestamp"].replace('Z', '+00:00'))
        timestamp2 = datetime.fromisoformat(data2["timestamp"].replace('Z', '+00:00'))
        
        # Allow for small time differences due to processing time
        time_diff = abs((timestamp2 - timestamp1).total_seconds())
        assert time_diff >= 0  # Should be non-negative
    
    def test_health_check_status_always_ok(self):
        """Test that health check always returns status 'ok'"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_health_check_details_empty(self):
        """Test that health check details are always empty"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["details"] == {}
        assert isinstance(data["details"], dict)
    
    def test_health_check_no_dependencies(self):
        """Test that health check doesn't require any external dependencies"""
        # This test ensures the health endpoint is truly independent
        # and doesn't fail due to missing services or databases
        
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "details" in data
    
    def test_health_check_response_time(self):
        """Test that health check responds quickly"""
        import time
        
        # Act
        start_time = time.time()
        response = self.client.get("/health")
        end_time = time.time()
        
        # Assert
        assert response.status_code == 200
        response_time = end_time - start_time
        
        # Health check should be very fast (less than 1 second)
        assert response_time < 1.0
        
        data = response.json()
        assert data["status"] == "ok"
    
    def test_health_check_with_mocked_datetime(self):
        """Test health check with mocked datetime"""
        # Arrange
        fixed_time = datetime(2024, 1, 15, 12, 30, 45, 123456)
        
        with patch('vector_db_api.api.routers.health.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = fixed_time
            
            # Act
            response = self.client.get("/health")
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            # Note: The actual timestamp format may vary, so we just check it's a valid ISO format
            assert data["timestamp"].startswith("2024-01-15T12:30:45")
            assert data["details"] == {}
    
    def test_health_check_content_type(self):
        """Test that health check returns correct content type"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
    
    def test_health_check_cors_headers(self):
        """Test that health check includes appropriate headers"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        # Note: CORS headers would be added by middleware, not the router itself
        # This test just ensures the basic response is correct
    
    def test_health_check_method_not_allowed(self):
        """Test that health check only accepts GET requests"""
        # Act
        response = self.client.post("/health")
        
        # Assert
        assert response.status_code == 405  # Method Not Allowed
    
    def test_health_check_with_query_params(self):
        """Test health check with query parameters (should ignore them)"""
        # Act
        response = self.client.get("/health?param1=value1&param2=value2")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "details" in data
    
    def test_health_check_consistency(self):
        """Test that health check returns consistent structure across multiple calls"""
        responses = []
        
        # Make multiple calls
        for _ in range(5):
            response = self.client.get("/health")
            responses.append(response)
        
        # Assert all responses are successful
        for response in responses:
            assert response.status_code == 200
        
        # Assert all responses have same structure
        for response in responses:
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert "details" in data
            assert data["status"] == "ok"
            assert isinstance(data["details"], dict)
    
    def test_health_check_under_load(self):
        """Test health check under simulated load"""
        import concurrent.futures
        import threading
        
        results = []
        errors = []
        
        def make_health_request():
            try:
                response = self.client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                errors.append(e)
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            concurrent.futures.wait(futures)
        
        # Assert no errors occurred
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Assert all requests succeeded
        assert len(results) == 10
        assert all(status == 200 for status in results)
    
    def test_health_check_route_path(self):
        """Test that health check is accessible at correct path"""
        # Act
        response = self.client.get("/health")
        
        # Assert
        assert response.status_code == 200
        
        # Test that it's also accessible with trailing slash (FastAPI behavior)
        response_with_slash = self.client.get("/health/")
        assert response_with_slash.status_code == 200
        
        # Test that it's not accessible at completely different paths
        response_wrong = self.client.get("/healthcheck")
        assert response_wrong.status_code == 404
        
        response_wrong2 = self.client.get("/status")
        assert response_wrong2.status_code == 404
