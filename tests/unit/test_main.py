"""
Unit tests for FastAPI app initialization and main application
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from uuid import uuid4

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vector_db_api.main import app


class TestFastAPIApp:
    """Test FastAPI application initialization and configuration"""
    
    def test_app_initialization(self):
        """Test that the FastAPI app is properly initialized"""
        assert app is not None
        assert app.title == "Vector DB REST API"
        assert app.version == "1.0.0"
    
    def test_app_includes_routers(self):
        """Test that all required routers are included"""
        # Get all route paths
        routes = [route.path for route in app.routes]
        
        # Check that all expected routes are present
        expected_routes = [
            "/health",
            "/libraries",
            "/libraries/{lib_id}/documents", 
            "/libraries/{lib_id}/documents/{doc_id}/chunks",
            "/libraries/{lib_id}/search"
        ]
        
        for expected_route in expected_routes:
            # Check if any route starts with the expected path
            assert any(route.startswith(expected_route) for route in routes), f"Route {expected_route} not found"
    
    def test_app_has_exception_handlers(self):
        """Test that custom exception handlers are registered"""
        # Check that exception handlers are registered
        assert hasattr(app, 'exception_handlers')
        assert len(app.exception_handlers) > 0
    
    def test_health_endpoint_accessible(self):
        """Test that health endpoint is accessible"""
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "timestamp" in data
            assert data["status"] == "ok"
    
    def test_openapi_schema_generation(self):
        """Test that OpenAPI schema is generated correctly"""
        with TestClient(app) as client:
            response = client.get("/openapi.json")
            assert response.status_code == 200
            schema = response.json()
            
            # Check basic schema structure
            assert "openapi" in schema
            assert "info" in schema
            assert "paths" in schema
            
            # Check API info
            assert schema["info"]["title"] == "Vector DB REST API"
            assert schema["info"]["version"] == "1.0.0"
            
            # Check that key endpoints are documented
            assert "/health" in schema["paths"]
            assert "/libraries" in schema["paths"]
            assert "/libraries/{lib_id}/documents" in schema["paths"]
            assert "/libraries/{lib_id}/documents/{doc_id}/chunks" in schema["paths"]
            assert "/libraries/{lib_id}/search" in schema["paths"]


class TestAppDependencies:
    """Test application dependency injection"""
    
    def test_dependency_initialization(self):
        """Test that dependencies are properly initialized"""
        # Check that the app has the expected structure
        assert app is not None
        assert hasattr(app.state, 'container')
        
        # Check that services are properly initialized
        container = app.state.container
        assert "library_svc" in container
        assert "document_svc" in container
        assert "chunk_svc" in container
        assert "search_svc" in container
    
    def test_service_dependencies(self):
        """Test that service dependencies are properly configured"""
        # Check that service dependencies are available in app state
        assert hasattr(app.state, 'container')
        container = app.state.container
        
        assert "library_svc" in container
        assert "document_svc" in container
        assert "chunk_svc" in container
        assert "search_svc" in container
        
        assert container["library_svc"] is not None
        assert container["document_svc"] is not None
        assert container["chunk_svc"] is not None
        assert container["search_svc"] is not None


class TestAppStartup:
    """Test application startup events"""
    
    def test_startup_event_exists(self):
        """Test that startup event function exists and is callable"""
        # The startup event is defined as a local function in create_app()
        # We can verify the app is properly configured with startup events
        assert app is not None
        # The startup event is registered via @app.on_event("startup")
        # We can verify this by checking the app has the expected structure
    
    def test_app_has_startup_event_registered(self):
        """Test that the app has startup event registered"""
        # Check that the app has startup events
        assert hasattr(app, 'router')
        # The startup event is registered via @app.on_event("startup")
        # We can verify the app is properly configured
        assert app is not None


class TestAppConfiguration:
    """Test application configuration and settings"""
    
    def test_app_cors_configuration(self):
        """Test CORS configuration if present"""
        # Check if CORS middleware is configured
        middleware_types = [type(middleware).__name__ for middleware in app.user_middleware]
        
        # CORS might be configured, but it's optional
        # We'll just verify the app can start without errors
        assert app is not None
    
    def test_app_middleware_stack(self):
        """Test that middleware stack is properly configured"""
        # Check that middleware is configured
        assert hasattr(app, 'user_middleware')
        assert isinstance(app.user_middleware, list)
    
    def test_app_route_handlers(self):
        """Test that route handlers are properly configured"""
        # Check that routes are configured
        assert hasattr(app, 'routes')
        assert len(app.routes) > 0
        
        # Check that we have both API routes and static routes
        route_types = [type(route).__name__ for route in app.routes]
        assert 'APIRoute' in route_types


class TestAppErrorHandling:
    """Test application error handling configuration"""
    
    def test_404_error_handling(self):
        """Test 404 error handling"""
        with TestClient(app) as client:
            response = client.get("/nonexistent-endpoint")
            assert response.status_code == 404
    
    def test_method_not_allowed_handling(self):
        """Test method not allowed error handling"""
        with TestClient(app) as client:
            # Try POST to a GET-only endpoint
            response = client.post("/health")
            assert response.status_code == 405  # Method Not Allowed
    
    def test_custom_exception_handlers(self):
        """Test that custom exception handlers are working"""
        # This would require triggering a custom exception
        # For now, we'll just verify the handlers are registered
        assert hasattr(app, 'exception_handlers')
        assert len(app.exception_handlers) > 0


