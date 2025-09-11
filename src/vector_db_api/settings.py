"""
Configuration settings for the Vector DB REST API.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = Field(default="Vector DB REST API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    
    # Vector Database Settings
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8001, description="ChromaDB port")
    chroma_persist_directory: str = Field(default="./chroma_db", description="ChromaDB persistence directory")
    
    # Embedding Settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Sentence transformer model for embeddings")
    embedding_dimension: int = Field(default=384, description="Dimension of embeddings")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

