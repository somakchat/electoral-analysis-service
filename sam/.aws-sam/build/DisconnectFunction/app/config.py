"""
Configuration module for Political Strategy Maker.
Supports both local development and AWS production environments.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List
from pathlib import Path
import os

# Load .env file explicitly using python-dotenv
# This ensures environment variables are loaded before Settings class is instantiated
from dotenv import load_dotenv

# Try to find .env file in multiple locations
_env_paths = [
    Path(__file__).parent.parent / ".env",  # backend/.env
    Path(__file__).parent.parent.parent / ".env",  # project root/.env
    Path.cwd() / ".env",  # current working directory
]

for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
        print(f"[Config] Loaded environment from: {_env_path}")
        break


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Runtime Environment
    app_env: str = Field(default="local", description="local | aws")
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="openai | gemini")
    
    # OpenAI Settings
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o")  # Most advanced model for best accuracy
    openai_embed_model: str = Field(default="text-embedding-3-large")  # Larger embeddings for better retrieval
    openai_temperature: float = Field(default=0.1)
    
    # Gemini Settings (fallback)
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-pro")
    gemini_embed_model: str = Field(default="text-embedding-004")
    
    # Local Storage Paths
    data_dir: str = Field(default="./data")
    index_dir: str = Field(default="./index")
    
    # AWS Resources (Production)
    aws_region: str = Field(default="us-east-1")
    s3_bucket: Optional[str] = Field(default=None)
    s3_kg_bucket: Optional[str] = Field(default=None)  # Knowledge Graph storage
    ddb_table_sessions: Optional[str] = Field(default=None)
    ddb_table_memory: Optional[str] = Field(default=None)
    ddb_table_entities: Optional[str] = Field(default=None)
    
    # OpenSearch Configuration (unified for local and AWS)
    opensearch_endpoint: Optional[str] = Field(default=None, description="OpenSearch endpoint URL")
    opensearch_index: str = Field(default="political-strategy-maker", description="Main vector index")
    opensearch_memory_index: str = Field(default="political-memory", description="Memory index")
    opensearch_port: int = Field(default=443)
    opensearch_use_ssl: bool = Field(default=True)
    opensearch_verify_certs: bool = Field(default=True)
    
    # RAG Configuration
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=100)
    top_k_retrieval: int = Field(default=20)
    top_k_rerank: int = Field(default=10)
    
    # Agent Configuration
    max_agent_iterations: int = Field(default=15)
    max_rpm: int = Field(default=10)
    agent_verbose: bool = Field(default=True)
    
    # Memory Configuration
    short_term_max_items: int = Field(default=50)
    relevance_threshold: float = Field(default=0.7)
    entity_types: List[str] = Field(default=["constituency", "candidate", "party", "issue", "leader"])
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra fields in .env
    }

    @property
    def is_aws(self) -> bool:
        return self.app_env.lower() == "aws"
    
    @property
    def is_local(self) -> bool:
        return self.app_env.lower() == "local"


# Helper to fetch secrets from AWS Secrets Manager
def _get_secret_from_aws(secret_name: str) -> Optional[str]:
    """Fetch a secret from AWS Secrets Manager."""
    try:
        import boto3
        import json
        
        region = os.getenv("AWS_REGION", os.getenv("DEPLOYMENT_REGION", "us-east-1"))
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        
        # Secret can be string or JSON
        secret = response.get("SecretString")
        if secret:
            try:
                # Try to parse as JSON
                data = json.loads(secret)
                return data.get("api_key", data.get("OPENAI_API_KEY", secret))
            except json.JSONDecodeError:
                return secret
        return None
    except Exception as e:
        print(f"[Config] Warning: Could not fetch secret {secret_name}: {e}")
        return None


# Global settings instance
settings = Settings()

# On AWS, fetch secrets from Secrets Manager if not already set
if settings.is_aws and not settings.openai_api_key:
    print("[Config] AWS environment detected - fetching secrets from Secrets Manager...")
    secret_name = os.getenv("OPENAI_SECRET_NAME", "political-strategy/openai-api-key")
    secret_value = _get_secret_from_aws(secret_name)
    if secret_value:
        # Update the settings object with the secret
        object.__setattr__(settings, 'openai_api_key', secret_value)
        print(f"[Config] OpenAI API key loaded from Secrets Manager: ***{secret_value[-4:]}")

# Log configuration on startup (only in debug or when explicitly requested)
if settings.debug or os.getenv("SHOW_CONFIG", "").lower() == "true":
    print(f"[Config] APP_ENV: {settings.app_env}")
    print(f"[Config] LLM_PROVIDER: {settings.llm_provider}")
    print(f"[Config] OPENAI_MODEL: {settings.openai_model}")
    print(f"[Config] OPENAI_API_KEY: {'***' + settings.openai_api_key[-4:] if settings.openai_api_key else 'NOT SET'}")
    print(f"[Config] GEMINI_API_KEY: {'***' + settings.gemini_api_key[-4:] if settings.gemini_api_key else 'NOT SET'}")
