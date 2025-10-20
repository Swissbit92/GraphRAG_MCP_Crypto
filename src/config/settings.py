# src/config/settings.py
# Configuration management for GraphRAG MCP using pydantic-settings.
# Reads from environment variables and .env file.
# Provides unified access to RAG, KG, and Pipeline settings.
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List

# Prefer pydantic-settings (v2) if available; fall back gracefully
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
except Exception:  # pragma: no cover
    from pydantic import BaseModel as BaseSettings  # type: ignore
    def SettingsConfigDict(**kwargs):  # type: ignore
        return {}

from pydantic import Field, validator
try:
    # Load .env if present (no-op if missing)
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------- Subsettings ----------
class OllamaSettings(BaseSettings):
    model: str = Field(default=os.getenv("OLLAMA_MODEL", "llama3.1:latest"))
    model_fallback: Optional[str] = Field(default=os.getenv("OLLAMA_MODEL_FALLBACK"))
    base: str = Field(default=os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434"))
    embed_model: str = Field(default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")


class ChromaSettings(BaseSettings):
    dir: str = Field(default=os.getenv("CHROMA_DIR", ".chroma"))
    collection: str = Field(default=os.getenv("CHROMA_COLLECTION", "whitepapers"))
    expand_per_entity: bool = Field(default=(os.getenv("RAG_EXPAND_PER_ENTITY", "true").lower() in ("1", "true", "yes")))
    disable_embed: bool = Field(default=(os.getenv("RAG_DISABLE_EMBED", "false").lower() in ("1", "true", "yes")))
    # Optional alternative embeddings
    use_ollama_embed: bool = Field(default=(os.getenv("USE_OLLAMA_EMBED", "true").lower() in ("1", "true", "yes")))
    sentence_transformer_model: Optional[str] = Field(default=os.getenv("SENTENCE_TRANSFORMER_MODEL"))

    model_config = SettingsConfigDict(extra="ignore")


class GraphDBSettings(BaseSettings):
    url: str = Field(default=os.getenv("GRAPHDB_URL", "http://localhost:7200"))
    repository: str = Field(default=os.getenv("GRAPHDB_REPOSITORY", "mcp_kg"))
    username: Optional[str] = Field(default=os.getenv("GRAPHDB_USERNAME"))
    password: Optional[str] = Field(default=os.getenv("GRAPHDB_PASSWORD"))
    push: bool = Field(default=(os.getenv("GRAPHDB_PUSH", "").lower() in ("1", "true", "yes")))
    batch_size: int = Field(default=int(os.getenv("GRAPHDB_BATCH_SIZE", "100")))

    model_config = SettingsConfigDict(env_prefix="GRAPHDB_", extra="ignore")


class KGSettings(BaseSettings):
    entity_only: bool = Field(default=(os.getenv("KG_ENTITY_ONLY", "true").lower() in ("1", "true", "yes")))
    ontology_path: Optional[str] = Field(default=os.getenv("KG_ONTOLOGY_PATH"))
    shapes_path: Optional[str] = Field(default=os.getenv("KG_SHAPES_PATH"))
    labels_dir: str = Field(default=os.getenv("KG_LABELS_DIR", "outputs/run_simple/labels"))
    docs_dir: str = Field(default=os.getenv("KG_DOCS_DIR", "outputs/run_simple/docs"))
    log_level: str = Field(default=os.getenv("KG_MCP_LOG_LEVEL", "INFO"))

    model_config = SettingsConfigDict(extra="ignore")


class RAGSettings(BaseSettings):
    outputs_dir: str = Field(default=os.getenv("RAG_OUTPUTS_DIR", "outputs/run_simple"))
    log_level: str = Field(default=os.getenv("RAG_MCP_LOG_LEVEL", "INFO"))
    qa_llm_model: Optional[str] = Field(default=os.getenv("QA_LLM_MODEL"))
    qa_mode: str = Field(default=os.getenv("QA_LLM_MODE", ""))  # "mock" to force offline
    qa_kg_enrich: bool = Field(default=(os.getenv("QA_KG_ENRICH", "true").lower() in ("1", "true", "yes")))

    model_config = SettingsConfigDict(extra="ignore")


class PipelineSettings(BaseSettings):
    rag_build: bool = Field(default=(os.getenv("RAG_BUILD", "true").lower() in ("1", "true", "yes")))
    cache_dir: str = Field(default=os.getenv("WP_EXPLORE_CACHE_DIR", "outputs/run_simple/.cache/labels"))
    embed_model_name: str = Field(default=os.getenv("EMBED_MODEL_NAME", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")))

    model_config = SettingsConfigDict(extra="ignore")


# ---------- Unified Settings ----------
class Settings(BaseSettings):
    """Unified configuration for GraphRAG MCP (RAG + KG + Pipeline)."""

    # Subtrees
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    graphdb: GraphDBSettings = Field(default_factory=GraphDBSettings)
    kg: KGSettings = Field(default_factory=KGSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    # General toggles
    python_unbuffered: bool = Field(default=(os.getenv("PYTHONUNBUFFERED", "1") == "1"))

    # Convenience mirrors (kept for easy drop-in into existing code if needed)
    @property
    def GRAPHDB_URL(self) -> str:
        return self.graphdb.url.rstrip("/")

    @property
    def GRAPHDB_REPOSITORY(self) -> str:
        return self.graphdb.repository

    @property
    def CHROMA_DIR(self) -> str:
        return self.chroma.dir

    @property
    def CHROMA_COLLECTION(self) -> str:
        return self.chroma.collection

    @property
    def OLLAMA_BASE(self) -> str:
        return self.ollama.base.rstrip("/")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def as_dict(self) -> dict:
        """For logging/debugging without leaking secrets (passwords masked)."""
        data = {
            "ollama": {
                "model": self.ollama.model,
                "model_fallback": self.ollama.model_fallback,
                "base": self.ollama.base,
                "embed_model": self.ollama.embed_model,
            },
            "chroma": {
                "dir": self.chroma.dir,
                "collection": self.chroma.collection,
                "expand_per_entity": self.chroma.expand_per_entity,
                "disable_embed": self.chroma.disable_embed,
                "use_ollama_embed": self.chroma.use_ollama_embed,
                "sentence_transformer_model": self.chroma.sentence_transformer_model,
            },
            "graphdb": {
                "url": self.graphdb.url,
                "repository": self.graphdb.repository,
                "username": bool(self.graphdb.username),
                "password": bool(self.graphdb.password),  # masked
                "push": self.graphdb.push,
                "batch_size": self.graphdb.batch_size,
            },
            "kg": {
                "entity_only": self.kg.entity_only,
                "ontology_path": self.kg.ontology_path,
                "shapes_path": self.kg.shapes_path,
                "labels_dir": self.kg.labels_dir,
                "docs_dir": self.kg.docs_dir,
                "log_level": self.kg.log_level,
            },
            "rag": {
                "outputs_dir": self.rag.outputs_dir,
                "log_level": self.rag.log_level,
                "qa_llm_model": self.rag.qa_llm_model,
                "qa_mode": self.rag.qa_mode,
                "qa_kg_enrich": self.rag.qa_kg_enrich,
            },
            "pipeline": {
                "rag_build": self.pipeline.rag_build,
                "cache_dir": self.pipeline.cache_dir,
                "embed_model_name": self.pipeline.embed_model_name,
            },
            "python_unbuffered": self.python_unbuffered,
        }
        return data


# Singleton instance
settings = Settings()


# ---------- Optional: small helpers for servers ----------
def apply_rag_logging():
    lvl = (settings.rag.log_level or "INFO").upper()
    os.environ["RAG_MCP_LOG_LEVEL"] = lvl


def apply_kg_logging():
    lvl = (settings.kg.log_level or "INFO").upper()
    os.environ["KG_MCP_LOG_LEVEL"] = lvl
