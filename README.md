# 🧠 GraphRAG MCP  
> **Entity-centric Retrieval-Augmented Generation for Crypto Whitepapers**  
> _Local-first • Private • FastMCP-ready_

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" />
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-Local%20LLMs-000000?logo=ollama&logoColor=white" />
  <img alt="ChromaDB" src="https://img.shields.io/badge/Chroma-Vector%20Store-241F31" />
  <img alt="GraphDB" src="https://img.shields.io/badge/GraphDB-Ontotext-FF5A1F" />
  <img alt="FastMCP" src="https://img.shields.io/badge/FastMCP-2.x-4F46E5" />
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-Local%20Orchestration-00BFFF" />
  <img alt="License" src="https://img.shields.io/badge/Privacy-Local%20Only-16a34a" />
</p>

---

## 1) ✨ Overview

**GraphRAG MCP** is a modular, **local-first** system that turns crypto whitepapers into an **entity-centric Knowledge Graph** and a **vector-searchable corpus**, then answers questions with **RAG + optional KG enrichment + LLM synthesis** — all via standardized **FastMCP** tools.

### Why this project?
- 🛡️ **Privacy by default:** runs entirely on your machine (Ollama, Chroma, GraphDB).
- ⚡ **Fast & focused:** entity-filtered retrieval narrows context to the right tokens/protocols.
- 🧩 **Composable:** exposes `rag.*` and `kg.*` tools so an **MCP Coordinator** or **Streamlit** app can orchestrate multi-tool workflows.
- 🧠 **Explainable answers:** returns **citations with doc/chunk/entity IDs** for every response.

---

### 🧱 Core Building Blocks

| Layer | What it does | Tech |
|---|---|---|
| **Knowledge Graph (KG)** | Stores canonical **entities** (Token/Protocol/Component/Org) for clean, cross-doc grounding | **GraphDB (Ontotext)**, SHACL-friendly ontology (`mcp-core.ttl`, `mcp-crypto.ttl`) |
| **Vector RAG** | Persists whitepaper **chunks + embeddings** and supports **entity-filtered** semantic search | **ChromaDB** + **Ollama** embeddings (`nomic-embed-text`) |
| **LLM Labeling & QA** | Labels chunks during ingest; later synthesizes concise answers with citations | **Ollama** models (e.g., `llama3.1`, `qwen2.5`) |
| **FastMCP Servers** | Expose all capabilities as standard tools for coordination | `rag.*` (search/embed/reindex/qa) & `kg.*` (sparql/validate/push) |

---

### 🔁 End-to-End Flow

