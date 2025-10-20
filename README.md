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

### 🔁 End-to-End Flow

**Core data flow:**
PDF → semantic_splitter → llm_chunk_tagger → postprocess
→ Chroma (vector store) + GraphDB (entity KG)
→ FastMCP: rag_server + kg_server

**Typical usage:**
1. Ingest and label whitepapers → build embeddings and insert entities.  
2. Ask questions via `rag.qa` (semantic + entity-filtered retrieval), optionally enrich with KG labels/aliases.  
3. Get concise LLM answers with inline citations to source chunks.

---

## 2️⃣ Features

### 🧩 Knowledge Graph (KG)
- **Entity-only architecture** using RDF/OWL ontologies (`mcp-core.ttl`, `mcp-crypto.ttl`).
- Built on **Ontotext GraphDB 11+** with SHACL validation and SPARQL/GraphQL endpoints.
- Stores canonical entities such as tokens, protocols, components, and organizations.
- Enables KG enrichment for RAG answers via aliases, labels, and relationships.

### 🔍 Vector Retrieval (RAG)
- **ChromaDB** acts as the persistent vector store for chunk embeddings.
- Embeddings generated using **Ollama’s** `nomic-embed-text` model.
- Supports **semantic** and **entity-filtered** retrieval modes for accurate context fetching.
- Each chunk contains structured metadata: `doc_id`, `chunk_id`, `entity_ids`, `section_type`, and `page`.

### 🧠 Local LLM Inference
- Uses **Ollama** for fully local inference — no external API keys required.
- Compatible with models like `llama3.1:latest`, `qwen2.5:14b-instruct`, or `mistral`.
- Performs labeling, summarization, and final QA synthesis.
- Includes deterministic **mock mode** for offline testing and CI.

### ⚙️ FastMCP Servers
- Two modular servers expose tools via **FastMCP 2.x**:
  - `rag` → `rag.search`, `rag.embed_and_index`, `rag.reindex`, `rag.delete`, `rag.health`, `rag.qa`
  - `kg` → `sparql_query`, `sparql_update`, `push_labels`, `validate_labels`, `list_documents`, `kg.health`
- Both run locally via stdio and are MCP-Coordinator compatible.

### 🔒 Privacy & Portability
- 100% offline operation — suitable for air-gapped or research environments.
- Reproducible local stack (GraphDB + Chroma + Ollama + FastMCP).
- Works seamlessly on Windows 11, macOS, or Linux.

### 🚀 Integration Ready
- Plug-and-play with **MCP Coordinators** or **Streamlit apps** for end-user Q&A.
- Can interoperate with other MCPs such as:
  - Brave API MCP (web search)
  - MongoDB MCP (strategy data)
  - Telegram MCP (messaging)
  - Gmail MCP (email retrieval)
- Returns clean JSON outputs for easy chaining into agentic workflows.

---

## 3️⃣ 🏗️ Architecture

The **GraphRAG MCP** architecture combines **Knowledge Graph reasoning**, **Vector-based retrieval**, and **Local LLM synthesis** — all under the **MCP** interoperability standard.  
It’s designed for *clarity*, *privacy*, and *modular scalability*.

---

### 🧭 High-Level Overview

| Layer | Technology | Purpose | Example Components |
|:------|:------------|:---------|:--------------------|
| 🗂 **Ingestion Layer** | Python + LangChain | Reads PDFs, splits into semantic chunks, labels with LLMs | `pdf_reader.py`, `semantic_splitter.py`, `llm_chunk_tagger.py` |
| 🧩 **Knowledge Graph Layer (KG)** | GraphDB (Ontotext) + RDFLib | Stores canonical entities (tokens, protocols, organizations) | `graphdb_sink.py`, `namespaces.py`, SHACL shapes |
| 💾 **Vector Retrieval Layer (RAG)** | ChromaDB + Ollama embeddings | Stores text chunks + metadata + embeddings for semantic retrieval | `chroma_store.py`, `.chroma/` |
| ⚙️ **MCP Layer** | FastMCP 2.x | Exposes standardized MCP tools (`rag.*`, `kg.*`) | `rag_server.py`, `kg_server.py` |
| 🧠 **LLM Synthesis Layer** | Ollama LLMs (`llama3.1`, `qwen2.5`) | Answers questions with retrieved context + KG enrichment | `rag.qa`, `llm_chunk_tagger` |
| 💬 **User Interface Layer** | MCP Coordinator / Streamlit | Connects multiple MCPs for conversational Q&A | Coordinator UI or custom Streamlit dashboard |

---

### 🔹 Data Flow Diagram

```text
            ┌────────────────────────────────────────┐
            │              Whitepapers               │
            │ (PDFs, research papers, documentation) │
            └───────────────────┬────────────────────┘
                                │
                                ▼
            ┌─────────────────────────────────────────────┐
            │      📄 Ingestion & Labeling                │
            │  pdf_reader → semantic_splitter →           │
            │  llm_chunk_tagger → postprocess             │
            └───────────────────┬─────────────────────────┘
                                 │
                        ┌────────┴────────┐
                        │                 │
                        ▼                 ▼
            ┌────────────────┐    ┌──────────────────────┐
            │ 🧠 GraphDB KG  │    │ 💾 Chroma RAG      │
            │ Entities & IRIs │   │ Chunks + Embeddings │
            └────────┬────────┘   └────────┬────────────┘
                     │                     │
                     ▼                     ▼
               ┌───────────────┐      ┌───────────────┐
               │ ⚙️ kg_server  │     │ ⚙️ rag_server │
               │ (FastMCP)     │      │ (FastMCP)     │
               └────────┬──────┘      └──────┬────────┘
                        │                    │
                        └────────┬───────────┘
                                 ▼
                  ┌────────────────────────────────┐
                  │ 💬 MCP Coordinator / Streamlit │
                  │  User-facing Q&A Interface     │
                  └────────────────────────────────┘

```

---

### 🧠 How It Works (Step-by-Step)

| Step | Description | Input | Output |
|:----:|:-------------|:------|:--------|
| **1️⃣** | **PDF Parsing** | Whitepaper PDF | Raw text pages |
| **2️⃣** | **Semantic Splitting** | Raw text | Meaningful chunks (by section/topic) |
| **3️⃣** | **LLM Labeling** | Chunk text | Entities, relations, and section labels |
| **4️⃣** | **Postprocessing** | Labeled chunks | Cleaned JSONL with canonical entity IRIs |
| **5️⃣** | **Indexing** | JSONL labels | Chroma embeddings + KG triples |
| **6️⃣** | **Retrieval (rag.search)** | Query text / entities | Relevant chunks |
| **7️⃣** | **Enrichment (optional)** | Retrieved entities | KG aliases, definitions |
| **8️⃣** | **Answer Synthesis (rag.qa)** | Question + context | Concise answer with citations |

---

### 🌐 Data Modalities

| Data Type | Storage | Example |
|:-----------|:--------|:--------|
| 🧱 **Entity** | GraphDB | `<https://kg.mcp.ai/id/token/bitcoin>` → `rdf:type crypto:Token` |
| 📜 **Chunk** | Chroma | “Bitcoin is a peer-to-peer electronic cash system…” |
| 🧩 **Embedding** | Chroma / Ollama | 768-dim `nomic-embed-text` vector |
| 🧮 **Provenance** | Metadata | `doc_id`, `chunk_id`, `page`, `entity_ids[]` |
| 💬 **Answer** | MCP JSON | `{ "answer": "...", "citations": [...] }` |

---

### 🧱 Core MCP Tools

| Server | Tool | Description |
|:--------|:------|:-------------|
| 🧩 **RAG** | `rag.search` | Semantic search over chunks |
| | `rag.embed_and_index` | Add new labeled chunks to index |
| | `rag.reindex` | Rebuild from outputs directory |
| | `rag.delete` | Delete by IDs or filters |
| | `rag.qa` | Question answering with LLM synthesis |
| | `rag.health` | Diagnostics and store info |
| 🧠 **KG** | `sparql_query` / `sparql_update` | Execute SPARQL against GraphDB |
| | `push_labels` / `validate_labels` | Add or validate KG entries |
| | `list_documents`, `get_chunk` | Retrieve document metadata |
| | `kg.health` | Check GraphDB repository status |

---

### 🧬 Technology Stack Summary

| Category | Technology | Purpose |
|:----------|:------------|:---------|
| **Language** | 🐍 Python 3.11 | Core pipeline and servers |
| **LLM Backend** | 🧠 Ollama | Local inference for labeling & QA |
| **Vector Store** | 💾 ChromaDB | Embeddings and chunk retrieval |
| **Knowledge Graph** | 🧩 GraphDB | RDF-based entity storage |
| **Interoperability** | ⚙️ FastMCP 2.x | Exposes tools for coordinators |
| **Testing** | 🧪 Pytest | Offline & integration tests |
| **Visualization / UI** | 💬 Streamlit / MCP Coordinator | Front-end for user Q&A |

---

## 4️⃣ ⚙️ Installation & Setup

Set up your local **GraphRAG MCP environment** in just a few steps!  
This stack runs fully offline and integrates seamlessly with **Ollama**, **GraphDB**, and **Chroma**.

---

### 🧾 Prerequisites

| Requirement | Description | Example |
|:-------------|:-------------|:----------|
| 🐍 **Python** | Version **3.11+** recommended | `python --version` → `Python 3.11.8` |
| 🧠 **Ollama** | Local LLM runtime (for inference + embeddings) | `ollama pull llama3.1:latest` |
| 🧩 **GraphDB Desktop 11+** | Local Knowledge Graph database | runs at `http://localhost:7200` |
| 💾 **ChromaDB** | Vector store for embeddings | auto-initialized under `.chroma/` |
| 🧰 **FastMCP** | Multi-Component Platform runtime (2.x) | installed via `pip` |

---

### 🧱 Folder Layout (simplified)

| Folder | Purpose | Example Contents |
|:--------|:----------|:----------------|
| `src/` | Core codebase | `pipeline.py`, `mcp/`, `kg/`, `rag/` |
| `outputs/run_simple/` | Generated outputs | labeled chunks, reports, embeddings |
| `.chroma/` | Chroma persistent vector store | `chroma.sqlite3`, `index/` |
| `.env` | Environment configuration | Ollama, GraphDB, Chroma settings |
| `tests/` | Offline unit tests | `test_rag_qa.py`, `test_kg_server.py` |

---

### 🧰 Step-by-Step Setup

#### 🪄 1️⃣ Clone & Create Virtual Environment
\`\`\`bash
git clone <your_repo_url>
cd GraphRAG_MCP
python -m venv .venv
\`\`\`

#### ⚡ 2️⃣ Activate Environment
| OS | Command |
|:---|:---------|
| 🪟 **Windows (PowerShell)** | `.venv\Scripts\activate` |
| 🐧 **Linux / macOS** | `source .venv/bin/activate` |

#### 📦 3️⃣ Install Dependencies
\`\`\`bash
pip install -r requirements.txt
\`\`\`

#### ⚙️ 4️⃣ Verify Installation
\`\`\`bash
python -m src.mcp.rag_server --list-tools
python -m src.mcp.kg_server --list-tools
\`\`\`

✅ You should see tools like **`rag.qa`**, **`rag.search`**, and **`kg.health`**.

---

### 🧠 Optional: Preload Ollama Models

| Model | Purpose | Pull Command |
|:-------|:----------|:--------------|
| 🦙 **llama3.1:latest** | Default reasoning + summarization model | `ollama pull llama3.1:latest` |
| 🧩 **nomic-embed-text** | Embedding model for RAG vectorization | `ollama pull nomic-embed-text` |
| 🤖 **qwen2.5:14b-instruct** | Larger model for complex QA tasks | `ollama pull qwen2.5:14b-instruct` |

---

### 🔍 Quick Sanity Check

Run a quick health diagnostic to ensure everything is configured correctly:

\`\`\`bash
pytest -q
python -m src.mcp.rag_server --run-tool rag.health
python -m src.mcp.kg_server --run-tool kg.health
\`\`\`

If both return ✅ **OK**, you’re ready to run the pipeline and start querying your **Knowledge Graph + RAG** system!

