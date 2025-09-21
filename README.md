# GraphDB_Desktop
Local MCP friendly knowledge Graph with SHACL shapes and GraphQL endpoints. Hosting Digital Assets (Crypto) and Crypto Trading information 

whitepaper-explore/
├─ .env                          # optional: OLLAMA_MODEL=qwen2.5:14b-instruct
├─ requirements.txt
├─ data/
│  └─ pdfs/                      # put your PDFs here (bitcoin, cardano, chainlink, binance, 1inch)
├─ outputs/
│  └─ .gitkeep
└─ src/
   ├─ pipeline.py
   ├─ ingest/
   │  └─ pdf_reader.py
   ├─ chunking/
   │  └─ semantic_splitter.py
   ├─ classify/
   │  └─ llm_chunk_tagger.py
   ├─ cluster/
   │  └─ simple_communities.py
   └─ reports/
      └─ summary.py
