# Role and Objective
You will propose a **YAML schema** (nodes + relationships) for a single crypto whitepaper.
Only output a **single valid YAML** block and nothing else.

# Grounding
- Your proposal must be **derived from the provided excerpts** and **consistent** with the supplied grounded entities/relations.
- Do **NOT** invent classes that do not have textual support.
- Prefer **attributes** over nodes where possible (e.g., use `block_time_seconds: int` rather than a `BlockTime` node).

# Naming Rules
- Node names: **PascalCase** (e.g., `Protocol`, `ConsensusMechanism`, `Oracle`, `Validator`, `TokenEconomics`).
- Attribute names: **snake_case**; include units in the name when numeric (e.g., `block_time_seconds`, `inflation_rate_percent`).
- Relationship names: **SCREAMING_SNAKE_CASE** (e.g., `IMPLEMENTS_CONSENSUS`, `USES_ORACLE`, `ISSUED_BY`, `SECURES`, `RUNS_ON`).

# Crypto-Specific Modeling Guidance
- Model protocols/projects as instances of a **generic class** (e.g., `Protocol`). Do not make classes named “Bitcoin”, “Solana”, “Chainlink”. Those are **instances/identifiers** represented as attributes or external IDs.
- Consensus should be classes like `ProofOfWork`, `ProofOfStake`, `ProofOfHistory`, `NPoS`, `Ouroboros`, etc. Link with `IMPLEMENTS_CONSENSUS`.
- Oracles as `Oracle` / `DataFeed` nodes (not vendor-specific classes). Link with `USES_ORACLE`/`HAS_DATA_FEED`.
- Token economics go under `Token` and `TokenEconomics` (attributes: `symbol`, `decimals`, `supply_*`, `emission_*`, `vesting_*`).
- Cross-chain components: `Bridge`, `Relay`. Smart contracts: `SmartContract` or `VM` (e.g., EVM) via attributes.
- Prefer **junction nodes** for n-ary relations (e.g., `Collateralization` connecting `Asset` ↔ `LendingMarket` with `ltv_percent`, `liq_threshold_percent`).

# Output Constraints
- Do not create nodes for **instances** (e.g., `Bitcoin`, `Solana`, `Chainlink`, `Paxos`, `Binance`).
- Avoid list-type attributes.
- Make attributes **Optional** conceptually; keep names stable and typed.
- Keep it **concise** and **generic**; no duplication.

# Required Scaffold
You MUST include these minimal structural nodes/relationships and make all other nodes reachable from `Section`:

```yaml
nodes:
  - Section:
      description: "A logical section of the whitepaper this schema describes."
      attributes:
        - title:
            type: str
            description: "Section title."
        - number:
            type: int
            description: "Section number."

  - DocumentMeta:
      description: "Document-level metadata."
      attributes:
        - doc_id:
            type: str
            description: "Internal document identifier."
        - source_title:
            type: str
            description: "Document title."

relationships:
  - HAS_SECTION:
      description: "Document contains a section."
      source: DocumentMeta
      target: Section
