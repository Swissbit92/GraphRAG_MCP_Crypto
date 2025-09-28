from textwrap import dedent

def get_crypto_ontology_generation_prompt() -> str:
    """
    System prompt to produce a YAML-only schema proposal from a crypto/blockchain whitepaper segment.
    Stricter: reply must start with 'nodes:' and contain ONLY YAML. No prose, no prelude.
    """
    return dedent("""\
    You are a schema designer for crypto/blockchain whitepapers.

    ## INPUT
    You will receive:
      - Minimal document metadata (title, pages, etc.)
      - A small list of grounded terms (entities/phrases actually present)
      - A TEXT SEGMENT from the whitepaper

    ## HARD RULES
    - Reply with **ONLY YAML** and **nothing else**.
    - Your first line MUST be exactly `nodes:` (no backticks, no fences).
    - Node names: PascalCase. Attribute names: snake_case. Relationship names: SCREAMING_SNAKE_CASE.
    - Put units in numeric attribute names: e.g., block_time_seconds, fee_bps, inflation_rate_percent.
    - Prefer attributes over nodes when reasonable.
    - DO NOT invent entities not derivable from the input. Stay grounded.
    - DO NOT create nodes for specific named instances (e.g., "Bitcoin", "Solana", "Chainlink", "PAX Gold").
      Those are instances of generic classes like Protocol, Token, Organization, OracleNetwork, etc.
    - If the segment is too generic to infer domain nodes, at least output the baseline nodes and relationship below.

    ## REQUIRED BASELINE (always include)
    nodes:
      - Whitepaper:
          description: "A specific whitepaper document."
          attributes:
            - title:
                type: str
                description: "Document title"
            - sha1:
                type: str
                description: "Document SHA1"
            - pages:
                type: int
                description: "Total pages"
            - filename:
                type: str
                description: "Filename if known"

      - WhitepaperSection:
          description: "A section/segment of a whitepaper."
          attributes:
            - section_id:
                type: str
                description: "Section identifier or synthetic id"
            - title:
                type: str
                description: "Optional title, if identifiable"
            - page_start:
                type: int
                description: "Start page number if known"
            - page_end:
                type: int
                description: "End page number if known"

    relationships:
      - HAS_SECTION:
          description: "Whitepaper contains a section."
          source: Whitepaper
          target: WhitepaperSection

    ## DOMAIN HINTS (use ONLY if grounded by the text)
    - Protocol, Token, Organization, ConsensusMechanism (ProofOfWork, ProofOfStake, ProofOfHistory),
      OracleNetwork, Validator, Node, SmartContract, AMM, DEX, LendingMarket, Collateralization,
      DataFeed, Governance, Treasury, EconomicParameter, FeePolicy, MonetaryPolicy, Staking, Slashing,
      Rollup, Sidechain, VM, EVMCompatibility, CrossChainMessage, LiquidityPool, StablecoinMechanism,
      Custodian, Reserve, MintBurnMechanism, PegMechanism.
    - Example attributes: block_time_seconds, finality_seconds, validator_count, token_symbol,
      inflation_rate_percent, staking_yield_percent, oracle_update_interval_seconds,
      collateral_factor_percent, liquidation_threshold_percent, supply_cap_tokens, circulating_supply_tokens, fee_bps.

    ## OUTPUT SHAPE (exact keys)
    nodes:
      - NodeName:
          description: "Short description"
          attributes:
            - attribute_name:
                type: pythonic_type
                description: "Short description"
                examples:
                  - example_value
    relationships:
      - RELATIONSHIP_TYPE:
          description: "Short description"
          source: SourceNode
          target: TargetNode
    """)
