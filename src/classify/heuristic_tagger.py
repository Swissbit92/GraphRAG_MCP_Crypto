import re

TOKENS = [
    "Bitcoin","BTC","Ethereum","ETH","Cardano","ADA","Solana","SOL",
    "Polkadot","DOT","Binance","BNB","PAX Gold","PAXG","Chainlink","LINK",
    "Aave","AAVE","Basic Attention Token","BAT","Bittensor","TAO","Paxos"
]
ORGS = ["Binance","Paxos","Web3 Foundation","Solana Labs","IOHK","Input Output","Ethereum Foundation","Brave"]

COMPONENT_HINTS = [
    "oracle","data feed","aggregator","dex aggregator","consensus","validator",
    "staking","relay","bridge","smart contract","vm","evm","slot","leader","sequencer"
]
CONSENSUS_HINTS = [
    "proof of work","proof-of-work","pow",
    "proof of stake","proof-of-stake","pos",
    "ouroboros","nominated proof-of-stake","npos","proof of history","poh","grandpa","babe","tendermint"
]

REL_PATTERNS = [
    (re.compile(r"\buses?\s+(?P<object>proof[-\s]of[-\s]\w+|ouroboros|poh|evm|oracles?)", re.I), "uses"),
    (re.compile(r"\bprovides?\s+(?P<object>data feeds?|price feeds?|liquidity|staking)\b", re.I), "provides"),
    (re.compile(r"\bserves?\s+as\s+(?P<object>gas|fee|governance|collateral)\b", re.I), "serves_as"),
]

def _find_terms(text, candidates):
    found = []
    low = text.lower()
    for c in candidates:
        if c.lower() in low:
            found.append(c)
    out, seen = [], set()
    for x in found:
        if x.lower() not in seen:
            out.append(x)
            seen.add(x.lower())
    return out

def _component_terms(text):
    hits, low = [], text.lower()
    for h in COMPONENT_HINTS + CONSENSUS_HINTS:
        if h in low:
            hits.append(h)
    return sorted(set(hits))

def tag_chunk_heuristic(text: str, title: str = "") -> dict:
    tokens = _find_terms(text, TOKENS)
    orgs = _find_terms(text, ORGS)
    comps = _component_terms(text)

    relations = []
    for rx, pred in REL_PATTERNS:
        for m in rx.finditer(text):
            obj = m.group("object").strip()
            subj = tokens[0] if tokens else (title.split()[0] if title else "Unknown")
            relations.append({
                "subject": subj,
                "predicate": pred,
                "object": obj,
                "confidence": 0.55,
                "evidence_span": [m.start(), m.end()]
            })

    section_type = []
    low = text.lower()
    if any(k in low for k in ("abstract", "introduction")):
        section_type.append("intro/abstract")
    if any(k in low for k in ("tokenomics","economics","supply","burn","issuance","collateral")):
        section_type.append("economics")
    if any(k in low for k in ("consensus","validator","leader","slot","pow","pos","ouroboros","poh")):
        section_type.append("consensus/architecture")

    return {
        "section_type": section_type or ["unknown"],
        "content_role": [],
        "entities": {
            "token": tokens,
            "protocol": [t for t in tokens if t in ["Bitcoin","Ethereum","Cardano","Solana","Polkadot","Binance"]],
            "component": comps,
            "organization": orgs
        },
        "relations": relations,
        "keyphrases": sorted(set(comps + tokens))[:10],
        "confidence_overall": 0.5 if (tokens or comps or relations) else 0.2
    }
