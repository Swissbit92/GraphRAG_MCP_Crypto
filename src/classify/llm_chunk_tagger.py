import os, json, time
import requests

SYSTEM_PROMPT = (
    "You label ONLY what the given text explicitly supports. "
    "No external knowledge. Return compact JSON with keys: "
    "section_type (array of strings), content_role (array), "
    "entities (object with arrays: token, protocol, component, organization), "
    "relations (array of {subject,predicate,object,confidence,evidence_span:[start,end]}), "
    "keyphrases (array), confidence_overall (0..1). "
    "Use evidence_span offsets relative to the provided text."
)

def _ollama_chat(messages, base, model):
    r = requests.post(f"{base}/api/chat", json={"model": model, "messages": messages, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

def tag_chunk(text: str, title: str = "") -> dict:
    base = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")
    model = os.getenv("OLLAMA_MODEL", "").strip()
    if not model:
        # Fallback: very simple heuristic tags (no LLM)
        return {
            "section_type": ["unknown"],
            "content_role": [],
            "entities": {"token": [], "protocol": [], "component": [], "organization": []},
            "relations": [],
            "keyphrases": list(sorted({w.lower() for w in text.split() if w.istitle()}) )[:10],
            "confidence_overall": 0.2
        }
    prompt = (
        f"Document title: {title}\n"
        f"Text:\n{text}\n\n"
        "Respond with pure JSON, no prose."
    )
    for attempt in range(2):
        try:
            raw = _ollama_chat(
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                base=base, model=model
            )
            # be forgiving about leading/trailing code fences
            raw = raw.strip().strip("`")
            if raw.startswith("json"): raw = raw[4:].strip()
            return json.loads(raw)
        except Exception:
            time.sleep(1)
    # last-resort fallback
    return {
        "section_type": ["unknown"],
        "content_role": [],
        "entities": {"token": [], "protocol": [], "component": [], "organization": []},
        "relations": [],
        "keyphrases": [],
        "confidence_overall": 0.0
    }
