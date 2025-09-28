import os
import json
import time
from typing import Dict, Any

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ✅ correct relative import (file is in the same 'classify' package)
from .heuristic_tagger import tag_chunk_heuristic

SYSTEM_PROMPT = (
    "You label ONLY what the given text explicitly supports. "
    "No external knowledge. Return STRICT JSON with keys:\n"
    "  section_type: array of strings,\n"
    "  content_role: array of strings,\n"
    "  entities: { token:[], protocol:[], component:[], organization:[] },\n"
    "  relations: [ {subject, predicate, object, confidence, evidence_span:[start,end]} ],\n"
    "  keyphrases: array of strings,\n"
    "  confidence_overall: number (0..1)\n"
    "Offsets in evidence_span are relative to the provided text. "
    "If unsure, leave arrays empty. No commentary—JSON ONLY."
)

USER_PROMPT_TMPL = """Document title: {title}

Text:
{chunk_text}

Respond with pure JSON, no prose, no code fences.
"""

def _make_llm() -> OllamaLLM | None:
    model = os.getenv("OLLAMA_MODEL", "").strip()
    base_url = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434").strip()
    if not model:
        return None
    try:
        llm = OllamaLLM(model=model, base_url=base_url, temperature=0.2)
        return llm
    except Exception:
        return None

def _parse_json_maybe(text: str) -> Dict[str, Any] | None:
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    try:
        return json.loads(s)
    except Exception:
        return None

def tag_chunk(text: str, title: str = "") -> dict:
    """
    Try LangChain+Ollama first for JSON labels.
    Falls back to a deterministic heuristic if LLM unavailable or parsing fails.
    """
    llm = _make_llm()
    if llm is None:
        return tag_chunk_heuristic(text, title=title)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TMPL),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    for _ in range(2):
        try:
            out = chain.invoke({"title": title, "chunk_text": text})
            obj = _parse_json_maybe(out)
            if obj and isinstance(obj, dict):
                obj.setdefault("section_type", ["unknown"])
                obj.setdefault("content_role", [])
                obj.setdefault("entities", {"token": [], "protocol": [], "component": [], "organization": []})
                obj.setdefault("relations", [])
                obj.setdefault("keyphrases", [])
                obj.setdefault("confidence_overall", 0.5)
                return obj
        except Exception:
            time.sleep(0.5)

    return tag_chunk_heuristic(text, title=title)