# src/util/chache.py
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

class JsonCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha1(s: str) -> str:
        h = hashlib.sha1()
        h.update(s.encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def get(self, key_text: str) -> Optional[Dict[str, Any]]:
        fname = self.root / f"{self._sha1(key_text)}.json"
        if not fname.exists():
            return None
        try:
            return json.loads(fname.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key_text: str, obj: Dict[str, Any]) -> None:
        fname = self.root / f"{self._sha1(key_text)}.json"
        try:
            fname.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
