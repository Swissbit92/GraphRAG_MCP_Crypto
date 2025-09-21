import re
from typing import List, Dict

HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*)\s+.+$")  # naive numbered heading
MIN_CHARS, MAX_CHARS = 600, 1200

def split_pages_to_chunks(pages: List[str]) -> List[Dict]:
    """
    Returns list of {page, char_start, char_end, text}
    Strategy:
      1) split by blank lines and headings
      2) merge small segments until MIN_CHARS
      3) cap at MAX_CHARS on sentence boundaries if possible
    """
    chunks = []
    for p_idx, page in enumerate(pages, start=1):
        raw_lines = page.splitlines()
        # coarse segments by blank lines or headings
        segs = []
        buf = []
        for ln in raw_lines:
            if HEADING_RE.match(ln.strip()) or not ln.strip():
                if buf:
                    segs.append("\n".join(buf).strip())
                    buf = []
                if HEADING_RE.match(ln.strip()):
                    segs.append(ln.strip())
            else:
                buf.append(ln)
        if buf:
            segs.append("\n".join(buf).strip())

        # merge to size window
        acc = ""
        for seg in segs:
            if not seg.strip():
                continue
            cand = (acc + "\n\n" + seg).strip() if acc else seg
            if len(cand) < MIN_CHARS:
                acc = cand
                continue
            # if acc + seg too large, flush acc then start new with seg
            if len(cand) > MAX_CHARS and acc:
                chunks.append({"page": p_idx, "text": acc})
                acc = seg
            else:
                acc = cand
            if len(acc) >= MIN_CHARS and len(acc) <= MAX_CHARS:
                chunks.append({"page": p_idx, "text": acc})
                acc = ""
        if acc.strip():
            chunks.append({"page": p_idx, "text": acc.strip()})
    # annotate offsets within chunk (we keep local start/end = 0..len-1)
    out = []
    for i, ch in enumerate(chunks):
        txt = ch["text"]
        out.append({
            "chunk_idx": i,
            "page": ch["page"],
            "char_start": 0,
            "char_end": len(txt),
            "text": txt
        })
    return out
