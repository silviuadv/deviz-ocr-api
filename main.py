import os
import re
import json
import base64
import requests
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

# --- BigQuery imports (safe) ---
try:
    from google.cloud import bigquery
    from google.oauth2 import service_account
except Exception:
    bigquery = None
    service_account = None


app = FastAPI(
    title="Deviz Parser Hybrid: Vision Layout Primary + GPT Fallback",
    version="0.3.0"
)

# =========================
# Models
# =========================

class DevizUrlPayload(BaseModel):
    url: str


class ExtractedItem(BaseModel):
    raw: Optional[str] = ""
    desc: str = ""
    kind: str = Field(default="part")  # part / labor
    qty: float = 0.0
    unit: str = ""
    unit_price: float = 0.0
    line_total: float = 0.0
    currency: str = "RON"
    warnings: List[str] = Field(default_factory=list)
    code: str = ""  # part code


class Totals(BaseModel):
    materials: Optional[float] = None
    labor: Optional[float] = None
    vat: Optional[float] = None
    subtotal_no_vat: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "RON"


class DocumentMeta(BaseModel):
    default_currency: str = "RON"
    service_name: Optional[str] = None
    client_name: Optional[str] = None
    vehicle: Optional[str] = None
    date: Optional[str] = None
    number: Optional[str] = None


class DevizResponse(BaseModel):
    document: DocumentMeta
    totals: Totals
    items: List[ExtractedItem]
    warnings: List[str] = Field(default_factory=list)
    sum_guess_from_lines: Optional[float] = None
    ocr_text_raw: Optional[str] = None
    debug: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Helpers: numbers, text
# =========================

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)

        s = str(x).strip()
        if not s:
            return None

        s = s.replace(" ", "")

        # Case A: has BOTH comma and dot -> assume comma = thousands, dot = decimal (e.g. 6,000.0)
        if "," in s and "." in s:
            s = s.replace(",", "")
            return float(s)

        # Case B: only comma
        if "," in s and "." not in s:
            # thousands grouping
            if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
                s = s.replace(",", "")
                return float(s)
            # comma as decimal
            if re.fullmatch(r"\d+,\d{1,4}", s):
                s = s.replace(",", ".")
                return float(s)
            # fallback: strip commas
            s = s.replace(",", "")
            return float(s)

        # Case C: only dot
        if "." in s and "," not in s:
            return float(s)

        # Case D: plain int
        if re.fullmatch(r"\d+", s):
            return float(s)

        return None
    except Exception:
        return None


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _contains_any(hay: str, needles: List[str]) -> bool:
    h = (hay or "").lower()
    return any(n.lower() in h for n in needles)


def _approx_equal(a: Optional[float], b: Optional[float], tol: float = 2.0) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


# =========================
# CODE extraction + validation
# =========================

STOP_CODES = {
    "h", "ore", "ora", "buc", "buc.", "pcs", "um", "u/m",
    "cantitate", "pret", "valoare", "total",
    "materiale", "manopera", "operatie", "cod", "nr", "nr."
}

def _clean_code_candidate(s: str) -> str:
    s = (s or "").strip().strip('"').strip()
    s = re.sub(r"\s+", "", s)
    return s

def _is_row_index_code(s: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}\.?", s))

def _is_valid_part_code(code: str) -> bool:
    c = _clean_code_candidate(code)
    if not c:
        return False

    low = c.lower()
    if low in STOP_CODES:
        return False

    if _is_row_index_code(c):
        return False

    # alphanumeric with letters+digits
    if len(c) >= 4 and re.search(r"[A-Za-z]", c) and re.search(r"\d", c):
        return True

    # with separators
    if len(c) >= 6 and re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z\.\-_/]+", c) and re.search(r"[.\-_/]", c):
        return True

    # numeric with dot segments: 365.372 etc
    if re.fullmatch(r"\d{2,6}(\.\d{2,6}){1,3}", c):
        return True

    return False

def _extract_code_from_text(raw_line: str, desc: str) -> Tuple[str, str]:
    candidates: List[str] = []

    for src in [raw_line or "", desc or ""]:
        s = _norm_spaces(src)
        if not s:
            continue
        toks = s.split()
        for t in toks[:3]:
            candidates.append(t)

    m = re.search(r"\bID[:\s]*([0-9A-Za-z\.\-_/]+)\b", raw_line or "", flags=re.IGNORECASE)
    if m:
        candidates.insert(0, m.group(1))

    for cand in candidates:
        c = _clean_code_candidate(cand)
        if _is_valid_part_code(c):
            desc2 = desc or ""
            if desc2.strip().startswith(cand.strip()):
                desc2 = _norm_spaces(desc2.strip()[len(cand.strip()):]).lstrip(" -:/")
            return c, desc2

    return "", desc or ""


# =========================
# Post-processing items (SAFE)
# =========================

_TOTAL_MARKERS = [
    "total", "subtotal", "total deviz", "total reparatie", "total manopera", "total materiale",
    "cost reparatie", "tot.man", "tot. mat", "tva"
]

# verbs that usually indicate labor/operation lines (when there is no clear material structure)
_LABOR_VERB_PREFIXES = [
    "inlocuit", "inlocuire", "schimb", "revizie", "control", "testare", "verificare",
    "montare", "demontare", "diagnostic", "reparatie", "reglaj", "curatare", "completare",
    "programare", "resetare", "calibrare"
]

def _looks_like_total_line(desc: str, raw: str) -> bool:
    s = (_norm_spaces(desc) + " " + _norm_spaces(raw)).lower()
    if any(m in s for m in _TOTAL_MARKERS):
        return True
    # "2. TOTAL" etc
    if re.fullmatch(r"\d+\.?\s*total.*", s.strip()):
        return True
    return False

def _looks_like_operation_line(desc: str, raw: str) -> bool:
    d = (_norm_spaces(desc) or "").lower()
    r = (_norm_spaces(raw) or "").lower()
    # starts with verb-ish
    if any(d.startswith(v) for v in _LABOR_VERB_PREFIXES):
        return True
    # patterns like "REVIZIE 0-" / "INLOCUIT ... 0-"
    if "0-" in d or "0-" in r:
        if any(v in d for v in _LABOR_VERB_PREFIXES) or any(v in r for v in _LABOR_VERB_PREFIXES):
            return True
    return False

def _postprocess_items(items: List[ExtractedItem]) -> List[ExtractedItem]:
    out: List[ExtractedItem] = []
    for it in (items or []):
        desc = it.desc or ""
        raw = it.raw or ""

        # drop TOTAL/SUBTOTAL lines
        if _looks_like_total_line(desc, raw):
            continue

        # If GPT marked as part but it's clearly an operation line, flip to labor
        if (it.kind or "").lower() == "part":
            if _looks_like_operation_line(desc, raw):
                # do NOT touch real parts that have codes (strong signal it's a part)
                if not (it.code and _is_valid_part_code(it.code)):
                    it.kind = "labor"
                    it.unit = "ore" if (it.unit or "").lower() in ["h", "ore", "ora", "oras"] else (it.unit or "")
                    it.code = ""  # labor code empty

        # Ensure labor never carries code
        if (it.kind or "").lower() == "labor":
            it.code = ""

        out.append(it)
    return out


# =========================
# Google Vision OCR
# =========================

class _Word:
    __slots__ = ("t", "x", "y", "h")
    def __init__(self, t: str, x: int, y: int, h: int):
        self.t = t
        self.x = x
        self.y = y
        self.h = h


def _vision_call(image_bytes: bytes) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_VISION_API_KEY")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    body = {
        "requests": [{
            "image": {"content": base64.b64encode(image_bytes).decode("utf-8")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }
    r = requests.post(url, json=body, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Google Vision error: {r.text}")
    return r.json()


def _extract_words_and_raw_text(vision_json: Dict[str, Any]) -> Tuple[List[_Word], str]:
    try:
        resp0 = (vision_json.get("responses") or [])[0] or {}
    except Exception:
        resp0 = {}

    raw_text = ""
    try:
        raw_text = resp0.get("fullTextAnnotation", {}).get("text", "") or ""
    except Exception:
        raw_text = ""

    words: List[_Word] = []
    pages = resp0.get("fullTextAnnotation", {}).get("pages", []) or []
    for page in pages:
        for block in page.get("blocks", []) or []:
            for para in block.get("paragraphs", []) or []:
                for w in para.get("words", []) or []:
                    txt = "".join((s.get("text", "") for s in (w.get("symbols", []) or [])))
                    verts = (w.get("boundingBox", {}) or {}).get("vertices", []) or []
                    if not verts:
                        continue
                    x = int(verts[0].get("x", 0) or 0)
                    y = int(verts[0].get("y", 0) or 0)
                    yb = int((verts[2].get("y", y + 10) if len(verts) > 2 else (y + 10)) or (y + 10))
                    h = max(1, yb - y)
                    if txt.strip():
                        words.append(_Word(txt, x, y, h))
    return words, raw_text


def reconstruct_lines_from_words(words: List[_Word]) -> List[str]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.y, w.x))
    lines: List[List[_Word]] = []

    for w in words_sorted:
        placed = False
        for line in lines:
            avg_y = sum(x.y for x in line) / len(line)
            avg_h = sum(x.h for x in line) / len(line)
            if abs(w.y - avg_y) < (avg_h * 0.55):
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    out: List[str] = []
    for line in lines:
        line = sorted(line, key=lambda w: w.x)
        out.append(_norm_spaces(" ".join(w.t for w in line)))
    return [x for x in out if x]


# =========================
# Parser A (generic lines)
# =========================

def _split_tail_numbers(line: str) -> Tuple[str, List[float]]:
    parts = line.split()
    nums: List[float] = []
    for p in reversed(parts):
        val = _safe_float(p)
        if val is None:
            break
        nums.insert(0, val)
    desc = " ".join(parts[: max(0, len(parts) - len(nums))])
    return _norm_spaces(desc), nums


def parse_generic_lines(lines: List[str], currency: str = "RON") -> List[ExtractedItem]:
    items: List[ExtractedItem] = []

    noise = [
        "pagina", "lucrarile", "perioada", "certificat", "garantie",
        "preturile nu contin tva", "generat cu autodeviz"
    ]

    for line in lines:
        l = _norm_spaces(line)
        if len(l) < 4:
            continue
        if _contains_any(l, noise):
            continue

        # do not parse totals here
        if _looks_like_total_line(l, l):
            continue

        desc, nums = _split_tail_numbers(l)
        if len(nums) < 2:
            continue

        qty = nums[0]
        unit_price = nums[1]
        line_total = nums[-1]

        if len(nums) >= 3:
            q, p, t = nums[-3], nums[-2], nums[-1]
            if abs((q * p) - t) <= 2.0:
                qty, unit_price, line_total = q, p, t
            else:
                if len(nums) == 2:
                    line_total = qty * unit_price

        kind = "part"
        unit = ""
        dlow = desc.lower()
        if "manopera" in dlow:
            kind = "labor"
            unit = "ore"
            desc = re.sub(r"(?i)\bmanopera\b", "", desc).strip()

        desc = re.sub(r"^\d+\.?\s*", "", desc).strip()
        if not desc:
            continue

        code = ""
        desc2 = desc
        if kind == "part":
            code, desc2 = _extract_code_from_text(raw_line=l, desc=desc)

        items.append(ExtractedItem(
            raw=l,
            desc=desc2,
            kind=kind,
            qty=float(qty or 0.0),
            unit=unit,
            unit_price=float(unit_price or 0.0),
            line_total=float(line_total or 0.0),
            currency=currency,
            warnings=[],
            code=code
        ))

    return items


# =========================
# Parser B (table aware - AutoDeviz)
# =========================

def _detect_autodeviz_table(lines: List[str]) -> bool:
    joined = "\n".join(lines[:200]).lower()
    return (
        ("operatie" in joined and "material" in joined and "timp" in joined and ("u.m" in joined or "um" in joined))
        or ("lucrari convenite" in joined)
    )


def _cluster_lines_with_words(words: List[_Word]) -> List[List[_Word]]:
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w.y, w.x))
    lines: List[List[_Word]] = []
    for w in words_sorted:
        placed = False
        for line in lines:
            avg_y = sum(x.y for x in line) / len(line)
            avg_h = sum(x.h for x in line) / len(line)
            if abs(w.y - avg_y) < (avg_h * 0.55):
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])
    for line in lines:
        line.sort(key=lambda w: w.x)
    return lines


def _line_text(line_words: List[_Word]) -> str:
    return _norm_spaces(" ".join(w.t for w in line_words))


def _parse_autodeviz_rows(words: List[_Word], currency: str = "RON") -> List[ExtractedItem]:
    lines_words = _cluster_lines_with_words(words)

    header_idx = None
    for i, lw in enumerate(lines_words):
        t = _line_text(lw).lower()
        if "operatie" in t and ("material" in t or "cant" in t) and ("timp" in t or "val" in t):
            header_idx = i
            break
    if header_idx is None:
        for i, lw in enumerate(lines_words):
            t = _line_text(lw).lower()
            if "lucrari convenite" in t:
                header_idx = i
                break
    if header_idx is None:
        return []

    items: List[ExtractedItem] = []

    def token_is_num(tok: str) -> bool:
        c = tok.replace(".", "").replace(",", ".")
        return bool(re.fullmatch(r"\d+(\.\d+)?", c))

    def tok_float(tok: str) -> Optional[float]:
        return _safe_float(tok)

    stop_markers = [
        "total manopera", "total materiale", "total reparatie", "cost reparatie",
        "certificat", "preturile nu contin", "executant", "verificat"
    ]

    for lw in lines_words[header_idx + 1:]:
        text = _line_text(lw)
        low = text.lower()
        if not text:
            continue
        if any(m in low for m in stop_markers):
            break

        if _looks_like_total_line(text, text):
            continue

        tokens = text.split()
        if not tokens:
            continue

        starts_like_code = bool(re.fullmatch(r"\d{3,4}", tokens[0])) or bool(re.fullmatch(r"\d{4}", tokens[0]))
        has_many_nums = sum(1 for t in tokens if token_is_num(t)) >= 2
        if not (starts_like_code or has_many_nums):
            continue

        rest = tokens[1:] if starts_like_code else tokens[:]
        num_positions = [(idx, t) for idx, t in enumerate(rest) if token_is_num(t)]
        if len(num_positions) < 2:
            continue

        i_time, tok_time = num_positions[0]
        i_lval, tok_lval = num_positions[1]

        time_h = tok_float(tok_time) or 0.0
        labor_val = tok_float(tok_lval) or 0.0

        op_tokens = rest[:i_time]
        operation = _norm_spaces(" ".join(op_tokens)).strip(' "')
        if not operation:
            operation = "operatie"

        material_val = None
        unit_price = None
        qty = None
        um = ""

        tail_nums = [(idx, t) for idx, t in enumerate(rest) if token_is_num(t)]
        if tail_nums:
            material_val = tok_float(tail_nums[-1][1])
        if len(tail_nums) >= 2:
            unit_price = tok_float(tail_nums[-2][1])
        if len(tail_nums) >= 3:
            qty = tok_float(tail_nums[-3][1])

        if qty is not None and unit_price is not None:
            qty_pos = None
            for idx, t in reversed(list(enumerate(rest))):
                if token_is_num(t) and _approx_equal(tok_float(t), qty, tol=0.0001):
                    qty_pos = idx
                    break
            if qty_pos is not None and qty_pos + 1 < len(rest):
                um_candidate = rest[qty_pos + 1]
                if not token_is_num(um_candidate):
                    um = um_candidate

        mat_desc = ""
        if qty is not None:
            qty_pos2 = None
            for idx in range(i_lval + 1, len(rest)):
                if token_is_num(rest[idx]) and _approx_equal(tok_float(rest[idx]), qty, tol=0.0001):
                    qty_pos2 = idx
                    break
            if qty_pos2 is not None:
                mat_tokens = rest[i_lval + 1: qty_pos2]
                mat_desc = _norm_spaces(" ".join(mat_tokens)).strip(' "')

        labor_unit_price = 0.0
        labor_warnings: List[str] = []
        if time_h > 0 and labor_val > 0:
            labor_unit_price = labor_val / time_h
        else:
            labor_warnings.append("missing_labor_time_or_value")

        items.append(ExtractedItem(
            raw=text,
            desc=operation,
            kind="labor",
            qty=float(time_h or 0.0),
            unit="ore",
            unit_price=float(labor_unit_price or 0.0),
            line_total=float(labor_val or 0.0),
            currency=currency,
            warnings=labor_warnings,
            code=""
        ))

        if mat_desc and qty is not None and unit_price is not None and material_val is not None:
            part_warnings: List[str] = []
            if abs((qty * unit_price) - material_val) > 2.0:
                part_warnings.append("derived_unit_price_or_total_mismatch")

            code, mat_desc2 = _extract_code_from_text(raw_line=text, desc=mat_desc)

            items.append(ExtractedItem(
                raw=text,
                desc=mat_desc2,
                kind="part",
                qty=float(qty or 0.0),
                unit=(um or ""),
                unit_price=float(unit_price or 0.0),
                line_total=float(material_val or 0.0),
                currency=currency,
                warnings=part_warnings,
                code=code
            ))

    return items


# =========================
# Totals extraction
# =========================

def _extract_totals_from_text(raw_text: str) -> Totals:
    t = (raw_text or "")
    low = t.lower()

    def find_money(patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                val = _safe_float(m.group(1))
                if val is not None:
                    return val
        return None

    totals = Totals(currency="RON")

    totals.materials = find_money([
        r"TOTAL\s+MATERIALE\s*[:\-]?\s*([\d\.\,]+)",
        r"Total\s+materiale\s*[:\-]?\s*([\d\.\,]+)",
    ])
    totals.labor = find_money([
        r"TOTAL\s+MANOPERA\s*[:\-]?\s*([\d\.\,]+)",
        r"Total\s+manopera\s*[:\-]?\s*([\d\.\,]+)",
    ])
    totals.grand_total = find_money([
        r"TOTAL\s+DEVIZ\s*[:\-]?\s*([\d\.\,]+)",
        r"TOTAL\s+REPARATIE\s*[:\-]?\s*([\d\.\,]+)",
        r"Total\s+cu\s+TVA\s*[:\-]?\s*([\d\.\,]+)",
    ])

    cur = "RON"
    if "lei" in low or "ron" in low:
        cur = "RON"
    totals.currency = cur
    return totals


# =========================
# OpenAI fallback (GPT-4o)
# =========================

def _openai_fallback(image_bytes: bytes, raw_text: str) -> DevizResponse:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    model = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"

    schema = {
        "type": "object",
        "properties": {
            "document": {
                "type": "object",
                "properties": {
                    "default_currency": {"type": "string"},
                    "service_name": {"type": ["string", "null"]},
                    "client_name": {"type": ["string", "null"]},
                    "vehicle": {"type": ["string", "null"]},
                    "date": {"type": ["string", "null"]},
                    "number": {"type": ["string", "null"]},
                },
                "required": ["default_currency"]
            },
            "totals": {
                "type": "object",
                "properties": {
                    "materials": {"type": ["number", "null"]},
                    "labor": {"type": ["number", "null"]},
                    "vat": {"type": ["number", "null"]},
                    "subtotal_no_vat": {"type": ["number", "null"]},
                    "grand_total": {"type": ["number", "null"]},
                    "currency": {"type": "string"},
                },
                "required": ["currency"]
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw": {"type": ["string", "null"]},
                        "desc": {"type": "string"},
                        "kind": {"type": "string"},
                        "qty": {"type": "number"},
                        "unit": {"type": "string"},
                        "unit_price": {"type": "number"},
                        "line_total": {"type": "number"},
                        "currency": {"type": "string"},
                        "warnings": {"type": "array", "items": {"type": "string"}},
                        "code": {"type": "string"}
                    },
                    "required": ["desc", "kind", "qty", "unit", "unit_price", "line_total", "currency", "warnings", "code"]
                }
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["document", "totals", "items", "warnings"]
    }

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    prompt = (
        "Extrage date structurate dintr-un deviz auto in format JSON.\n"
        "Vreau sa returnezi:\n"
        "- items: fiecare linie relevanta din manopera si piese (kind = 'labor' sau 'part')\n"
        "- totals: materials, labor, grand_total, currency (RON)\n"
        "Reguli IMPORTANT:\n"
        "- Nu inventa linii 'TOTAL' / 'SUBTOTAL' ca items. Daca vezi TOTAL/SUBTOTAL, NU le pune in items.\n"
        "- Daca o linie e o operatie (ex: INLOCUIT, SCHIMB, REVIZIE, CONTROL, TESTARE, VERIFICARE), marcheaz-o ca kind='labor'.\n"
        "- Daca o linie descrie o piesa/material (filtru, ulei, carcasa, capac, buson etc), marcheaz-o ca kind='part'.\n"
        "- Pentru piese, daca exista un cod (alfanumeric sau numeric cu punct), pune-l in field 'code'.\n"
        "- Daca nu exista cod, pune code=\"\".\n"
    )

    req = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                    {"type": "input_text", "text": "OCR raw text (optional context):\n" + (raw_text or "")[:8000]},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "deviz_schema",
                "schema": schema
            }
        }
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(req),
        timeout=90
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {r.text}")

    data = r.json()
    out_text = None
    try:
        for o in data.get("output", []):
            for c in o.get("content", []):
                if c.get("type") == "output_text":
                    out_text = c.get("text")
                    break
            if out_text:
                break
    except Exception:
        out_text = None

    if not out_text:
        raise HTTPException(status_code=502, detail="OpenAI returned empty output_text")

    parsed = json.loads(out_text)

    items: List[ExtractedItem] = []
    for it in (parsed.get("items") or []):
        try:
            obj = ExtractedItem(**it)
            if obj.kind == "part":
                obj.code = _clean_code_candidate(obj.code)
                if not _is_valid_part_code(obj.code):
                    obj.code = ""
            else:
                obj.code = ""
            items.append(obj)
        except Exception:
            continue

    # SAFE postprocess (fix total + labor/part mislabels)
    items = _postprocess_items(items)

    resp = DevizResponse(
        document=DocumentMeta(**(parsed.get("document", {}) or {"default_currency": "RON"})),
        totals=Totals(**(parsed.get("totals", {}) or {"currency": "RON"})),
        items=items,
        warnings=parsed.get("warnings") or [],
        sum_guess_from_lines=None,
        ocr_text_raw=raw_text,
        debug={"used_parser": "openai_fallback", "fallback_used": True}
    )
    return resp


# =========================
# Scoring + orchestrator
# =========================

def _score_result(resp: DevizResponse) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    items = resp.items or []

    if len(items) < 2:
        reasons.append("too_few_items")

    gt = resp.totals.grand_total
    if gt is None or gt <= 0:
        reasons.append("missing_grand_total")

    m = resp.totals.materials
    l = resp.totals.labor
    if gt and m is not None and l is not None:
        if abs((m + l) - gt) > 10.0:
            reasons.append("totals_mismatch_gt_vs_m_plus_l")

    warn_count = sum(1 for it in items for w in (it.warnings or []) if "derived" in w or "mismatch" in w)
    if len(items) > 0 and (warn_count / max(1, len(items))) > 0.7:
        reasons.append("too_many_derived_warnings")

    ok = len(reasons) == 0
    return ok, reasons


def _build_response_from_primary(words: List[_Word], raw_text: str) -> DevizResponse:
    totals = _extract_totals_from_text(raw_text)
    currency = totals.currency or "RON"

    lines = reconstruct_lines_from_words(words)
    items: List[ExtractedItem] = []

    used_parser = "generic_lines"

    if _detect_autodeviz_table(lines):
        items_tbl = _parse_autodeviz_rows(words, currency=currency)
        if len(items_tbl) >= 2:
            items = items_tbl
            used_parser = "autodeviz_table"
        else:
            items = parse_generic_lines(lines, currency=currency)
            used_parser = "generic_lines"
    else:
        items = parse_generic_lines(lines, currency=currency)
        used_parser = "generic_lines"

    # SAFE postprocess (total + labor/part mislabels)
    items = _postprocess_items(items)

    sum_guess = sum((it.line_total or 0.0) for it in items) if items else None

    if totals.materials is None:
        totals.materials = sum((it.line_total or 0.0) for it in items if it.kind == "part") if items else None
    if totals.labor is None:
        totals.labor = sum((it.line_total or 0.0) for it in items if it.kind == "labor") if items else None
    if totals.grand_total is None and totals.materials is not None and totals.labor is not None:
        totals.grand_total = (totals.materials or 0.0) + (totals.labor or 0.0)

    doc = DocumentMeta(default_currency=currency)

    resp = DevizResponse(
        document=doc,
        totals=totals,
        items=items,
        warnings=[],
        sum_guess_from_lines=sum_guess,
        ocr_text_raw=raw_text,
        debug={
            "used_parser": used_parser,
            "fallback_used": False,
            "lines_used": len(lines),
            "preview_lines": lines[:5],
        }
    )
    return resp


def _process_image(image_bytes: bytes) -> DevizResponse:
    vision_json = _vision_call(image_bytes)
    words, raw_text = _extract_words_and_raw_text(vision_json)

    if not words and not raw_text:
        raise HTTPException(status_code=502, detail="Google Vision returned empty result")

    primary = _build_response_from_primary(words, raw_text)
    ok, reasons = _score_result(primary)

    threshold = float(os.getenv("FALLBACK_SCORE_THRESHOLD", "1").strip() or "1")
    need_fallback = (not ok) and threshold >= 1

    if need_fallback:
        fb = _openai_fallback(image_bytes=image_bytes, raw_text=raw_text)
        fb.warnings = (fb.warnings or []) + ["fallback_used_due_to: " + ",".join(reasons)]
        fb.debug = fb.debug or {}
        fb.debug.update({
            "primary_failed_reasons": reasons,
            "primary_used_parser": primary.debug.get("used_parser"),
        })
        return fb

    if not ok:
        primary.warnings = (primary.warnings or []) + ["primary_low_confidence: " + ",".join(reasons)]

    return primary


# =========================
# Download helper
# =========================

def _download_file(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download file: {r.status_code}")
    return r.content


# =========================
# BigQuery price lookup
# =========================

class PriceLookupRequest(BaseModel):
    desc: str
    brand: Optional[str] = None
    min_price: float = 0.0
    token_limit: int = 6
    observed_price: float = 0.0  # OPTIONAL: pretul din deviz (unit price sau line_total, cum vrei tu)
    debug: bool = False


class PriceLookupResponse(BaseModel):
    source_count: int = 0
    price_min: float = 0.0
    price_median: float = 0.0
    price_max: float = 0.0
    price_p90: float = 0.0
    tokens_used: List[str] = Field(default_factory=list)

    # IMPORTANT: astea trebuie la top-level ca sa apara ca label in Make
    MarketVerdict: str = ""
    MarketConfidence: float = 0.0
    MarketRatioToP90: float = 0.0

    debug: Dict[str, Any] = Field(default_factory=dict)


_FEED_STOPWORDS = {
    "si", "sau", "cu", "din", "de", "la", "pe", "ptr", "pentru",
    "set", "kit", "buc", "buc.", "uc", "um", "u.m", "um.", "pcs",
    "lei", "ron", "tva", "total", "pret", "valoare", "cantitate",
    "manopera", "operatie", "revizie", "generala"
}

_NOISE_REGEX = r"\b(compatibil|potrivit|se potriveste|nu include|fara|universal|set complet|cadou)\b"

def _normalize_for_search(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\.\-_/ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _extract_search_tokens(desc: str, limit: int = 6) -> List[str]:
    s = _normalize_for_search(desc)
    if not s:
        return []

    raw_tokens = s.split()

    tokens: List[str] = []
    for t in raw_tokens:
        if t in _FEED_STOPWORDS:
            continue
        if len(t) <= 2:
            continue
        if re.fullmatch(r"\d{1,2}", t):
            continue
        tokens.append(t)

    tokens = sorted(list(dict.fromkeys(tokens)), key=lambda x: (-len(x), x))
    return tokens[: max(1, min(limit, len(tokens)))]

def _get_bq_client() -> "bigquery.Client":
    if bigquery is None or service_account is None:
        raise HTTPException(status_code=500, detail="BigQuery deps missing. Add google-cloud-bigquery to requirements.")

    cred_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON", "").strip()
    if not cred_json:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_APPLICATION_CREDENTIALS_JSON")

    try:
        info = json.loads(cred_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}")

    creds = service_account.Credentials.from_service_account_info(info)
    project = os.getenv("BQ_PROJECT", "").strip() or info.get("project_id") or ""
    if not project:
        raise HTTPException(status_code=500, detail="Server missing BQ_PROJECT (and no project_id in credentials)")

    location = os.getenv("BQ_LOCATION", "").strip() or None
    client = bigquery.Client(project=project, credentials=creds, location=location)
    return client

def _bq_table_fqn() -> str:
    project = os.getenv("BQ_PROJECT", "").strip()
    dataset = os.getenv("BQ_DATASET", "").strip()
    table = os.getenv("BQ_TABLE", "").strip()
    if not project or not dataset or not table:
        raise HTTPException(status_code=500, detail="Server missing BQ_PROJECT/BQ_DATASET/BQ_TABLE")
    return f"`{project}.{dataset}.{table}`"

def _market_verdict(observed: float, p90: float, source_count: int) -> Tuple[str, float, float]:
    """
    Return (verdict, confidence, ratio_to_p90)
    - verdict: ok / suspect / teapa / unknown
    - confidence: 0..1
    - ratio_to_p90: observed/p90 (0 if not computable)
    """
    obs = float(observed or 0.0)
    p = float(p90 or 0.0)
    if obs <= 0 or p <= 0 or source_count <= 0:
        return ("unknown", 0.0, 0.0)

    ratio = obs / p

    # confidence based on source_count (simple & stable)
    # 0..1 grows with more sources, saturates ~200
    conf = min(1.0, max(0.0, (source_count / 200.0)))

    # verdict thresholds (tweakable later)
    if ratio <= 1.10:
        return ("ok", conf, ratio)
    if ratio <= 1.35:
        return ("suspect", conf, ratio)
    return ("teapa", conf, ratio)

@app.post("/price_lookup", response_model=PriceLookupResponse)
def price_lookup(payload: PriceLookupRequest):
    desc = _norm_spaces(payload.desc or "")
    if not desc:
        return PriceLookupResponse(source_count=0, tokens_used=[], MarketVerdict="unknown", MarketConfidence=0.0, MarketRatioToP90=0.0)

    tokens = _extract_search_tokens(desc, limit=int(payload.token_limit or 6))
    if not tokens:
        return PriceLookupResponse(source_count=0, tokens_used=[], MarketVerdict="unknown", MarketConfidence=0.0, MarketRatioToP90=0.0)

    brand = _normalize_for_search(payload.brand or "")
    min_price = float(payload.min_price or 0.0)
    observed_price = float(payload.observed_price or 0.0)
    want_debug = bool(payload.debug)

    table_fqn = _bq_table_fqn()

    # search in title+description
    search_expr = "LOWER(CONCAT(IFNULL(title,''), ' ', IFNULL(description,'')))"

    where_clauses = ["price > @min_price"]
    params = [bigquery.ScalarQueryParameter("min_price", "FLOAT64", min_price)]

    for i, tok in enumerate(tokens):
        pname = f"t{i}"
        where_clauses.append(f"{search_expr} LIKE CONCAT('%', @{pname}, '%')")
        params.append(bigquery.ScalarQueryParameter(pname, "STRING", tok))

    if brand:
        where_clauses.append("LOWER(brand) LIKE CONCAT('%', @brand, '%')")
        params.append(bigquery.ScalarQueryParameter("brand", "STRING", brand))

    # noise filter always ON (safe)
    where_clauses.append(f"NOT REGEXP_CONTAINS({search_expr}, r'{_NOISE_REGEX}')")

    sql = f"""
    SELECT
      COUNT(*) AS source_count,
      ROUND(MIN(price), 2) AS price_min,
      ROUND(APPROX_QUANTILES(price, 2)[OFFSET(1)], 2) AS price_median,
      ROUND(MAX(price), 2) AS price_max,
      ROUND(APPROX_QUANTILES(price, 100)[OFFSET(90)], 2) AS price_p90
    FROM {table_fqn}
    WHERE {" AND ".join(where_clauses)}
    """

    client = _get_bq_client()
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    try:
        rows = list(client.query(sql, job_config=job_config).result())
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"BigQuery query failed: {e}")

    if not rows:
        verdict, conf, ratio = _market_verdict(observed_price, 0.0, 0)
        return PriceLookupResponse(
            source_count=0,
            price_min=0.0,
            price_median=0.0,
            price_max=0.0,
            price_p90=0.0,
            tokens_used=tokens,
            MarketVerdict=verdict,
            MarketConfidence=conf,
            MarketRatioToP90=ratio,
            debug={"sql": sql} if want_debug else {}
        )

    r0 = rows[0]
    sc = int(r0.get("source_count") or 0)
    pmin = float(r0.get("price_min") or 0.0)
    pmed = float(r0.get("price_median") or 0.0)
    pmax = float(r0.get("price_max") or 0.0)
    pp90 = float(r0.get("price_p90") or 0.0)

    verdict, conf, ratio = _market_verdict(observed_price, pp90, sc)

    dbg = {}
    if want_debug:
        dbg = {
            "table_fqn": table_fqn,
            "sql": sql,
            "tokens_used": tokens,
            "brand_filter": brand or None,
            "min_price": min_price,
            "noise_filter": True,
            "identity": {
                "bq_client_project": getattr(client, "project", None),
                "bq_location_env": os.getenv("BQ_LOCATION", None),
            }
        }

    return PriceLookupResponse(
        source_count=sc,
        price_min=pmin,
        price_median=pmed,
        price_max=pmax,
        price_p90=pp90,
        tokens_used=tokens,
        MarketVerdict=verdict,
        MarketConfidence=float(conf),
        MarketRatioToP90=float(ratio),
        debug=dbg
    )


# =========================
# Routes
# =========================

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/process_deviz_url", response_model=DevizResponse)
def process_deviz_url(payload: DevizUrlPayload):
    image_bytes = _download_file(payload.url)
    return _process_image(image_bytes)

@app.post("/process_deviz_file", response_model=DevizResponse)
async def process_deviz_file(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return _process_image(data)
