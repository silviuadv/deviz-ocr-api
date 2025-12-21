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
    version="0.2.0"
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

    # part code (string gol daca nu e gasit / nu e valid)
    code: str = ""


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
            # if looks like thousands grouping: 1,234 or 12,345,678
            if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
                s = s.replace(",", "")
                return float(s)
            # else assume comma is decimal separator: 123,45
            if re.fullmatch(r"\d+,\d{1,4}", s):
                s = s.replace(",", ".")
                return float(s)
            # fallback: strip commas
            s = s.replace(",", "")
            return float(s)

        # Case C: only dot (normal decimal): 3.00 / 90.00
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
# CODE (part code) extraction + validation
# =========================

STOP_CODES = {
    "h", "ore", "ora", "buc", "buc.", "pcs", "um", "u/m",
    "cantitate", "pret", "valoare", "total",
    "materiale", "manopera", "operatie", "cod", "nr", "nr."
}


def _clean_code_candidate(s: str) -> str:
    s = (s or "").strip().strip('"').strip()
    s = re.sub(r"\s+", "", s)  # codurile de obicei n-au spatii
    return s


def _is_row_index_code(s: str) -> bool:
    # "1" "2" "3" "4" "5" "1." "01" "001" etc
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

    # accept alfanumeric (min 4) care contine macar o litera si o cifra
    if len(c) >= 4 and re.search(r"[A-Za-z]", c) and re.search(r"\d", c):
        return True

    # accept coduri cu separatori, destul de lungi: 62-7304-46037, 5NU0R4JHI, etc
    if len(c) >= 6 and re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z\.\-_/]+", c) and re.search(r"[.\-_/]", c):
        return True

    # accept numeric cu punct (ex: 365.372) ca posibil cod de piesa/serie
    if re.fullmatch(r"\d{2,6}(\.\d{2,6}){1,3}", c):
        return True

    return False


def _extract_code_from_text(raw_line: str, desc: str) -> Tuple[str, str]:
    """
    Intoarce (code, desc_without_code_if_removed).
    Codul e extras conservator, doar daca e valid.
    Daca nu, code = "" si desc ramane neschimbat.
    """
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
# Google Vision: OCR + layout (words with coords)
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
# Parser A (generic lines) - good for Deviz 1/2
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
        "pagina", "total deviz", "total materiale", "total manopera",
        "lucrarile", "perioada", "certificat", "garantie"
    ]

    for line in lines:
        l = _norm_spaces(line)
        if len(l) < 4:
            continue
        if _contains_any(l, noise):
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
# Parser B (table aware) - for Deviz 3 / AutoDeviz style
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
            idx_mv, tok_mv = tail_nums[-1]
            material_val = tok_float(tok_mv)
        if len(tail_nums) >= 2:
            idx_up, tok_up = tail_nums[-2]
            unit_price = tok_float(tok_up)
        if len(tail_nums) >= 3:
            idx_q, tok_q = tail_nums[-3]
            qty = tok_float(tok_q)

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

        # labor item
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

        # material item
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
# Totals extraction from raw text
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
        "- items: fiecare linie din manopera si piese (kind = 'labor' sau 'part')\n"
        "- totals: materials, labor, grand_total, currency (RON)\n"
        "- document: service_name, client_name, vehicle, date, number daca se vad\n"
        "Reguli:\n"
        "- Nu inventa valori. Daca lipseste ceva, pune null sau 0 unde e numeric.\n"
        "- Daca un rand contine si operatie si material, creeaza 2 items: unul labor si unul part.\n"
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
# Download helper for URL endpoint
# =========================

def _download_file(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download file: {r.status_code}")
    return r.content


# =========================
# BigQuery price lookup (B: search title + description always)
# =========================

class PriceLookupRequest(BaseModel):
    desc: str
    brand: Optional[str] = None
    min_price: float = 0.0
    token_limit: int = 6

    # debug payload (optional)
    debug: bool = False

    # always search title+description
    use_description: bool = True

    # remove obvious marketing / compat noise
    noise_filter: bool = True


class PriceLookupResponse(BaseModel):
    source_count: int = 0
    price_min: Optional[float] = None
    price_median: Optional[float] = None
    price_max: Optional[float] = None
    tokens_used: List[str] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


_FEED_STOPWORDS = {
    "si", "sau", "cu", "din", "de", "la", "pe", "ptr", "pentru",
    "set", "kit", "buc", "buc.", "uc", "um", "u.m", "um.", "pcs",
    "lei", "ron", "tva", "total", "pret", "valoare", "cantitate",
    "manopera", "operatie", "revizie", "generala",

    # extra noise words
    "compatibil", "potrivit", "include", "fara", "nu"
}


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
        # exclude pure numbers that are too short (like 1 2 3)
        if re.fullmatch(r"\d{1,2}", t):
            continue
        tokens.append(t)

    # prefer longer tokens first (reduces wild matches like "ulei")
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

    return bigquery.Client(project=project, credentials=creds)


def _bq_table_fqn() -> str:
    project = os.getenv("BQ_PROJECT", "").strip()
    dataset = os.getenv("BQ_DATASET", "").strip()
    table = os.getenv("BQ_TABLE", "").strip()
    if not project or not dataset or not table:
        raise HTTPException(status_code=500, detail="Server missing BQ_PROJECT/BQ_DATASET/BQ_TABLE")
    return f"`{project}.{dataset}.{table}`"


@app.post("/price_lookup", response_model=PriceLookupResponse)
def price_lookup(payload: PriceLookupRequest):
    desc = _norm_spaces(payload.desc or "")
    if not desc:
        return PriceLookupResponse(source_count=0, tokens_used=[], debug={"reason": "empty_desc"} if payload.debug else {})

    tokens = _extract_search_tokens(desc, limit=int(payload.token_limit or 6))
    if not tokens:
        return PriceLookupResponse(source_count=0, tokens_used=[], debug={"reason": "no_tokens"} if payload.debug else {})

    brand = _normalize_for_search(payload.brand or "")
    min_price = float(payload.min_price or 0.0)

    table_fqn = _bq_table_fqn()

    # B: search title + description always (safe)
    text_expr = "LOWER(CONCAT(IFNULL(title,''), ' ', IFNULL(description,'')))"

    where_clauses = ["price > @min_price"]
    params = [
        bigquery.ScalarQueryParameter("min_price", "FLOAT64", min_price),
    ]

    # enforce ALL tokens
    for i, tok in enumerate(tokens):
        pname = f"t{i}"
        where_clauses.append(f"{text_expr} LIKE CONCAT('%', @{pname}, '%')")
        params.append(bigquery.ScalarQueryParameter(pname, "STRING", tok))

    if brand:
        where_clauses.append("LOWER(brand) LIKE CONCAT('%', @brand, '%')")
        params.append(bigquery.ScalarQueryParameter("brand", "STRING", brand))

    # anti-noise filter (optional but recommended)
    if payload.noise_filter:
        where_clauses.append(
            f"NOT REGEXP_CONTAINS({text_expr}, r'\\b(compatibil|potrivit|se potriveste|nu include|fara|universal|set complet|cadou)\\b')"
        )

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
        dbg = {"table_fqn": table_fqn, "sql": sql, "tokens_used": tokens} if payload.debug else {}
        return PriceLookupResponse(source_count=0, tokens_used=tokens, debug=dbg)

    r0 = rows[0]

    dbg: Dict[str, Any] = {}
    if payload.debug:
        dbg = {
            "table_fqn": table_fqn,
            "sql": sql,
            "tokens_used": tokens,
            "brand_filter": brand or None,
            "min_price": min_price,
            "noise_filter": payload.noise_filter,
            "price_p90": float(r0.get("price_p90")) if r0.get("price_p90") is not None else None,
            "identity": {
                "bq_client_project": getattr(client, "project", None),
                "bq_location_env": os.getenv("BQ_LOCATION"),
            }
        }

    return PriceLookupResponse(
        source_count=int(r0.get("source_count") or 0),
        price_min=float(r0.get("price_min")) if r0.get("price_min") is not None else None,
        price_median=float(r0.get("price_median")) if r0.get("price_median") is not None else None,
        price_max=float(r0.get("price_max")) if r0.get("price_max") is not None else None,
        tokens_used=tokens,
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
