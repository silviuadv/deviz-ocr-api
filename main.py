import os
import re
import json
import base64
import math
import requests
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(
    title="Deviz Parser Hybrid: Vision Layout Primary + GPT Fallback",
    version="0.1.0"
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
    part_code: Optional[str] = None  # <--- NEW: cod piesa (daca exista)
    warnings: List[str] = Field(default_factory=list)


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
            # thousands grouping: 1,234 or 12,345,678
            if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
                s = s.replace(",", "")
                return float(s)
            # decimal comma: 123,45
            if re.fullmatch(r"\d+,\d{1,4}", s):
                s = s.replace(",", ".")
                return float(s)
            # fallback: strip commas
            s = s.replace(",", "")
            return float(s)

        # Case C: only dot (normal decimal)
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
# Part code extraction (NEW)
# - prevents false positives like "1 2 3 4 5"
# =========================

# candidate tokens like: 5NU0QVAHL, 365.372, ABC-1234, etc.
_CODE_CANDIDATE_RE = re.compile(r"\b[A-Z0-9][A-Z0-9\.\-\/]{3,}\b", re.IGNORECASE)

def _is_junk_code(tok: str) -> bool:
    t = (tok or "").strip().strip(":").strip()
    if not t:
        return True

    # lista/index: "1", "2", "3", "1.", "2."
    if re.fullmatch(r"\d{1,2}\.?", t):
        return True

    # ani / numere simple gen 2019
    if re.fullmatch(r"\d{4}", t):
        return True

    # numere mari dar doar cifre -> de obicei nu e cod piesa
    if re.fullmatch(r"\d{3,}", t):
        return True

    # prea scurt
    if len(t) < 5:
        return True

    # acceptam explicit coduri gen "365.372"
    if re.fullmatch(r"\d{3}\.\d{3}", t):
        return False

    # in rest: vrem cel putin o litera si o cifra
    has_alpha = any(ch.isalpha() for ch in t)
    has_digit = any(ch.isdigit() for ch in t)
    if not (has_alpha and has_digit):
        return True

    return False


def _extract_part_code_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = _CODE_CANDIDATE_RE.findall(text)
    for c in candidates:
        c = c.strip().strip(":")
        if not _is_junk_code(c):
            return c
    return None


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
        "pagina", "total deviz", "total materiale", "total manopera", "lucrarile",
        "perioada", "certificat", "garantie", "factura", "c.i.f", "nr ord", "reg com"
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

        part_code = None
        if kind == "part":
            part_code = _extract_part_code_from_text(l) or _extract_part_code_from_text(desc)

        items.append(ExtractedItem(
            raw=l,
            desc=desc,
            kind=kind,
            qty=float(qty or 0.0),
            unit=unit,
            unit_price=float(unit_price or 0.0),
            line_total=float(line_total or 0.0),
            currency=currency,
            part_code=part_code,
            warnings=[]
        ))

    return items


# =========================
# Parser B (column/table aware) - for Deviz 3 style
# =========================

def _detect_autodeviz_table(lines: List[str]) -> bool:
    joined = "\n".join(lines[:200]).lower()
    return (
        ("operatie" in joined and "material" in joined and "timp" in joined and "u.m" in joined)
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

        code = tokens[0] if starts_like_code else ""
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
            part_code=None,
            warnings=labor_warnings
        ))

        if mat_desc and qty is not None and unit_price is not None and material_val is not None:
            part_warnings: List[str] = []
            if abs((qty * unit_price) - material_val) > 2.0:
                part_warnings.append("derived_unit_price_or_total_mismatch")

            part_code = _extract_part_code_from_text(text) or _extract_part_code_from_text(mat_desc)

            items.append(ExtractedItem(
                raw=text,
                desc=mat_desc,
                kind="part",
                qty=float(qty or 0.0),
                unit=um or "",
                unit_price=float(unit_price or 0.0),
                line_total=float(material_val or 0.0),
                currency=currency,
                part_code=part_code,
                warnings=part_warnings
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
    if "eur" in low or "euro" in low:
        cur = "EUR"
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
                        "part_code": {"type": ["string", "null"]},
                        "warnings": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["desc", "kind", "qty", "unit", "unit_price", "line_total", "currency", "part_code", "warnings"]
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
        "- pentru piese: daca exista un cod piesa (alfa-numeric), pune-l in part_code, altfel null\n"
        "- totals: materials, labor, grand_total, currency\n"
        "- document: service_name, client_name, vehicle, date, number daca se vad\n"
        "Reguli:\n"
        "- Nu inventa valori. Daca lipseste ceva, pune null sau 0 unde e numeric.\n"
        "- Pastreaza cantitatile si totalurile exact cum apar.\n"
        "- Daca un rand contine si operatie si material (tabel combinat), creeaza 2 items.\n"
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

    resp = DevizResponse(
        document=DocumentMeta(**(parsed.get("document", {}) or {"default_currency": "RON"})),
        totals=Totals(**(parsed.get("totals", {}) or {"currency": "RON"})),
        items=[ExtractedItem(**it) for it in (parsed.get("items") or [])],
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


if __name__ == "__main__":
    print("Run with: uvicorn main:app --host 0.0.0.0 --port 8000")
