import os
import re
import json
import base64
import logging
from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from PIL import Image

# Optional PDF support (recommended)
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# Optional OpenAI (fallback)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ----------------------------
# Config + Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("deviz-ocr")

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o").strip()
FALLBACK_SCORE_THRESHOLD = float(os.getenv("FALLBACK_SCORE_THRESHOLD", "0.65"))

if not GOOGLE_VISION_API_KEY:
    log.warning("GOOGLE_VISION_API_KEY is missing. Primary OCR will fail.")
if not OPENAI_API_KEY:
    log.warning("OPENAI_API_KEY is missing. Fallback will fail.")
if OpenAI is None:
    log.warning("openai package not available. Fallback will fail.")


app = FastAPI(title="Deviz Parser Hybrid: Vision Layout Primary + GPT Fallback", version="0.1.0")


# ----------------------------
# Models
# ----------------------------
class DevizUrlPayload(BaseModel):
    url: str = Field(..., description="Public URL for the deviz file (png/jpg/pdf).")


class ExtractedItem(BaseModel):
    type: str  # "part" | "labor"
    description: str
    qty: float
    unit: Optional[str] = None
    unit_price: float
    line_total: float
    currency: str = "RON"
    warnings: List[str] = []


class ExtractedTotals(BaseModel):
    materials: Optional[float] = None
    labor: Optional[float] = None
    vat: Optional[float] = None
    subtotal_no_vat: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "RON"


class DevizResult(BaseModel):
    ok: bool = True
    source: str  # "google_vision" | "gpt_fallback"
    score: float
    items: List[ExtractedItem]
    totals: ExtractedTotals
    debug: Dict[str, Any] = {}


# ----------------------------
# Helpers: file download + normalize to image bytes
# ----------------------------
def _download_bytes(url: str) -> Tuple[bytes, str]:
    try:
        r = requests.get(url, timeout=60, allow_redirects=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download url: {e}")

    if r.status_code >= 400:
        raise HTTPException(status_code=400, detail=f"Could not download url. HTTP {r.status_code}")

    content_type = (r.headers.get("Content-Type") or "").lower().split(";")[0].strip()
    return r.content, content_type


def _pdf_first_page_to_png_bytes(pdf_bytes: bytes) -> bytes:
    if fitz is None:
        raise HTTPException(
            status_code=400,
            detail="PDF received but PyMuPDF is not installed. Add 'pymupdf' to requirements.txt."
        )
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count < 1:
        raise HTTPException(status_code=400, detail="Empty PDF.")
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=220, alpha=False)  # decent OCR quality
    return pix.tobytes("png")


def _normalize_to_png_bytes(file_bytes: bytes, content_type: str) -> bytes:
    # If PDF -> render first page to PNG
    if "pdf" in content_type or file_bytes[:4] == b"%PDF":
        return _pdf_first_page_to_png_bytes(file_bytes)

    # Otherwise try PIL to convert to PNG
    try:
        img = Image.open(BytesIO(file_bytes))
        img = img.convert("RGB")
        out = BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsupported image format or corrupted file: {e}")


# ----------------------------
# Google Vision: call + layout reconstruction
# ----------------------------
class _TextWord:
    __slots__ = ("text", "x", "y", "h")
    def __init__(self, text: str, x: int, y: int, h: int):
        self.text = text
        self.x = x
        self.y = y
        self.h = h


def _google_vision_document_text_detection(png_bytes: bytes) -> Dict[str, Any]:
    if not GOOGLE_VISION_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing GOOGLE_VISION_API_KEY")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    body = {
        "requests": [{
            "image": {"content": base64.b64encode(png_bytes).decode("utf-8")},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }
    r = requests.post(url, json=body, timeout=90)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Google Vision error: {r.status_code} {r.text}")
    return r.json()


def _reconstruct_lines_from_vision(vision_json: Dict[str, Any]) -> List[str]:
    # Extract words + bounding boxes
    words: List[_TextWord] = []

    try:
        resp0 = (vision_json.get("responses") or [])[0] or {}
        pages = (((resp0.get("fullTextAnnotation") or {}).get("pages")) or [])
        for page in pages:
            for block in page.get("blocks", []) or []:
                for para in block.get("paragraphs", []) or []:
                    for w in para.get("words", []) or []:
                        txt = "".join((s.get("text", "") for s in (w.get("symbols", []) or []))).strip()
                        if not txt:
                            continue
                        verts = (((w.get("boundingBox") or {}).get("vertices")) or [])
                        if len(verts) < 3:
                            continue
                        x = int(verts[0].get("x", 0) or 0)
                        y = int(verts[0].get("y", 0) or 0)
                        y2 = int(verts[2].get("y", y + 10) or (y + 10))
                        h = max(6, y2 - y)
                        words.append(_TextWord(txt, x, y, h))
    except Exception:
        words = []

    if not words:
        # fallback to raw text if any
        resp0 = (vision_json.get("responses") or [])[0] or {}
        raw = (((resp0.get("fullTextAnnotation") or {}).get("text")) or "").strip()
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # Sort by Y then X for clustering
    words.sort(key=lambda w: (w.y, w.x))

    lines: List[List[_TextWord]] = []
    for w in words:
        placed = False
        for line in lines:
            avg_y = sum(x.y for x in line) / len(line)
            avg_h = sum(x.h for x in line) / len(line)
            if abs(w.y - avg_y) <= max(6.0, avg_h * 0.55):
                line.append(w)
                placed = True
                break
        if not placed:
            lines.append([w])

    # For each line: sort by X, then add spaces based on gaps (helps stop "zippering")
    final_lines: List[str] = []
    for line_words in lines:
        line_words.sort(key=lambda w: w.x)
        parts: List[str] = []
        prev_x = None
        for ww in line_words:
            if prev_x is None:
                parts.append(ww.text)
                prev_x = ww.x
                continue
            gap = ww.x - prev_x
            if gap > 55:
                parts.append("   ")
            elif gap > 25:
                parts.append("  ")
            else:
                parts.append(" ")
            parts.append(ww.text)
            prev_x = ww.x
        txt = "".join(parts).strip()
        if txt:
            final_lines.append(txt)

    # A bit of cleanup: remove duplicates that happen sometimes
    cleaned: List[str] = []
    last = None
    for ln in final_lines:
        if ln == last:
            continue
        cleaned.append(ln)
        last = ln
    return cleaned


# ----------------------------
# Parsing: extract items + totals from reconstructed lines
# Works for model 1 (Materiale/Piese + Manopera) and model 2 (Lista materiale + Lista operatiuni)
# ----------------------------
_NUM_RE = re.compile(r"^\d{1,3}([.,]\d{3})*([.,]\d+)?$|^\d+([.,]\d+)?$")

def _to_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    # "1,070.00" / "1.070,00" / "1070.00" / "1070,00"
    # heuristic: if both "," and ".", treat the last separator as decimal.
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # 1.070,00
            s = s.replace(".", "").replace(",", ".")
        else:
            # 1,070.00
            s = s.replace(",", "")
    else:
        # single separator
        if s.count(",") == 1 and s.count(".") == 0:
            s = s.replace(",", ".")
        elif s.count(".") > 1 and s.count(",") == 0:
            s = s.replace(".", "")
        elif s.count(",") > 1 and s.count(".") == 0:
            s = s.replace(",", "")

    try:
        return float(s)
    except Exception:
        return None


def _extract_trailing_numbers(tokens: List[str]) -> Tuple[List[str], List[float]]:
    # Walk from end; collect numeric tokens until hit text
    nums: List[float] = []
    cut = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        t = tokens[i].strip()
        if not t:
            continue
        tt = t.replace("RON", "").strip()
        if _NUM_RE.match(tt):
            v = _to_float(tt)
            if v is None:
                break
            nums.insert(0, v)
            cut = i
        else:
            break
    desc_tokens = tokens[:cut]
    return desc_tokens, nums


def _best_qty_price_total(nums: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Tries to pick qty, unit_price, total from a list of extracted numbers.
    Usually last 3 are qty, price, total, but zippering can reorder.
    Returns best guess + warnings.
    """
    warnings: List[str] = []
    if len(nums) < 2:
        return None, None, None, ["missing_fields"]

    # candidates: pick triples among last up to 5 nums
    tail = nums[-5:] if len(nums) > 5 else nums[:]
    best = None
    best_err = 10**18

    # If exactly 2 nums: assume qty, total OR qty, price; we cannot know total -> treat second as total
    if len(nums) == 2:
        qty, second = nums[0], nums[1]
        # qty should be small-ish; if not, swap
        if qty > 1000 and second <= 1000:
            qty, second = second, qty
        # we treat second as total, and unit_price = total/qty if qty != 0
        unit_price = (second / qty) if qty else second
        return qty, unit_price, second, ["derived_unit_price"]

    # brute force choose 3 values from tail (order matters by meaning)
    # we try mapping (qty, price, total) with constraints
    candidates = []
    for i in range(len(tail)):
        for j in range(len(tail)):
            for k in range(len(tail)):
                if len({i, j, k}) != 3:
                    continue
                q, p, t = tail[i], tail[j], tail[k]
                if q <= 0 or p < 0 or t < 0:
                    continue
                # qty tends to be <= 1000, price <= 1e7, total <= 1e7
                if q > 50000:
                    continue
                err = abs((q * p) - t)
                candidates.append((err, q, p, t))

    if not candidates:
        # fallback: take last three
        q, p, t = nums[-3], nums[-2], nums[-1]
        warnings.append("heuristic_last_three")
        return q, p, t, warnings

    candidates.sort(key=lambda x: x[0])
    err, q, p, t = candidates[0]

    # Accept if err small, else still return but warn
    if err > max(1.0, t * 0.03):
        warnings.append(f"inconsistent_total: qty*unit_price={q*p:.2f} vs total={t:.2f}")
    return q, p, t, warnings


def _detect_section(line_l: str) -> Optional[str]:
    # model 1
    if "manopera" == line_l.strip():
        return "labor"
    if "materiale/piese" in line_l or "materiale" == line_l.strip() or "piese" == line_l.strip():
        return "part"

    # model 2
    if "lista materiale" in line_l:
        return "part"
    if "lista operatiuni" in line_l or "lista opera" in line_l:
        return "labor"

    return None


def _should_skip_noise(line_l: str) -> bool:
    noise = [
        "c.i.f", "nr ord reg com", "adresa:", "judet:", "telefon:",
        "date client", "date produs", "serie de sasiu", "numar inmatriculare",
        "pagina", "garantie", "lucrarile au fost", "in conformitate", "hg ",
        "subtotal", "total materiale", "total manopera", "total deviz", "tva"
    ]
    return any(n in line_l for n in noise)


def _parse_lines_to_items_and_totals(lines: List[str]) -> Tuple[List[ExtractedItem], ExtractedTotals, Dict[str, Any]]:
    items: List[ExtractedItem] = []
    totals = ExtractedTotals(currency="RON")
    debug: Dict[str, Any] = {"lines_used": 0, "section": None, "found_totals": {}}

    current_section: Optional[str] = None

    # Totals patterns (both models)
    re_total_deviz = re.compile(r"\btotal\s+deviz\b", re.IGNORECASE)
    re_total_cu_tva = re.compile(r"\btotal\s+cu\s+tva\b", re.IGNORECASE)
    re_total_materiale = re.compile(r"\btotal\s+materiale\b", re.IGNORECASE)
    re_total_manopera = re.compile(r"\btotal\s+manopera\b", re.IGNORECASE)
    re_subtotal_ftva = re.compile(r"\bsubtotal\b.*\bf\.?tva\b", re.IGNORECASE)
    re_tva = re.compile(r"\btva\b", re.IGNORECASE)

    for raw in lines:
        line = (raw or "").strip()
        if not line:
            continue
        line_l = line.lower().strip()

        sec = _detect_section(line_l)
        if sec:
            current_section = sec
            debug["section"] = current_section
            continue

        # totals lines
        # (some PDFs put numbers on next line; we handle same-line first)
        def grab_last_number(s: str) -> Optional[float]:
            toks = s.replace(":", " ").split()
            _, nums = _extract_trailing_numbers(toks)
            return nums[-1] if nums else None

        if re_total_deviz.search(line_l):
            v = grab_last_number(line)
            if v is not None:
                totals.grand_total = v
                debug["found_totals"]["grand_total"] = v
            continue

        if re_total_cu_tva.search(line_l):
            v = grab_last_number(line)
            if v is not None:
                totals.grand_total = v
                debug["found_totals"]["grand_total"] = v
            continue

        if re_total_materiale.search(line_l):
            v = grab_last_number(line)
            if v is not None:
                totals.materials = v
                debug["found_totals"]["materials"] = v
            continue

        if re_total_manopera.search(line_l):
            v = grab_last_number(line)
            if v is not None:
                totals.labor = v
                debug["found_totals"]["labor"] = v
            continue

        if re_subtotal_ftva.search(line_l):
            v = grab_last_number(line)
            if v is not None:
                totals.subtotal_no_vat = v
                debug["found_totals"]["subtotal_no_vat"] = v
            continue

        # A single TVA line sometimes exists with value
        if re_tva.fullmatch(line_l.strip(" :")) or line_l.startswith("tva"):
            v = grab_last_number(line)
            if v is not None:
                totals.vat = v
                debug["found_totals"]["vat"] = v
            continue

        if _should_skip_noise(line_l):
            continue

        # Parse table row candidates: must have numbers
        tokens = line.replace("\t", " ").split()
        desc_tokens, nums = _extract_trailing_numbers(tokens)
        if len(nums) < 2:
            continue

        desc = " ".join(desc_tokens).strip()
        desc = re.sub(r"^\d+\.?\s*", "", desc)  # remove "1." etc

        # If section not set, guess based on keywords
        typ = current_section
        if not typ:
            if "manopera" in line_l:
                typ = "labor"
            else:
                typ = "part"

        # For labor lines, remove leading "manopera"
        if typ == "labor":
            desc = re.sub(r"^manopera\s+", "", desc, flags=re.IGNORECASE).strip()

        # Guess unit
        unit = None
        # common: BUC, h, ore
        if re.search(r"\b(buc|h|ore)\b", line_l):
            m = re.search(r"\b(buc|h|ore)\b", line_l)
            unit = m.group(1) if m else None
            if unit == "h":
                unit = "ore"
            if unit == "buc":
                unit = "buc"

        qty, unit_price, total, warns = _best_qty_price_total(nums)
        if qty is None or unit_price is None or total is None:
            continue

        # A VERY common zippering in labor: qty and price swapped or extra numbers from header
        # Fix: if qty looks like a price and unit_price looks like qty, swap if it improves consistency
        if typ == "labor":
            # qty should be small (hours)
            if qty > 50 and unit_price <= 50:
                # try swap
                new_qty, new_price = unit_price, qty
                if abs((new_qty * new_price) - total) < abs((qty * unit_price) - total):
                    qty, unit_price = new_qty, new_price
                    warns.append("swapped_qty_price")

            # Another case: extracted total is actually qty (like 3.0) and total is elsewhere
            # If total is tiny and qty*price is big, assume total should be qty*price
            if total <= 10 and (qty * unit_price) >= 50:
                warns.append("fixed_total_as_qty_price")
                total = round(qty * unit_price, 2)

        # If still inconsistent but we can derive total from qty*price and it's close to known totals, fix it
        if any(w.startswith("inconsistent_total") for w in warns):
            derived = round(qty * unit_price, 2)
            # if derived looks plausible, and current total is suspicious
            if derived > 0 and (abs(derived - total) > max(1.0, derived * 0.03)):
                # keep warning, do not overwrite by default
                pass

        if len(desc) < 2:
            continue

        items.append(ExtractedItem(
            type="labor" if typ == "labor" else "part",
            description=desc,
            qty=float(qty),
            unit=unit,
            unit_price=float(unit_price),
            line_total=float(total),
            currency="RON",
            warnings=warns
        ))
        debug["lines_used"] += 1

    # Post totals: if not found, compute from items
    parts_sum = round(sum(i.line_total for i in items if i.type == "part"), 2) if items else 0.0
    labor_sum = round(sum(i.line_total for i in items if i.type == "labor"), 2) if items else 0.0
    if totals.materials is None and parts_sum > 0:
        totals.materials = parts_sum
    if totals.labor is None and labor_sum > 0:
        totals.labor = labor_sum
    if totals.grand_total is None and (parts_sum + labor_sum) > 0:
        totals.grand_total = round(parts_sum + labor_sum, 2)

    return items, totals, debug


def _score_extraction(items: List[ExtractedItem], totals: ExtractedTotals) -> float:
    if not items:
        return 0.0

    # Consistency ratio
    consistent = 0
    for it in items:
        if abs((it.qty * it.unit_price) - it.line_total) <= max(1.0, it.line_total * 0.03):
            consistent += 1
    ratio_consistent = consistent / max(1, len(items))

    # Has both sections?
    has_part = any(i.type == "part" for i in items)
    has_labor = any(i.type == "labor" for i in items)
    ratio_sections = 1.0 if (has_part and has_labor) else 0.6 if (has_part or has_labor) else 0.0

    # Totals plausibility
    sum_items = round(sum(i.line_total for i in items), 2)
    total_score = 0.7
    if totals.grand_total is not None and sum_items > 0:
        diff = abs(totals.grand_total - sum_items)
        total_score = 1.0 if diff <= max(2.0, totals.grand_total * 0.03) else 0.5

    score = 0.55 * ratio_consistent + 0.25 * ratio_sections + 0.20 * total_score
    return round(float(score), 4)


# ----------------------------
# GPT fallback: Vision-to-JSON
# ----------------------------
_FALLBACK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["part", "labor"]},
                    "description": {"type": "string"},
                    "qty": {"type": "number"},
                    "unit": {"type": ["string", "null"]},
                    "unit_price": {"type": "number"},
                    "line_total": {"type": "number"},
                    "currency": {"type": "string"}
                },
                "required": ["type", "description", "qty", "unit_price", "line_total", "currency"]
            }
        },
        "totals": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "materials": {"type": ["number", "null"]},
                "labor": {"type": ["number", "null"]},
                "vat": {"type": ["number", "null"]},
                "subtotal_no_vat": {"type": ["number", "null"]},
                "grand_total": {"type": ["number", "null"]},
                "currency": {"type": "string"}
            },
            "required": ["currency"]
        }
    },
    "required": ["items", "totals"]
}


def _gpt_fallback_parse(png_bytes: bytes) -> Tuple[List[ExtractedItem], ExtractedTotals, Dict[str, Any]]:
    if OpenAI is None:
        raise HTTPException(status_code=500, detail="openai package not installed on server")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    client = OpenAI(api_key=OPENAI_API_KEY)

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    prompt = (
        "Extract structured data from this Romanian auto repair estimate/invoice (deviz). "
        "Return ONLY valid JSON matching the provided schema. "
        "Rules: items must be split into part vs labor. qty, unit_price, line_total must be numeric. "
        "If unit is shown as BUC use 'buc', if shown as 'h' or 'ore' use 'ore'. Currency RON."
    )

    # Using Responses API style
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64}"}
            ]
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "deviz_extract",
                "schema": _FALLBACK_SCHEMA,
                "strict": True
            }
        }
    )

    out_text = resp.output_text
    try:
        data = json.loads(out_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"GPT fallback returned invalid JSON: {e}")

    items = []
    for it in data.get("items", []):
        items.append(ExtractedItem(
            type=it["type"],
            description=it["description"],
            qty=float(it["qty"]),
            unit=it.get("unit"),
            unit_price=float(it["unit_price"]),
            line_total=float(it["line_total"]),
            currency=it.get("currency") or "RON",
            warnings=[]
        ))

    t = data.get("totals") or {}
    totals = ExtractedTotals(
        materials=t.get("materials"),
        labor=t.get("labor"),
        vat=t.get("vat"),
        subtotal_no_vat=t.get("subtotal_no_vat"),
        grand_total=t.get("grand_total"),
        currency=t.get("currency") or "RON"
    )

    return items, totals, {"gpt_model": OPENAI_MODEL}


# ----------------------------
# Orchestrator
# ----------------------------
def _process_png_bytes(png_bytes: bytes) -> DevizResult:
    vision_json = _google_vision_document_text_detection(png_bytes)
    lines = _reconstruct_lines_from_vision(vision_json)

    items, totals, debug = _parse_lines_to_items_and_totals(lines)
    score = _score_extraction(items, totals)

    debug["score_components"] = {
        "items_count": len(items),
        "grand_total": totals.grand_total,
    }
    debug["preview_lines"] = lines[:40]  # for debug, safe to keep some

    if score < FALLBACK_SCORE_THRESHOLD:
        log.info(f"Score {score} below threshold {FALLBACK_SCORE_THRESHOLD}, using GPT fallback.")
        fb_items, fb_totals, fb_debug = _gpt_fallback_parse(png_bytes)
        fb_score = _score_extraction(fb_items, fb_totals)

        return DevizResult(
            ok=True,
            source="gpt_fallback",
            score=fb_score,
            items=fb_items,
            totals=fb_totals,
            debug={"primary_score": score, "threshold": FALLBACK_SCORE_THRESHOLD, **fb_debug}
        )

    return DevizResult(
        ok=True,
        source="google_vision",
        score=score,
        items=items,
        totals=totals,
        debug=debug
    )


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/process_deviz_url", response_model=DevizResult)
def process_deviz_url(payload: DevizUrlPayload):
    file_bytes, content_type = _download_bytes(payload.url)
    png_bytes = _normalize_to_png_bytes(file_bytes, content_type)
    return _process_png_bytes(png_bytes)


@app.post("/process_deviz_file", response_model=DevizResult)
async def process_deviz_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    content_type = (file.content_type or "").lower()
    png_bytes = _normalize_to_png_bytes(file_bytes, content_type)
    return _process_png_bytes(png_bytes)
