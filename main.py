import os
import re
import json
import base64
import requests
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel

app = FastAPI(title="Deviz Parser Hybrid: Vision Layout Primary + GPT Fallback")

GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ----------------------------
# Models
# ----------------------------

class Vehicle(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None
    year_range: Optional[str] = None

class ExtractedItem(BaseModel):
    raw: str
    desc: str
    kind: str  # part / labor / fee / unknown
    qty: Optional[float] = None
    unit: Optional[str] = None
    unit_price: Optional[float] = None
    line_total: Optional[float] = None
    currency: str = "unknown"
    warnings: List[str] = []

class ParseResult(BaseModel):
    source: str  # "google_vision_layout" or "gpt_fallback"
    total_detected: Optional[float] = None
    currency: str = "unknown"
    sum_lines: Optional[float] = None
    warnings: List[str] = []
    items: List[ExtractedItem] = []
    score: Dict[str, Any] = {}
    reconstructed_text_preview: List[str] = []

# ----------------------------
# Helpers: numbers / text
# ----------------------------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)

def detect_currency(text: str) -> str:
    hits = CURRENCY_RE.findall(text or "")
    hits = [h.lower() for h in hits]
    if not hits:
        return "unknown"
    if "eur" in hits or "euro" in hits:
        if "ron" in hits or "lei" in hits:
            return "mixed"
        return "EUR"
    return "RON"

def normalize_num_token(tok: str) -> Optional[float]:
    # handles "1,070.00" or "1070.00" or "1.070,00"
    t = tok.strip()
    if not t:
        return None
    # keep digits and separators only
    if not re.search(r"\d", t):
        return None

    # if both ',' and '.' appear -> assume one is thousands
    if "," in t and "." in t:
        # common in RO OCR: 1,070.00 = 1070.00
        # if comma before dot -> comma thousands
        if t.find(",") < t.find("."):
            t = t.replace(",", "")
            try:
                return float(t)
            except:
                return None
        # else: 1.070,00 -> dot thousands, comma decimals
        t = t.replace(".", "").replace(",", ".")
        try:
            return float(t)
        except:
            return None

    # only comma -> could be decimal
    if "," in t and "." not in t:
        t = t.replace(".", "").replace(",", ".")
        try:
            return float(t)
        except:
            return None

    # only dot
    try:
        return float(t.replace(",", ""))
    except:
        return None

def guess_kind(desc: str) -> str:
    d = (desc or "").lower()
    labor_k = ["manopera", "ore", "ora", "labor", "diagnoza", "diagnostic", "verificare"]
    fee_k = ["taxa", "consumabile", "materiale", "transport", "ecotaxa", "deviz", "service"]
    if any(k in d for k in labor_k):
        return "labor"
    if any(k in d for k in fee_k):
        return "fee"
    return "part" if len(d) > 0 else "unknown"

def is_noise_line(line: str) -> bool:
    l = (line or "").strip().lower()
    if len(l) < 3:
        return True
    # obvious headers/footers
    junk = ["c.i.f", "cif", "beneficiar", "adresa", "telefon", "nr reg", "pagina", "page", "semnatura", "stampila"]
    if any(j in l for j in junk):
        return True
    # table headers
    table_hdr = ["denumire", "u/m", "cantitate", "pret", "valoare", "red", "tva", "subtotal", "total materiale", "total manopera"]
    if any(h == l or l.startswith(h) for h in table_hdr):
        return True
    return False

# ----------------------------
# Google Vision call + layout reconstruction
# ----------------------------

class TextWord:
    def __init__(self, text: str, x: int, y: int, h: int):
        self.text = text
        self.x = x
        self.y = y
        self.h = h

def call_google_vision_text_detection(image_b64: str) -> Dict[str, Any]:
    if not GOOGLE_VISION_API_KEY:
        raise RuntimeError("Missing GOOGLE_VISION_API_KEY")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={GOOGLE_VISION_API_KEY}"
    body = {
        "requests": [{
            "image": {"content": image_b64},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }
    r = requests.post(url, json=body, timeout=60)
    r.raise_for_status()
    return r.json()

def reconstruct_lines_from_google_vision(vision_json: Dict[str, Any]) -> List[str]:
    words: List[TextWord] = []
    try:
        pages = vision_json["responses"][0]["fullTextAnnotation"]["pages"]
        for page in pages:
            for block in page.get("blocks", []):
                for paragraph in block.get("paragraphs", []):
                    for w in paragraph.get("words", []):
                        txt = "".join(s.get("text", "") for s in w.get("symbols", []))
                        verts = w.get("boundingBox", {}).get("vertices", [])
                        if not verts:
                            continue
                        x = int(verts[0].get("x", 0))
                        y = int(verts[0].get("y", 0))
                        yb = int(verts[2].get("y", y + 10)) if len(verts) > 2 else y + 10
                        h = max(1, yb - y)
                        words.append(TextWord(txt, x, y, h))
    except Exception:
        # fallback raw text
        raw = vision_json.get("responses", [{}])[0].get("fullTextAnnotation", {}).get("text", "")
        return raw.split("\n") if raw else []

    if not words:
        raw = vision_json.get("responses", [{}])[0].get("fullTextAnnotation", {}).get("text", "")
        return raw.split("\n") if raw else []

    words.sort(key=lambda w: w.y)

    lines: List[List[TextWord]] = []
    for word in words:
        placed = False
        for ln in lines:
            avg_y = sum(w.y for w in ln) / len(ln)
            avg_h = sum(w.h for w in ln) / len(ln)
            if abs(word.y - avg_y) < (avg_h * 0.6):
                ln.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    out = []
    for ln in lines:
        ln.sort(key=lambda w: w.x)
        out.append(" ".join(w.text for w in ln).strip())
    return [x for x in out if x]

# ----------------------------
# Parsing reconstructed lines into items
# ----------------------------

UNIT_MAP = {
    "buc": "buc", "buc.": "buc", "pcs": "buc",
    "h": "ore", "ore": "ore", "ora": "ore"
}

def parse_line_as_item(line: str, default_currency: str) -> Optional[ExtractedItem]:
    raw = (line or "").strip()
    if not raw or is_noise_line(raw):
        return None

    l = raw.strip()
    # collect numeric tokens from end
    parts = l.split()
    nums: List[float] = []
    # how many tokens at end are numeric
    consumed = 0
    for tok in reversed(parts):
        v = normalize_num_token(tok)
        if v is None:
            break
        nums.insert(0, v)
        consumed += 1

    # if no numbers => skip
    if len(nums) == 0:
        # still allow labor lines (sometimes numbers are on next line) -> skip here
        return None

    desc_tokens = parts[:-consumed] if consumed > 0 else parts[:]
    desc = " ".join(desc_tokens).strip()

    # detect unit token inside desc (like "BUC", "h")
    unit = None
    for t in parts:
        tt = t.lower()
        if tt in UNIT_MAP:
            unit = UNIT_MAP[tt]
            break

    kind = guess_kind(desc)

    qty = unit_price = line_total = None
    warnings: List[str] = []

    # best case: 3 numbers = qty, price, total (or qty, price, total)
    if len(nums) >= 3:
        # try last 3 as qty, unit_price, total
        q, p, tot = nums[-3], nums[-2], nums[-1]
        # validate
        if q > 0 and p > 0 and tot > 0:
            qty, unit_price, line_total = q, p, tot
        else:
            warnings.append("bad_numbers")
    elif len(nums) == 2:
        # sometimes row has qty and total only (no unit price)
        # pick qty=first, total=second
        q, tot = nums[0], nums[1]
        if q > 0 and tot > 0:
            qty, line_total = q, tot
            warnings.append("missing_unit_price")
        else:
            warnings.append("bad_numbers")
    else:
        # 1 number only -> likely noise (dates, ids)
        return None

    # currency
    currency = default_currency
    if CURRENCY_RE.search(l):
        currency = detect_currency(l)
        if currency == "mixed":
            currency = default_currency

    # sanity check if all 3 exist
    if qty is not None and unit_price is not None and line_total is not None:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")

    # filter out ridiculous totals from IDs etc
    if line_total is not None and line_total > 1_000_000:
        return None

    if len(desc) < 3:
        return None

    return ExtractedItem(
        raw=raw,
        desc=desc.lower(),
        kind=kind,
        qty=qty,
        unit=unit,
        unit_price=unit_price,
        line_total=line_total,
        currency=currency,
        warnings=warnings
    )

def extract_total_from_lines(lines: List[str]) -> Optional[float]:
    # prefer "TOTAL DEVIZ" or "Total cu TVA" etc
    priority = [
        re.compile(r"\btotal\s+deviz\b", re.IGNORECASE),
        re.compile(r"\btotal\s+cu\s+tva\b", re.IGNORECASE),
        re.compile(r"\btotal\b", re.IGNORECASE),
    ]
    candidates: List[Tuple[int, float]] = []
    for i, ln in enumerate(lines):
        nums = [normalize_num_token(t) for t in ln.split()]
        nums = [n for n in nums if n is not None]
        if not nums:
            continue
        for pr, rx in enumerate(priority):
            if rx.search(ln):
                candidates.append((pr, nums[-1]))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def score_quality(items: List[ExtractedItem], total: Optional[float]) -> Dict[str, Any]:
    valid3 = [it for it in items if it.qty is not None and it.unit_price is not None and it.line_total is not None]
    sum_lines = sum(it.line_total for it in valid3) if valid3 else 0.0
    mismatch = None
    if total is not None and sum_lines > 0:
        mismatch = abs(total - sum_lines)
    # simple score 0..1
    score = 0.0
    if total is not None:
        score += 0.35
    score += min(0.45, 0.15 * len(valid3))  # more items => higher
    if mismatch is not None:
        if mismatch <= max(2.0, 0.05 * total):
            score += 0.20
    return {
        "valid_items_with_qty_price_total": len(valid3),
        "sum_lines_valid": sum_lines if valid3 else None,
        "total": total,
        "mismatch_abs": mismatch,
        "score_0_1": round(min(1.0, score), 3),
    }

# ----------------------------
# GPT fallback (Vision LLM)
# ----------------------------

def call_gpt_vision_extract(image_b64: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # schema output we want
    system = "You extract structured data from Romanian car repair estimates (devize). Output strict JSON."
    user = {
        "task": "Extract line items (parts + labor) and the final total from this deviz image. Return JSON.",
        "rules": [
            "For each item return: desc, kind(part|labor|fee|unknown), qty, unit, unit_price, line_total, currency.",
            "If unit is 'h' treat as 'ore'.",
            "Prefer 'TOTAL DEVIZ' or 'Total cu TVA' as total.",
            "Do not hallucinate values; if missing set null."
        ]
    }

    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "text", "text": json.dumps(user, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=90
    )
    r.raise_for_status()
    data = r.json()
    txt = data["choices"][0]["message"]["content"]
    return json.loads(txt)

# ----------------------------
# Endpoint: upload file, primary vision layout, fallback to GPT if weak
# ----------------------------

@app.post("/process_deviz_file")
async def process_deviz_file(
    file: UploadFile = File(...),
    vin: str = Form(""),
    brand: str = Form(""),
    model: str = Form(""),
    engine: str = Form(""),
    year_range: str = Form(""),
):
    content = await file.read()
    image_b64 = base64.b64encode(content).decode("utf-8")

    # 1) Google Vision + layout reconstruction
    vision_json = call_google_vision_text_detection(image_b64)
    lines = reconstruct_lines_from_google_vision(vision_json)

    default_currency = detect_currency("\n".join(lines))
    total = extract_total_from_lines(lines)

    items: List[ExtractedItem] = []
    for ln in lines:
        it = parse_line_as_item(ln, default_currency)
        if it:
            items.append(it)

    score = score_quality(items, total)

    # 2) decide fallback
    need_fallback = False
    if score["score_0_1"] < 0.55:
        need_fallback = True
    if score["valid_items_with_qty_price_total"] < 2:
        need_fallback = True
    if total is None:
        need_fallback = True

    warnings: List[str] = []
    if need_fallback:
        warnings.append("low_confidence_primary_layout_parser -> gpt_fallback")

        gpt = call_gpt_vision_extract(image_b64)

        # expect gpt keys: total_detected, currency, items[]
        gpt_total = gpt.get("total_detected") or gpt.get("total") or None
        gpt_currency = gpt.get("currency") or default_currency or "unknown"
        gpt_items = []
        for x in gpt.get("items", []):
            gpt_items.append(ExtractedItem(
                raw=x.get("raw") or x.get("desc") or "",
                desc=(x.get("desc") or "").lower(),
                kind=x.get("kind") or "unknown",
                qty=x.get("qty"),
                unit=x.get("unit"),
                unit_price=x.get("unit_price"),
                line_total=x.get("line_total"),
                currency=x.get("currency") or gpt_currency,
                warnings=x.get("warnings") or [],
            ))

        # recompute sum_lines
        valid3 = [it for it in gpt_items if it.qty is not None and it.unit_price is not None and it.line_total is not None]
        sum_lines = sum(it.line_total for it in valid3) if valid3 else None

        return ParseResult(
            source="gpt_fallback",
            total_detected=gpt_total,
            currency=gpt_currency,
            sum_lines=sum_lines,
            warnings=warnings,
            items=gpt_items,
            score={"primary_score": score, "fallback_used": True},
            reconstructed_text_preview=lines[:8],
        )

    # primary accepted
    valid3 = [it for it in items if it.qty is not None and it.unit_price is not None and it.line_total is not None]
    sum_lines = sum(it.line_total for it in valid3) if valid3 else None

    return ParseResult(
        source="google_vision_layout",
        total_detected=total,
        currency=default_currency,
        sum_lines=sum_lines,
        warnings=warnings,
        items=items,
        score=score,
        reconstructed_text_preview=lines[:8],
    )

from pydantic import BaseModel
import requests

class DevizUrlPayload(BaseModel):
    url: str

@app.post("/process_deviz_url")
def process_deviz_url(payload: DevizUrlPayload):
    # descarca fisierul din Airtable (link-ul din attachment)
    r = requests.get(payload.url, timeout=60)
    if r.status_code != 200:
        return {"error": "failed_to_download", "status_code": r.status_code, "text": r.text[:500]}

    content = r.content

    # IMPORTANT:
    # Aici trebuie sa chemi aceeasi logica pe care o folosesti deja in /process_deviz_file,
    # doar ca ii dai bytes in loc de UploadFile.
    #
    # Daca ai deja o functie care face procesarea (recomandat), cheam-o aici.
    # Exemplu:
    # return process_image_bytes(content, filename="deviz.png")

    # Daca momentan nu ai functie separata, spune-mi cum se numeste endpoint-ul tau intern
    # (ce face /process_deviz_file) si iti dau exact refactor-ul minim.

    return {"ok": True, "bytes": len(content)}
    
@app.get("/health")
def health():
    return {"ok": True}
