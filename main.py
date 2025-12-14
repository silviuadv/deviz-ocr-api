import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Normalizer")

# ---------------- Models ----------------

class Vehicle(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None
    year_range: Optional[str] = None

class InputPayload(BaseModel):
    vin: str
    vehicle: Optional[Vehicle] = None
    ocr_text: str

# ---------------- Helpers ----------------

def clean_text(s: str) -> str:
    s = s.replace("\t", " ")
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()

def normalize_text(s: str) -> str:
    s = clean_text(s).lower()

    # 2,142.00 -> 2142.00 (comma as thousands separator)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # 1,50 -> 1.50 (comma as decimal separator)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)

    return s

def normalize_line(s: str) -> str:
    return normalize_text(s)

# ---------------- Regex ----------------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

# up to 4 decimals (1000.0000)
NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,4})?)(?!\w)")

TOTAL_LINE_RE = re.compile(r"\b(total\s*(de\s*plata|general|cu\s*tva)?|de\s*plata)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
HG_RE = re.compile(r"\bhg\s*\d+\s*/\s*\d+\b", re.IGNORECASE)

ID_TAG_RE = re.compile(r"\b(id|nr|numar|serie)\s*[:#]", re.IGNORECASE)
PART_CODE_RE = re.compile(r"^[a-z0-9]{6,}$", re.IGNORECASE)
NUM_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d{1,4})?\s*$")

# ---------------- Detection ----------------

def detect_currency(text: str) -> str:
    hits = CURRENCY_RE.findall(text or "")
    if not hits:
        return "unknown"
    hits = [h.lower() for h in hits]
    if "eur" in hits or "euro" in hits:
        if "ron" in hits or "lei" in hits:
            return "mixed"
        return "EUR"
    return "RON"

def guess_kind(desc: str) -> str:
    desc = desc or ""
    labor_k = ["manopera", "ore", "ora", "labor", "diagnoza", "diagnostic", "verificare", "reparatie", "montare", "caroserie"]
    fee_k = ["taxa", "consumabile", "materiale", "transport", "ecotaxa", "service"]
    part_k = [
        "filtru", "ulei", "placute", "disc", "kit", "buj", "piesa", "garnitura", "curea",
        "pompa", "senzor", "acumulator", "agrafe", "accesorii", "frana"
    ]
    if any(k in desc for k in labor_k):
        return "labor"
    if any(k in desc for k in fee_k):
        return "fee"
    if any(k in desc for k in part_k):
        return "part"
    return "unknown"

def is_noise_line(l: str) -> bool:
    # keep numeric-only lines (they are qty/prices)
    if NUM_ONLY_RE.match(l or ""):
        return False

    l = l or ""
    if l.startswith("manopera "):
        return False

    # hard footer noise
    footer_noise = [
        "generat cu", "autodeviz", "produs al", "vega", "web", "www", "http", "https",
        "pagina", "page", "semnatura", "stampila",
        "in conformitate cu", "unitatea noastra garanteaza",
    ]
    if any(k in l for k in footer_noise):
        return True
    if HG_RE.search(l):
        return True

    # exact table headers
    header_exact = {
        "materiale/piese",
        "denumire piese si materiale utilizate",
        "denumirea serviciilor prestate",
        "u/m", "cantitate", "pret lista", "pret deviz", "valoare", "red", "tva",
        "beneficiar", "comanda", "observatii", "descriere obiect",
        "km.bord", "km bord", "km. bord", "km bord:",
        "pret manopera", "fara tva", "%"
    }
    if l.strip() in header_exact:
        return True

    # dates alone
    if DATE_RE.search(l) and not any(x in l for x in ["ron", "lei", "eur", "euro", "buc", "ore", "ora", "h"]):
        return True

    if len(l.strip()) < 2:
        return True

    return False

def looks_like_item_start(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False

    if line.startswith("manopera "):
        return True

    parts = line.split()
    if len(parts) < 2:
        return False

    first = parts[0].strip(" -:;")
    if PART_CODE_RE.match(first) and len(first) >= 6 and first not in {"total", "subtotal"}:
        return True

    return False

def is_boundary_line(nln: str) -> bool:
    nln = nln or ""
    if "subtotal" in nln:
        return True
    if nln.startswith("id:"):
        return True
    if ID_TAG_RE.search(nln):
        return True
    return False

# ---------------- Parsing ----------------

def pick_qty_from_numeric_lines(block_lines: List[str]) -> Optional[float]:
    # prefer first numeric-only line that looks like qty (0.001..100)
    for ln in block_lines:
        n = normalize_line(ln)
        if NUM_ONLY_RE.match(n):
            try:
                v = float(n)
            except Exception:
                continue
            if 0.001 <= v <= 100:
                return v
    return None

def parse_block(block_lines: List[str], default_currency: str) -> Optional[Dict[str, Any]]:
    if not block_lines:
        return None

    raw_block = "\n".join(clean_text(x) for x in block_lines).strip()
    if not raw_block:
        return None

    # currency
    currency = default_currency
    if CURRENCY_RE.search(raw_block):
        c = detect_currency(raw_block)
        if c != "mixed":
            currency = c

    # first line -> desc
    first_line_norm = normalize_line(block_lines[0])
    desc = first_line_norm
    desc = CURRENCY_RE.sub(" ", desc)
    desc = UNIT_RE.sub(" ", desc)
    desc = NUM_RE.sub(" ", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" -;:/")

    kind = guess_kind(desc)

    # unit
    unit = None
    m_unit = UNIT_RE.search(normalize_text(raw_block))
    if m_unit:
        unit = m_unit.group(1).lower().replace(".", "")
        if unit == "ora":
            unit = "ore"

    # nums from lines AFTER first line (avoid "set 12" poisoning qty)
    rest_text = normalize_text("\n".join(block_lines[1:])) if len(block_lines) > 1 else ""
    nums = []
    if rest_text:
        try:
            nums = [float(x) for x in NUM_RE.findall(rest_text)]
        except Exception:
            nums = []

    # qty
    qty = None
    # if explicit "X buc/ore" exists anywhere, use it
    if unit:
        m_qty = re.search(rf"\b(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b", normalize_text(raw_block))
        if m_qty:
            try:
                qty = float(m_qty.group(1))
            except Exception:
                qty = None

    if qty is None:
        qty = pick_qty_from_numeric_lines(block_lines)

    unit_price = None
    line_total = None

    # special: labor lines often have 3 numbers: qty, unit_price, total
    if kind == "labor":
        # keep only plausible labor numbers
        labor_nums = [n for n in nums if 0 < n <= 100000]
        if len(labor_nums) >= 3:
            # use first 3 in order
            qty = qty if qty is not None else labor_nums[0]
            unit_price = labor_nums[1]
            line_total = labor_nums[2]
        else:
            # fallback to last two if present
            if len(labor_nums) >= 1:
                line_total = labor_nums[-1]
            if len(labor_nums) >= 2:
                unit_price = labor_nums[-2]

    else:
        # parts/fees: last two numbers are typically unit_price and total (in the block)
        if len(nums) >= 1:
            line_total = nums[-1]
        if len(nums) >= 2:
            unit_price = nums[-2]

    # sanity: absurd totals per line
    if isinstance(line_total, float) and line_total > 100000:
        return None

    warnings: List[str] = []
    confidence = {"qty": 0.0, "unit": 0.0, "unit_price": 0.0, "line_total": 0.0}

    if qty is not None:
        confidence["qty"] = 0.7
    if unit is not None:
        confidence["unit"] = 0.7
    if unit_price is not None:
        confidence["unit_price"] = 0.6
    if line_total is not None:
        confidence["line_total"] = 0.7

    if qty is not None and unit_price is not None and line_total is not None and qty != 0:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")
    else:
        warnings.append("missing_fields")

    if kind == "unknown" and qty is None and unit is None and unit_price is None and line_total is None:
        return None

    return {
        "raw": raw_block,
        "desc": desc,
        "kind": kind,
        "qty": qty,
        "unit": unit,
        "unit_price": unit_price,
        "line_total": line_total,
        "currency": currency,
        "confidence_fields": confidence,
        "warnings": warnings,
    }

def extract_total(text: str, currency_default: str) -> Optional[Dict[str, Any]]:
    lines = [normalize_line(x) for x in (text or "").split("\n") if x is not None]

    # 1) find total line and number on same or next lines
    for i, ln in enumerate(lines):
        if TOTAL_LINE_RE.search(ln):
            nums = [float(x) for x in NUM_RE.findall(ln)]
            if nums:
                return {"total": nums[-1], "currency": currency_default}

            for j in range(1, 4):
                if i + j < len(lines):
                    nxt = lines[i + j]
                    nums2 = [float(x) for x in NUM_RE.findall(nxt)]
                    if nums2:
                        return {"total": nums2[-1], "currency": currency_default}

    # 2) fallback max in tail excluding ID-like
    tail = lines[-160:] if len(lines) > 160 else lines
    nums_tail: List[float] = []
    for ln in tail:
        if DATE_RE.search(ln):
            continue
        if HG_RE.search(ln):
            continue
        if "km.bord" in ln or "km bord" in ln:
            continue
        if ID_TAG_RE.search(ln) or ln.startswith("id:"):
            continue

        for x in NUM_RE.findall(ln):
            try:
                n = float(x)
            except Exception:
                continue
            if 1.0 <= n <= 100000:
                nums_tail.append(n)

    if not nums_tail:
        return None
    return {"total": max(nums_tail), "currency": currency_default}

# ---------------- API ----------------

@app.post("/parse")
def parse(payload: InputPayload):
    default_currency = detect_currency(payload.ocr_text)

    raw_lines = (payload.ocr_text or "").split("\n")

    cleaned_lines: List[str] = []
    for ln in raw_lines:
        nln = normalize_line(ln)
        if not nln:
            continue
        if is_noise_line(nln):
            continue
        cleaned_lines.append(ln)

    # build blocks and cut at boundaries (subtotal/id)
    blocks: List[List[str]] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            blocks.append(current)
            current = []

    for ln in cleaned_lines:
        nln = normalize_line(ln)

        if is_boundary_line(nln):
            flush()
            continue

        if looks_like_item_start(nln):
            flush()
            current = [ln]
        else:
            if current:
                current.append(ln)

    flush()

    items: List[Dict[str, Any]] = []
    for b in blocks:
        parsed = parse_block(b, default_currency)
        if parsed:
            items.append(parsed)

    totals = extract_total(payload.ocr_text, default_currency)

    # sum only plausible totals
    line_totals = [
        x.get("line_total")
        for x in items
        if isinstance(x.get("line_total"), (int, float)) and x.get("line_total") is not None and x.get("line_total") <= 100000
    ]
    sum_guess = float(sum(line_totals)) if line_totals else None

    warnings: List[str] = []
    if totals and sum_guess is not None:
        if abs(totals["total"] - sum_guess) > max(2.0, 0.05 * totals["total"]):
            warnings.append("document_total_mismatch_vs_sum_of_lines")

    return {
        "document": {
            "vin": payload.vin,
            "vehicle": payload.vehicle.model_dump() if payload.vehicle else None,
            "default_currency": default_currency,
        },
        "totals": totals,
        "sum_guess_from_lines": sum_guess,
        "items": items,
        "warnings": warnings,
    }

@app.get("/health")
def health():
    return {"ok": True}
