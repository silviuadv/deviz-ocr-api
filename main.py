import re
from typing import Any, Dict, List, Optional, Tuple

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
    # 2,142.00 -> 2142.00 (comma thousands)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # 1,50 -> 1.50 (comma decimals)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s

def normalize_line(s: str) -> str:
    return normalize_text(s)

# ---------------- Regex ----------------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,4})?)(?!\w)")
NUM_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d{1,4})?\s*$")

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
HG_RE = re.compile(r"\bhg\s*\d+\s*/\s*\d+\b", re.IGNORECASE)

ID_TAG_RE = re.compile(r"\b(id|nr|numar|serie)\s*[:#]", re.IGNORECASE)
PART_CODE_RE = re.compile(r"^[a-z0-9]{6,}$", re.IGNORECASE)

# total labels (ordered by importance)
TOTAL_DEVIZ_RE = re.compile(r"\b(total\s*deviz)\b", re.IGNORECASE)
TOTAL_CU_TVA_RE = re.compile(r"\b(total\s*cu\s*tva)\b", re.IGNORECASE)
TOTAL_DE_PLATA_RE = re.compile(r"\b(total\s*(de\s*plata|general))\b", re.IGNORECASE)
TOTAL_MANOPERA_RE = re.compile(r"\b(total\s*manopera)\b", re.IGNORECASE)
TOTAL_MATERIALE_RE = re.compile(r"\b(total\s*materiale)\b", re.IGNORECASE)
TOTAL_GENERIC_RE = re.compile(r"\btotal\b", re.IGNORECASE)

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
        "pompa", "senzor", "acumulator", "agrafe", "accesorii", "frana",
        "ambreiaj", "carcasa", "buson", "capac"
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

    noise_keywords = [
        # older footer noise
        "generat cu", "autodeviz", "produs al", "vega", "web", "www", "http", "https",
        "pagina", "page", "semnatura", "stampila",
        "in conformitate cu", "unitatea noastra garanteaza",

        # business/header junk (deviz model 2)
        "c.i.f", "cif", "nr ord reg com", "registru", "judet:", "adresa:", "telefon:",
        "date client", "date produs", "serie de sasiu", "numar inmatriculare",
        "lucrarile au fost", "perioada de garantie", "au fost efectuate de",
        "comanda / deviz", "service auto", "numar: dvz", "pagina 1 din 1",
        "piese si subansamble", "defectiune", "timp estimativ", "alte obs",
        "piese furnizate de client",
    ]
    if any(k in l for k in noise_keywords):
        return True
    if HG_RE.search(l):
        return True

    header_exact = {
        # common table headers
        "materiale/piese",
        "denumire piese si materiale utilizate",
        "denumirea serviciilor prestate",
        "u/m", "cantitate", "pret lista", "pret deviz", "valoare", "red", "tva",
        "beneficiar", "comanda", "observatii", "descriere obiect",
        "km.bord", "km bord", "km. bord", "km bord:",
        "pret manopera", "fara tva", "%",

        # model 2 headers
        "lista materiale necesare:",
        "lista operatiuni necesare:",
        "denumire materiale",
        "u.m. cantitate",
        "pret",
        "valoare",
        "nr.cod produs",
        "nr. cod produs denumire materiale",
    }
    if l.strip() in header_exact:
        return True

    # drop obvious date-only lines
    if DATE_RE.search(l) and not any(x in l for x in ["ron", "lei", "eur", "euro", "buc", "ore", "ora", "h"]):
        return True

    if len(l.strip()) < 2:
        return True

    return False

def looks_like_item_start(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False

    l = normalize_line(line)

    if l.startswith("manopera "):
        return True

    # model 1: cod piesa (ex: 5NU0QVAHL ...)
    parts = line.split()
    if len(parts) >= 2:
        first = parts[0].strip(" -:;")
        if PART_CODE_RE.match(first) and len(first) >= 6 and first not in {"total", "subtotal"}:
            return True

    # model 2: denumiri fara cod, in caps (CARCASA AMBREIAJ..., BUSON FILTRU..., etc)
    letters = sum(ch.isalpha() for ch in line)
    if letters >= 6:
        if line == line.upper() and not any(k in l for k in [
            "lista materiale", "lista operatiuni", "denumire", "u.m", "cantitate", "pret", "valoare",
            "total", "date client", "date produs"
        ]):
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

# ---------------- Core block parsing (parts/fees) ----------------

def parse_block(block_lines: List[str], default_currency: str) -> Optional[Dict[str, Any]]:
    if not block_lines:
        return None

    raw_block = "\n".join(clean_text(x) for x in block_lines).strip()
    if not raw_block:
        return None

    currency = default_currency
    if CURRENCY_RE.search(raw_block):
        c = detect_currency(raw_block)
        if c != "mixed":
            currency = c

    lblock = normalize_text(raw_block)

    unit = None
    mu = UNIT_RE.search(lblock)
    if mu:
        unit = mu.group(1).lower().replace(".", "")
        if unit == "ora":
            unit = "ore"

    first_line = normalize_line(block_lines[0])
    desc = first_line
    desc = CURRENCY_RE.sub(" ", desc)
    desc = UNIT_RE.sub(" ", desc)
    desc = NUM_RE.sub(" ", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" -;:/")

    kind = guess_kind(desc)

    # extract numbers from the rest of block
    rest = normalize_text("\n".join(block_lines[1:])) if len(block_lines) > 1 else ""
    nums: List[float] = []
    if rest:
        nums = [float(x) for x in NUM_RE.findall(rest)]

    # qty from numeric-only lines (0.001..100)
    qty = None
    for ln in block_lines[1:]:
        nln = normalize_line(ln)
        if NUM_ONLY_RE.match(nln):
            v = float(nln)
            if 0.001 <= v <= 100:
                qty = v
                break

    unit_price = None
    line_total = None

    if len(nums) >= 1:
        line_total = nums[-1]
    if len(nums) >= 2:
        unit_price = nums[-2]

    # Fix 1: qty=1 and only one price => that price is both unit_price and total
    if qty == 1.0:
        price_candidates = [n for n in nums if n >= 0.01]
        if len(price_candidates) == 1:
            unit_price = price_candidates[0]
            line_total = price_candidates[0]

    # Fix 2: if we have qty and total, choose unit_price closest to total/qty
    if qty is not None and line_total is not None and nums and qty != 0:
        target_price = line_total / qty
        unit_price = min(nums, key=lambda x: abs(x - target_price))

    # ignore absurd
    if isinstance(line_total, float) and line_total > 1000000:
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

    # drop pure unknown junk
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

# ---------------- Total extraction ----------------

def extract_total(text: str, currency_default: str) -> Optional[Dict[str, Any]]:
    lines_raw = (text or "").split("\n")
    lines = [normalize_line(x) for x in lines_raw]

    def get_number_near(i: int) -> Optional[float]:
        nums = [float(x) for x in NUM_RE.findall(lines[i])]
        if nums:
            return nums[-1]
        for j in range(1, 4):
            if i + j < len(lines):
                nxt = lines[i + j].strip()
                if not nxt:
                    continue
                if DATE_RE.search(nxt) or HG_RE.search(nxt):
                    continue
                nums2 = [float(x) for x in NUM_RE.findall(nxt)]
                if nums2:
                    return nums2[-1]
        return None

    candidates: List[Tuple[int, int, float]] = []  # (score, index, value)

    for i, ln in enumerate(lines):
        if not ln:
            continue

        score = None
        if TOTAL_DEVIZ_RE.search(ln):
            score = 100
        elif TOTAL_CU_TVA_RE.search(ln):
            score = 95
        elif TOTAL_DE_PLATA_RE.search(ln):
            score = 90
        elif TOTAL_MANOPERA_RE.search(ln):
            score = 40
        elif TOTAL_MATERIALE_RE.search(ln):
            score = 30
        elif TOTAL_GENERIC_RE.search(ln):
            score = 10

        if score is None:
            continue

        val = get_number_near(i)
        if val is None:
            continue
        if val <= 0 or val > 1000000:
            continue

        candidates.append((score, i, val))

    if candidates:
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best = candidates[0]
        return {"total": best[2], "currency": currency_default}

    tail = lines[-180:] if len(lines) > 180 else lines
    nums_tail: List[float] = []
    for ln in tail:
        if DATE_RE.search(ln) or HG_RE.search(ln):
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
            if 1.0 <= n <= 1000000:
                nums_tail.append(n)

    if not nums_tail:
        return None
    return {"total": max(nums_tail), "currency": currency_default}

# ---------------- Labor section parser (column-based) ----------------

def parse_labor_section(clean_lines: List[str], default_currency: str) -> List[Dict[str, Any]]:
    nlines = [normalize_line(x) for x in clean_lines]

    try:
        start_idx = next(i for i, ln in enumerate(nlines) if ln.strip() == "manopera")
    except StopIteration:
        return []

    labor_descs: List[str] = []
    i = start_idx + 1
    while i < len(nlines):
        ln = nlines[i].strip()
        raw_ln = clean_lines[i].strip()

        if not ln:
            i += 1
            continue

        if ln.startswith("id:") or "subtotal" in ln:
            break

        if ln.startswith("manopera "):
            labor_descs.append(raw_ln)

        i += 1

    target = len(labor_descs)
    if target == 0:
        return []

    def parse_num_and_decimals(raw_line: str) -> Optional[Tuple[float, int]]:
        s = clean_text(raw_line)
        s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
        s = re.sub(r"(\d),(\d)", r"\1.\2", s)
        s = s.strip()
        if not NUM_ONLY_RE.match(s):
            return None
        dec = 0
        if "." in s:
            dec = len(s.split(".")[-1])
        try:
            return float(s), dec
        except Exception:
            return None

    nums_tuples: List[Tuple[float, int]] = []
    for j in range(i, len(nlines)):
        nl = nlines[j].strip()
        parsed = parse_num_and_decimals(clean_lines[j])
        if parsed:
            v, d = parsed
            if 0 < v <= 100000:
                nums_tuples.append((v, d))
        if "total" in nl and ("tva" in nl or "deviz" in nl):
            break

    qtys = [v for (v, d) in nums_tuples if d == 3 and 0.001 <= v <= 100]
    prices = [v for (v, d) in nums_tuples if d == 2 and 0.01 <= v <= 10000]  # model 2 uses 2 decimals for prices
    totals = [v for (v, d) in nums_tuples if d == 2 and 0.01 <= v <= 100000]  # totals also 2 decimals

    # when both are 2 decimals, we disambiguate by pairing:
    # (qty, price, total) tends to appear in order; we take first 2*target after qtys as [price,total] pairs
    triplets: List[Tuple[Optional[float], Optional[float], Optional[float]]] = []
    if len(qtys) >= target:
        # build a flat list of 2-decimals after filtering out obvious grand totals
        two_dec = [v for (v, d) in nums_tuples if d == 2 and v <= 100000]
        # try to pick pairs per labor row: (price, total)
        for idx in range(target):
            q = qtys[idx]
            p = two_dec[idx * 2] if idx * 2 < len(two_dec) else None
            t = two_dec[idx * 2 + 1] if idx * 2 + 1 < len(two_dec) else None
            triplets.append((q, p, t))
    else:
        for idx in range(target):
            q = qtys[idx] if idx < len(qtys) else None
            p = prices[idx] if idx < len(prices) else None
            t = totals[idx] if idx < len(totals) else None
            triplets.append((q, p, t))

    items: List[Dict[str, Any]] = []
    for idx, desc_raw in enumerate(labor_descs):
        desc_norm = normalize_line(desc_raw).replace("manopera ", "").strip()
        qty, unit_price, line_total = triplets[idx] if idx < len(triplets) else (None, None, None)

        warnings: List[str] = []
        confidence = {"qty": 0.0, "unit": 0.0, "unit_price": 0.0, "line_total": 0.0}

        if qty is not None:
            confidence["qty"] = 0.9
        if unit_price is not None:
            confidence["unit_price"] = 0.7
        if line_total is not None:
            confidence["line_total"] = 0.7

        if qty is None or unit_price is None or line_total is None:
            warnings.append("missing_fields")

        items.append({
            "raw": desc_raw,
            "desc": desc_norm,
            "kind": "labor",
            "qty": qty,
            "unit": "ore",
            "unit_price": unit_price,
            "line_total": line_total,
            "currency": default_currency,
            "confidence_fields": confidence,
            "warnings": warnings,
        })

    return items

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

    labor_items = parse_labor_section(cleaned_lines, default_currency)

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

        if nln.strip() == "manopera" or nln.startswith("manopera "):
            flush()
            continue

        if looks_like_item_start(ln):
            flush()
            current = [ln]
        else:
            if current:
                current.append(ln)

    flush()

    items: List[Dict[str, Any]] = []
    for b in blocks:
        parsed = parse_block(b, default_currency)
        if parsed and parsed.get("kind") != "labor":
            items.append(parsed)

    items.extend(labor_items)

    totals = extract_total(payload.ocr_text, default_currency)

    line_totals = [
        x.get("line_total")
        for x in items
        if isinstance(x.get("line_total"), (int, float)) and x.get("line_total") is not None and x.get("line_total") <= 1000000
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
