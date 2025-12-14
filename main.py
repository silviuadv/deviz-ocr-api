import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Normalizer (zipper)")

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

# ---------------- Normalization ----------------

def clean_text(s: str) -> str:
    s = (s or "").replace("\t", " ").replace("\u00a0", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()

def normalize_text(s: str) -> str:
    s = clean_text(s).lower()
    # 2,142.00 -> 2142.00 (remove thousands commas)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # 1,50 -> 1.50
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s

def normalize_line(s: str) -> str:
    return normalize_text(s)

def has_letters(s: str, n: int = 3) -> bool:
    return sum(ch.isalpha() for ch in (s or "")) >= n

def is_all_capsish(raw: str) -> bool:
    raw = (raw or "").strip()
    letters = [c for c in raw if c.isalpha()]
    if len(letters) < 6:
        return False
    return raw == raw.upper()

# ---------------- Regex ----------------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,4})?)(?!\w)")
NUM_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d{1,4})?\s*$")

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
HG_RE = re.compile(r"\bhg\s*\d+\s*/\s*\d+\b", re.IGNORECASE)

PART_CODE_RE = re.compile(r"^[a-z0-9]{6,}$", re.IGNORECASE)

TOTAL_DEVIZ_RE = re.compile(r"\b(total\s*deviz)\b", re.IGNORECASE)
TOTAL_CU_TVA_RE = re.compile(r"\b(total\s*cu\s*tva)\b", re.IGNORECASE)
TOTAL_DE_PLATA_RE = re.compile(r"\b(total\s*(de\s*plata|general))\b", re.IGNORECASE)
TOTAL_GENERIC_RE = re.compile(r"\btotal\b", re.IGNORECASE)

# ---------------- Core helpers ----------------

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
    desc = (desc or "")
    labor_k = ["manopera", "labor", "diagnoza", "diagnostic", "verificare", "reparatie", "montare", "caroserie"]
    fee_k = ["taxa", "consumabile", "transport", "ecotaxa", "service"]
    part_k = ["filtru", "ulei", "placute", "disc", "kit", "buj", "piesa", "garnitura", "curea", "pompa", "senzor",
              "acumulator", "agrafe", "accesorii", "frana", "ambreiaj", "carcasa", "buson", "capac"]
    if any(k in desc for k in labor_k):
        return "labor"
    if any(k in desc for k in fee_k):
        return "fee"
    if any(k in desc for k in part_k):
        return "part"
    return "unknown"

def is_noise_line(n: str) -> bool:
    """
    NOTE: numeric-only lines are NOT noise (they are qty/price lines in model 2)
    """
    if NUM_ONLY_RE.match(n or ""):
        return False

    n = (n or "")
    if not n.strip():
        return True

    # drop legal paragraphs, dates, etc
    if HG_RE.search(n):
        return True
    if DATE_RE.search(n) and not TOTAL_GENERIC_RE.search(n):
        return True

    # strong junk phrases
    junk = [
        "c.i.f", "cif", "nr ord reg com", "registru", "judet:", "adresa:", "telefon:",
        "date client", "date produs", "serie de sasiu", "numar inmatriculare",
        "piese si subansamble", "defectiune", "timp estimativ", "alte obs", "piese furnizate de client",
        "lucrarile au fost", "perioada de garantie", "au fost efectuate de",
        "pagina", "comanda / deviz", "numar: dvz",
        "midsoft", "municipiul", "bucuresti",
        "generat cu", "autodeviz", "produs al", "semnatura", "stampila",
    ]
    if any(k in n for k in junk):
        return True

    # table headers
    headers = {
        "lista materiale necesare:",
        "lista operatiuni necesare:",
        "nr.cod produs",
        "nr. cod produs denumire materiale",
        "denumire materiale",
        "u.m. cantitate",
        "u/m", "u.m", "cantitate", "pret", "valoare",
        "pret lista", "pret deviz", "fara tva", "red", "%", "tva",
        "materiale/piese", "manopera",
    }
    if n.strip() in headers:
        return True

    return False

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
                if HG_RE.search(nxt):
                    continue
                nums2 = [float(x) for x in NUM_RE.findall(nxt)]
                if nums2:
                    return nums2[-1]
        return None

    candidates: List[Tuple[int, int, float]] = []
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
        elif TOTAL_GENERIC_RE.search(ln):
            score = 10
        if score is None:
            continue
        val = get_number_near(i)
        if val is None:
            continue
        if val <= 0 or val > 1_000_000:
            continue
        candidates.append((score, i, val))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return {"total": candidates[0][2], "currency": currency_default}

# ---------------- Section helpers ----------------

def split_section(lines: List[str], start_contains: str, end_contains_any: List[str]) -> List[str]:
    n = [normalize_line(x) for x in lines]
    start = None
    for i, ln in enumerate(n):
        if start_contains in ln:
            start = i
            break
    if start is None:
        return []

    end = len(lines)
    for j in range(start + 1, len(lines)):
        for ep in end_contains_any:
            if ep in n[j]:
                end = j
                return lines[start:end]
    return lines[start:end]

# ---------------- Zippering (model 2) ----------------

def zipper_group_numbers(nums: List[float]) -> List[Tuple[float, float, float]]:
    triples: List[Tuple[float, float, float]] = []
    for i in range(0, len(nums), 3):
        if i + 2 >= len(nums):
            break
        triples.append((nums[i], nums[i + 1], nums[i + 2]))
    return triples

def normalize_desc_for_storage(raw_desc: str) -> str:
    d = normalize_line(raw_desc)
    d = CURRENCY_RE.sub(" ", d)
    d = UNIT_RE.sub(" ", d)
    d = NUM_RE.sub(" ", d)
    d = re.sub(r"\s{2,}", " ", d).strip(" -;:/")
    return d

def parse_material_table_zip(section_lines: List[str]) -> List[Dict[str, Any]]:
    """
    Works for model 2:
      - collects descriptions
      - collects numeric-only lines
      - zips description[i] with triples[i] (qty, price, total)
    Handles multi-line descriptions like "arc ambreiaj..." by attaching as continuation.
    """
    descs: List[str] = []
    nums: List[float] = []
    pending: Optional[str] = None

    for raw in section_lines:
        nr = normalize_line(raw)
        if not nr:
            continue

        # stop if we hit next sections/totals
        if "lista operatiuni necesare" in nr or "total deviz" in nr or "total materiale" in nr or "total manopera" in nr:
            break

        # skip headers
        if any(k in nr for k in ["denumire materiale", "nr.cod produs", "u.m", "cantitate", "pret", "valoare", "comanda / deviz"]):
            continue

        # numbers
        if NUM_ONLY_RE.match(nr):
            nums.append(float(nr))
            continue

        # ignore junk
        if is_noise_line(nr):
            continue

        # continuation line (not ALL CAPS) -> append to previous desc
        if pending and has_letters(raw, 3) and (not is_all_capsish(raw)):
            pending = clean_text(pending + " " + raw)
            continue

        # new desc
        if has_letters(raw, 3):
            if pending:
                descs.append(pending)
            pending = clean_text(raw)

    if pending:
        descs.append(pending)

    triples = zipper_group_numbers(nums)
    n = min(len(descs), len(triples))

    items: List[Dict[str, Any]] = []
    for i in range(n):
        qty, unit_price, line_total = triples[i]
        raw_desc = descs[i]
        desc = normalize_desc_for_storage(raw_desc)

        warnings: List[str] = []
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")

        items.append({
            "raw": clean_text(raw_desc),
            "desc": desc,
            "kind": "part",
            "qty": qty,
            "unit": "buc",  # model 2 materials are typically pieces; you can set None if you prefer
            "unit_price": unit_price,
            "line_total": line_total,
            "currency": "unknown",
            "confidence_fields": {"qty": 0.9, "unit": 0.4, "unit_price": 0.9, "line_total": 0.9},
            "warnings": warnings,
        })
    return items

def parse_labor_table_zip(section_lines: List[str]) -> List[Dict[str, Any]]:
    """
    For model 2 labor:
      descs: lines containing "Manopera ..."
      nums: numeric-only lines after the table header (qty, price, total) repeating
    """
    descs: List[str] = []
    nums: List[float] = []
    started = False

    for raw in section_lines:
        nr = normalize_line(raw)
        if not nr:
            continue

        if "total deviz" in nr or "total materiale" in nr or "total manopera" in nr:
            break

        # detect table header, then start collecting nums
        if ("u.m" in nr and "cantitate" in nr and "pret" in nr and "valoare" in nr) or ("u.m" in nr and "cantitate" in nr and "pret" in nr):
            started = True
            continue

        if is_noise_line(nr):
            continue

        if "manopera" in nr and has_letters(raw, 3):
            descs.append(clean_text(raw))
            continue

        if started and NUM_ONLY_RE.match(nr):
            nums.append(float(nr))

    triples = zipper_group_numbers(nums)
    n = min(len(descs), len(triples))

    items: List[Dict[str, Any]] = []
    for i in range(n):
        qty, unit_price, line_total = triples[i]
        raw_desc = descs[i]
        desc = normalize_desc_for_storage(raw_desc.replace("manopera", "").strip())

        warnings: List[str] = []
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")

        items.append({
            "raw": clean_text(raw_desc),
            "desc": desc,
            "kind": "labor",
            "qty": qty,
            "unit": "ore",
            "unit_price": unit_price,
            "line_total": line_total,
            "currency": "unknown",
            "confidence_fields": {"qty": 0.9, "unit": 0.8, "unit_price": 0.9, "line_total": 0.9},
            "warnings": warnings,
        })
    return items

def parse_model2_zip(all_lines: List[str], default_currency: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    mat_section = split_section(
        all_lines,
        start_contains="lista materiale necesare",
        end_contains_any=["lista operatiuni necesare", "total deviz", "total materiale", "total manopera"],
    )
    mat_items = parse_material_table_zip(mat_section)
    for it in mat_items:
        it["currency"] = default_currency
    items.extend(mat_items)

    op_section = split_section(
        all_lines,
        start_contains="lista operatiuni necesare",
        end_contains_any=["total deviz", "total materiale", "total manopera"],
    )
    op_items = parse_labor_table_zip(op_section)
    for it in op_items:
        it["currency"] = default_currency
    items.extend(op_items)

    return items

# ---------------- Model 1 (block) parsing ----------------

def looks_like_item_start_model1(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    n = normalize_line(line)

    # part code at start (e.g. 5NU0QVAHL ...)
    parts = line.split()
    if len(parts) >= 2:
        first = parts[0].strip(" -:;")
        if PART_CODE_RE.match(first) and first not in {"total", "subtotal"}:
            return True

    if n.startswith("manopera "):
        return True

    return False

def parse_block_model1(block_lines: List[str], default_currency: str) -> Optional[Dict[str, Any]]:
    if not block_lines:
        return None

    raw_block = "\n".join(clean_text(x) for x in block_lines).strip()
    if not raw_block:
        return None

    lblock = normalize_text(raw_block)

    if is_noise_line(lblock):
        return None

    currency = default_currency

    # unit
    unit = None
    mu = UNIT_RE.search(lblock)
    if mu:
        unit = mu.group(1).lower().replace(".", "")
        if unit in ("ora", "h"):
            unit = "ore"

    # description from first line
    first_line = normalize_line(block_lines[0])
    desc = normalize_desc_for_storage(block_lines[0])
    kind = guess_kind(desc)

    # numbers from rest
    rest = normalize_text("\n".join(block_lines[1:])) if len(block_lines) > 1 else ""
    nums: List[float] = [float(x) for x in NUM_RE.findall(rest)] if rest else []

    # qty from numeric-only lines
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

    # better: if qty and total exist, choose price closest to total/qty
    if qty is not None and line_total is not None and nums and qty != 0:
        target = line_total / qty
        unit_price = min(nums, key=lambda x: abs(x - target))

    # avoid absurd totals
    if isinstance(line_total, float) and line_total > 1_000_000:
        return None

    warnings: List[str] = []
    if qty is not None and unit_price is not None and line_total is not None and qty != 0:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")
    else:
        warnings.append("missing_fields")

    # skip useless garbage
    if kind == "unknown" and "missing_fields" in warnings:
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
        "confidence_fields": {"qty": 0.7 if qty is not None else 0.0,
                              "unit": 0.7 if unit is not None else 0.0,
                              "unit_price": 0.6 if unit_price is not None else 0.0,
                              "line_total": 0.7 if line_total is not None else 0.0},
        "warnings": warnings,
    }

# ---------------- API ----------------

@app.post("/parse")
def parse(payload: InputPayload):
    default_currency = detect_currency(payload.ocr_text)
    all_lines = (payload.ocr_text or "").split("\n")

    items: List[Dict[str, Any]] = []

    # A) Try zippering tables (MODEL 2). If sections exist, this gives best results.
    items.extend(parse_model2_zip(all_lines, default_currency))

    # B) Fallback block parsing (MODEL 1).
    # Keep only useful lines or item starts.
    cleaned_lines: List[str] = []
    for ln in all_lines:
        nln = normalize_line(ln)
        if not nln:
            continue
        if is_noise_line(nln) and not looks_like_item_start_model1(ln):
            continue
        cleaned_lines.append(ln)

    # Build blocks starting with item-start lines
    blocks: List[List[str]] = []
    current: List[str] = []

    def flush():
        nonlocal current
        if current:
            blocks.append(current)
            current = []

    for ln in cleaned_lines:
        if looks_like_item_start_model1(ln):
            flush()
            current = [ln]
        else:
            if current:
                current.append(ln)
    flush()

    for b in blocks:
        parsed = parse_block_model1(b, default_currency)
        if parsed:
            items.append(parsed)

    totals = extract_total(payload.ocr_text, default_currency)

    line_totals = [
        x.get("line_total")
        for x in items
        if isinstance(x.get("line_total"), (int, float)) and x.get("line_total") is not None and x.get("line_total") <= 1_000_000
    ]
    sum_guess = float(sum(line_totals)) if line_totals else None

    doc_warnings: List[str] = []
    if totals and sum_guess is not None:
        if abs(totals["total"] - sum_guess) > max(2.0, 0.05 * totals["total"]):
            doc_warnings.append("document_total_mismatch_vs_sum_of_lines")

    return {
        "document": {
            "vin": payload.vin,
            "vehicle": payload.vehicle.model_dump() if payload.vehicle else None,
            "default_currency": default_currency,
        },
        "totals": totals,
        "sum_guess_from_lines": sum_guess,
        "items": items,
        "warnings": doc_warnings,
    }

@app.get("/health")
def health():
    return {"ok": True}
