import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Normalizer (zipper v3)")

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

def has_digits(s: str) -> bool:
    return any(ch.isdigit() for ch in (s or ""))

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
    labor_k = ["manopera", "labor", "diagnoza", "diagnostic", "verificare", "reparatie", "montare", "caroserie", "ambreiaj"]
    fee_k = ["taxa", "consumabile", "transport", "ecotaxa", "service"]
    part_k = ["filtru", "ulei", "placute", "disc", "kit", "buj", "piesa", "garnitura", "curea", "pompa", "senzor",
              "acumulator", "agrafe", "accesorii", "frana", "ambreiaj", "carcasa", "buson", "capac", "arc"]
    if any(k in desc for k in labor_k):
        return "labor"
    if any(k in desc for k in fee_k):
        return "fee"
    if any(k in desc for k in part_k):
        return "part"
    return "unknown"

def is_noise_line(n: str) -> bool:
    if NUM_ONLY_RE.match(n or ""):
        return False
    n = (n or "")
    if not n.strip():
        return True
    if HG_RE.search(n):
        return True
    if DATE_RE.search(n) and not TOTAL_GENERIC_RE.search(n):
        return True

    junk = [
        "c.i.f", "cif", "nr ord reg com", "registru", "judet:", "adresa:", "telefon:",
        "date client", "date produs", "serie de sasiu", "numar inmatriculare",
        "piese si subansamble", "defectiune", "timp estimativ", "alte obs", "piese furnizate de client",
        "lucrarile au fost", "perioada de garantie", "au fost efectuate de",
        "pagina", "comanda / deviz", "numar: dvz",
        "generat cu", "autodeviz", "produs al", "semnatura", "stampila",
    ]
    if any(k in n for k in junk):
        return True

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

def normalize_desc_for_storage(raw_desc: str) -> str:
    d = normalize_line(raw_desc)
    d = CURRENCY_RE.sub(" ", d)
    d = UNIT_RE.sub(" ", d)
    d = NUM_RE.sub(" ", d)
    d = re.sub(r"\s{2,}", " ", d).strip(" -;:/")
    return d

# ---------------- Total extraction ----------------

def extract_total(text: str, currency_default: str) -> Optional[Dict[str, Any]]:
    lines_raw = (text or "").split("\n")
    lines = [normalize_line(x) for x in lines_raw]

    def get_number_near(i: int) -> Optional[float]:
        nums = [float(x) for x in NUM_RE.findall(lines[i])]
        if nums:
            return nums[-1]
        for j in range(1, 6):
            if i + j < len(lines):
                nxt = lines[i + j].strip()
                if not nxt:
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

# ---------------- Zippering helpers ----------------

def zipper_group_numbers(nums: List[float]) -> List[Tuple[float, float, float]]:
    triples: List[Tuple[float, float, float]] = []
    for i in range(0, len(nums), 3):
        if i + 2 >= len(nums):
            break
        triples.append((nums[i], nums[i + 1], nums[i + 2]))
    return triples

def is_material_desc_line(raw: str) -> bool:
    r = clean_text(raw)
    n = normalize_line(r)
    if not r or not has_letters(r, 3):
        return False
    if is_noise_line(n):
        return False
    # traps
    if n.strip() in {"service auto", "service auto la colt srl"}:
        return False
    if "comanda / deviz" in n or "numar:" in n or "data:" in n:
        return False
    return True

def should_join_as_continuation(prev_desc: str, raw_line: str) -> bool:
    """
    Join only if it's a short continuation WITHOUT digits.
    Prevents 'arc ambreiaj5 ...' from sticking to the previous line.
    """
    r = clean_text(raw_line)
    if not r:
        return False
    if has_digits(r):
        return False
    # short continuation like "set", "ptr", etc.
    return len(r) <= 20 and has_letters(r, 3)

# ---------------- MODEL 2 parsers ----------------

def parse_material_table_zip(section_lines: List[str], default_currency: str) -> List[Dict[str, Any]]:
    descs: List[str] = []
    nums: List[float] = []
    pending: Optional[str] = None

    collecting = False

    for raw in section_lines:
        nr = normalize_line(raw)
        if not nr:
            continue

        if "lista operatiuni necesare" in nr or "total materiale" in nr or "total manopera" in nr or "total deviz" in nr:
            break

        # start after header
        if ("denumire materiale" in nr) or (("u.m" in nr or "u/m" in nr) and "cantitate" in nr and "pret" in nr):
            collecting = True
            continue

        if not collecting:
            continue

        # numbers (qty/price/total)
        if NUM_ONLY_RE.match(nr):
            nums.append(float(nr))
            continue

        # description lines
        if is_material_desc_line(raw):
            if pending and should_join_as_continuation(pending, raw):
                pending = clean_text(pending + " " + raw)
                continue

            if pending:
                descs.append(pending)
            pending = clean_text(raw)
            continue

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
            "unit": "buc",
            "unit_price": unit_price,
            "line_total": line_total,
            "currency": default_currency,
            "confidence_fields": {"qty": 0.9, "unit": 0.7, "unit_price": 0.9, "line_total": 0.9},
            "warnings": warnings,
        })
    return items

def parse_labor_table_zip(section_lines: List[str], default_currency: str) -> List[Dict[str, Any]]:
    """
    Model 2 labor often OCRs as a STREAM:
      h
      3.00
      90.00
      h
      3.00
      60.00
      270.00
      180.00
    We pair each 'Manopera ...' description with the next (unit, qty, price, value)
    in sequence.
    """
    descs: List[str] = []
    stream: List[Any] = []  # mix of "UNIT" tokens + float numbers

    in_table = False

    for raw in section_lines:
        nr = normalize_line(raw)
        if not nr:
            continue

        if "total deviz" in nr:
            break

        # start when we see the header (U.M / Cantitate etc)
        if ("u.m" in nr or "u/m" in nr) and ("cantitate" in nr or "cantit" in nr):
            in_table = True
            continue

        # descriptions
        if nr.startswith("manopera ") and has_letters(raw, 3):
            descs.append(clean_text(raw))
            continue

        if not in_table:
            continue

        # unit tokens
        if nr.strip() in {"h", "ora", "ore"}:
            stream.append("UNIT")
            continue

        # numeric-only
        if NUM_ONLY_RE.match(nr):
            stream.append(float(nr))
            continue

        # ignore totals/subtotals text
        if "total materiale" in nr or "total manopera" in nr:
            continue

    if not descs:
        return []

    items: List[Dict[str, Any]] = []

    idx = 0  # pointer in stream
    for raw_desc in descs:
        # find next UNIT
        while idx < len(stream) and stream[idx] != "UNIT":
            idx += 1
        if idx >= len(stream):
            break
        idx += 1  # consume UNIT

        # expect qty, price
        if idx + 1 >= len(stream) or not isinstance(stream[idx], float) or not isinstance(stream[idx + 1], float):
            break
        qty = float(stream[idx]); unit_price = float(stream[idx + 1])
        idx += 2

        # value might appear after ALL rows, so we allow scanning forward:
        # take the next float as line_total
        while idx < len(stream) and not isinstance(stream[idx], float):
            idx += 1
        if idx >= len(stream):
            break
        line_total = float(stream[idx])
        idx += 1

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
            "currency": default_currency,
            "confidence_fields": {"qty": 0.9, "unit": 0.9, "unit_price": 0.9, "line_total": 0.9},
            "warnings": warnings,
        })

    return items

def parse_model2_zip(all_lines: List[str], default_currency: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    mat_section = split_section(
        all_lines,
        start_contains="lista materiale necesare",
        end_contains_any=["lista operatiuni necesare", "total deviz"],
    )
    if mat_section:
        items.extend(parse_material_table_zip(mat_section, default_currency))

    op_section = split_section(
        all_lines,
        start_contains="lista operatiuni necesare",
        end_contains_any=["total deviz"],
    )
    if op_section:
        items.extend(parse_labor_table_zip(op_section, default_currency))

    return items

# ---------------- Model 1 fallback (block parsing) ----------------

def looks_like_item_start_model1(line: str) -> bool:
    line = (line or "").strip()
    if not line:
        return False
    n = normalize_line(line)

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

    unit = None
    mu = UNIT_RE.search(lblock)
    if mu:
        unit = mu.group(1).lower().replace(".", "")
        if unit in ("ora", "h"):
            unit = "ore"

    desc = normalize_desc_for_storage(block_lines[0])
    kind = guess_kind(desc)

    rest = normalize_text("\n".join(block_lines[1:])) if len(block_lines) > 1 else ""
    nums: List[float] = [float(x) for x in NUM_RE.findall(rest)] if rest else []

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

    if qty is not None and line_total is not None and nums and qty != 0:
        target = line_total / qty
        unit_price = min(nums, key=lambda x: abs(x - target))

    if isinstance(line_total, float) and line_total > 1_000_000:
        return None

    warnings: List[str] = []
    if qty is not None and unit_price is not None and line_total is not None and qty != 0:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")
    else:
        warnings.append("missing_fields")

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
        "currency": default_currency,
        "confidence_fields": {
            "qty": 0.7 if qty is not None else 0.0,
            "unit": 0.7 if unit is not None else 0.0,
            "unit_price": 0.6 if unit_price is not None else 0.0,
            "line_total": 0.7 if line_total is not None else 0.0
        },
        "warnings": warnings,
    }

# ---------------- API ----------------

@app.post("/parse")
def parse(payload: InputPayload):
    default_currency = detect_currency(payload.ocr_text)
    if default_currency == "unknown":
        # in RO devizele sunt aproape mereu RON chiar daca nu scrie explicit
        default_currency = "RON"

    all_lines = (payload.ocr_text or "").split("\n")

    items: List[Dict[str, Any]] = []

    text_norm = normalize_text(payload.ocr_text or "")
    has_model2 = ("lista materiale necesare" in text_norm) or ("lista operatiuni necesare" in text_norm)

    if has_model2:
        items = parse_model2_zip(all_lines, default_currency)
    else:
        cleaned_lines: List[str] = []
        for ln in all_lines:
            nln = normalize_line(ln)
            if not nln:
                continue
            if is_noise_line(nln) and not looks_like_item_start_model1(ln):
                continue
            cleaned_lines.append(ln)

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
