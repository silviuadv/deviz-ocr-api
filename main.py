import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Normalizer")

class Vehicle(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None
    year_range: Optional[str] = None

class InputPayload(BaseModel):
    vin: str
    vehicle: Optional[Vehicle] = None
    ocr_text: str

def clean_text(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    s = s.replace("\u00a0", " ")
    return s.strip()

def normalize_text(s: str) -> str:
    s = clean_text(s).lower()

    # 2,142.00 -> 2142.00 (virgula separator de mii)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # 1,50 -> 1.50 (virgula separator zecimal)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)

    return s

def normalize_line(s: str) -> str:
    return normalize_text(s)

# --------- regex-uri utile ---------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

# accepta pana la 4 zecimale (1000.0000)
NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,4})?)(?!\w)")

TOTAL_LINE_RE = re.compile(r"\b(total\s*(de\s*plata|general|cu\s*tva)?|de\s*plata)\b", re.IGNORECASE)

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
HG_RE = re.compile(r"\bhg\s*\d+\s*/\s*\d+\b", re.IGNORECASE)

# cod piesa tipic: multe cifre/litere fara spatii, minim 6-7 caractere
PART_CODE_RE = re.compile(r"^[a-z0-9]{6,}$", re.IGNORECASE)

def detect_currency(text: str) -> str:
    hits = CURRENCY_RE.findall(text)
    if not hits:
        return "unknown"
    hits = [h.lower() for h in hits]
    if "eur" in hits or "euro" in hits:
        if "ron" in hits or "lei" in hits:
            return "mixed"
        return "EUR"
    return "RON"

def guess_kind(desc: str) -> str:
    labor_k = ["manopera", "ore", "ora", "labor", "diagnoza", "diagnostic", "verificare"]
    fee_k = ["taxa", "consumabile", "materiale", "transport", "ecotaxa"]
    part_k = ["filtru", "ulei", "placute", "disc", "kit", "buj", "piesa", "garnitura", "curea", "pompa", "senzor",
              "acumulator", "agrafe", "accesorii", "frana"]
    if any(k in desc for k in labor_k):
        return "labor"
    if any(k in desc for k in fee_k):
        return "fee"
    if any(k in desc for k in part_k):
        return "part"
    return "unknown"

def is_noise_line(l: str) -> bool:
    # footer/header frecvent
    footer_noise = [
        "generat cu", "autodeviz", "produs al", "vega", "web", "www", "http", "https",
        "pagina", "page", "semnatura", "stampila",
        "in conformitate cu", "unitatea noastra garanteaza",
    ]
    if any(k in l for k in footer_noise):
        return True
    if HG_RE.search(l):
        return True
    if len(l) < 3:
        return True
    # linii de coloane / titluri tabel
    header_k = [
        "materiale/piese", "manopera", "denumire", "u/m", "cantitate",
        "pret lista", "pret deviz", "valoare", "red", "subtotal", "tva",
        "beneficiar", "comanda", "observatii", "descriere obiect",
        "km.bord", "km bord",
    ]
    if any(k in l for k in header_k):
        return True
    # linii cu date (cand nu sunt clar item)
    if DATE_RE.search(l) and not any(u in l for u in ["buc", "ore", "ora", "h", "ron", "lei", "eur", "euro"]):
        return True
    return False

# --------- parser pe blocuri ---------

def looks_like_item_start(line: str) -> bool:
    """
    Detecteaza inceput de item.
    Exemplu: "5NU0QXG89 ACUMULATOR 60AH"
    """
    if not line:
        return False
    parts = line.split()
    if len(parts) < 2:
        return False
    first = parts[0]
    # cod fara spatii, destul de lung
    if PART_CODE_RE.match(first) and len(first) >= 6:
        # sa nu fie "subtotal", "total", etc
        if first in ["subtotal", "total"]:
            return False
        return True
    return False

def parse_block(block_lines: List[str], default_currency: str) -> Optional[Dict[str, Any]]:
    """
    Dintr-un bloc (item start + urmatoarele linii) extrage:
    desc, qty, unit, unit_price, line_total.
    """
    raw_block = "\n".join(block_lines).strip()
    if not raw_block:
        return None

    lblock = normalize_text(raw_block)

    # currency
    currency = default_currency
    if CURRENCY_RE.search(lblock):
        c = detect_currency(lblock)
        currency = c if c != "mixed" else default_currency

    # unit
    unit = None
    um = UNIT_RE.search(lblock)
    if um:
        unit = um.group(1).lower().replace(".", "")
        if unit == "ora":
            unit = "ore"

    # qty (cauta "12 buc" sau "4 ore")
    qty = None
    if unit:
        m_qty = re.search(rf"\b(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b", lblock)
        if m_qty:
            qty = float(m_qty.group(1))

    # toate numerele din bloc
    nums = [float(x) for x in NUM_RE.findall(lblock)]

    # guardrail: daca nu sunt numere deloc, nu e item util
    if not nums:
        # dar pastram piesa ca item "fara pret" daca pare piesa
        pass

    # euristica pentru devize tip tabel:
    # ultimele 1-3 numere tind sa fie: pret_lista, pret_deviz, valoare
    unit_price = None
    line_total = None

    if len(nums) >= 1:
        line_total = nums[-1]
    if len(nums) >= 2:
        unit_price = nums[-2]

    # guardrail: sume absurde
    if line_total is not None and line_total > 100000:
        return None

    # desc: luam prima linie a blocului (de obicei cod + descriere)
    first_line = normalize_line(block_lines[0])
    # scoatem numere/unitati/moneda din prima linie ca sa ramana text
    desc = first_line
    desc = CURRENCY_RE.sub(" ", desc)
    desc = UNIT_RE.sub(" ", desc)
    desc = NUM_RE.sub(" ", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" -;:/")

    kind = guess_kind(desc)

    warnings = []
    confidence = {"qty": 0.0, "unit": 0.0, "unit_price": 0.0, "line_total": 0.0}

    if qty is not None:
        confidence["qty"] = 0.7
    if unit is not None:
        confidence["unit"] = 0.7
    if unit_price is not None:
        confidence["unit_price"] = 0.6
    if line_total is not None:
        confidence["line_total"] = 0.7

    if qty and unit_price and line_total:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")
    else:
        warnings.append("missing_fields")

    # daca e doar header sau zgomot mascat, nu-l salvam
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
    lines = [normalize_line(x) for x in text.split("\n")]
    candidates = []
    for ln in lines:
        if TOTAL_LINE_RE.search(ln):
            nums = [float(x) for x in NUM_RE.findall(ln)]
            if nums:
                candidates.append(nums[-1])
    if not candidates:
        return None
    return {"total": candidates[-1], "currency": currency_default}

@app.post("/parse")
def parse(payload: InputPayload):
    default_currency = detect_currency(payload.ocr_text)

    # 1) curatare linii
    raw_lines = payload.ocr_text.split("\n")
    lines = []
    for ln in raw_lines:
        nl = normalize_line(ln)
        if not nl:
            continue
        if is_noise_line(nl):
            continue
        lines.append(ln)  # pastram originalul pentru raw

    # 2) construim blocuri: item start + urmatoarele linii pana la urmatorul item start
    blocks: List[List[str]] = []
    current: List[str] = []

    for ln in lines:
        nln = normalize_line(ln)
        if looks_like_item_start(nln):
            if current:
                blocks.append(current)
            current = [ln]
        else:
            if current:
                # continua blocul curent
                current.append(ln)

    if current:
        blocks.append(current)

    # 3) parse pe blocuri
    items: List[Dict[str, Any]] = []
    for b in blocks:
        parsed = parse_block(b, default_currency)
        if parsed:
            items.append(parsed)

    totals = extract_total(payload.ocr_text, default_currency)

    line_totals = [
        x["line_total"]
        for x in items
        if isinstance(x.get("line_total"), (int, float)) and x["line_total"] is not None and x["line_total"] <= 100000
    ]
    sum_guess = sum(line_totals) if line_totals else None

    warnings = []
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
        "warnings": warnings
    }

@app.get("/health")
def health():
    return {"ok": True}
