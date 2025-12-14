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

# --- helpers ---

def clean_text(s: str) -> str:
    s = s.replace("\t", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    s = s.replace("\u00a0", " ")
    return s.strip()

def normalize_line(s: str) -> str:
    s = clean_text(s).lower()

    # 2,142.00 -> 2142.00 (virgula ca separator de mii)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # 1,50 -> 1.50 (virgula ca separator zecimal)
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)

    return s

# pattern-uri de zgomot
DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
ID_LIKE_RE = re.compile(r"\b(id|nr|numar|serie)\s*[:#]?\s*[\w-]{6,}\b", re.IGNORECASE)

def is_header_or_noise(line: str) -> bool:
    noise_keywords = [
        "deviz", "factura", "client", "cui", "nr", "data", "adresa",
        "telefon", "email", "subtotal", "tva", "serie", "numar",
        # headere de tabel
        "materiale/piese", "manopera", "denumire", "u/m", "cantitate",
        "pret lista", "pret deviz", "valoare", "red", "km.bord", "km bord",
        "descriere obiect", "beneficiar", "comanda", "observatii",
    ]

    footer_noise = [
        "generat cu", "autodeviz", "produs al", "vega", "web", "www", "http", "https",
        "pagina", "page", "semnatura", "stampila",
        "in conformitate cu", "unitatea noastra garanteaza",
        "hg ",  # HG 394/1995 etc.
    ]

    if any(k in line for k in footer_noise):
        return True

    if len(line) < 3:
        return True

    letters = sum(ch.isalpha() for ch in line)
    if letters <= 1 and len(line) < 10:
        return True

    # daca e un header clar (cu keyword-uri) si NU are preturi/unitati, il tai
    if any(k in line for k in noise_keywords):
        # daca are unitati/moneda, poate fi item, nu tai automat
        if any(u in line for u in ["buc", "ore", "ora", "h", "pcs", "lei", "ron", "eur", "euro"]):
            return False
        # daca are cifre dar e gen "km.bord 120000", tot zgomot
        return True

    return False

# detectii
CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

# numere: accepta pana la 4 zecimale (1000.0000)
MONEY_NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,4})?)(?!\w)")

QTY_HINT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(x|\*)\b", re.IGNORECASE)

# include si "total cu tva"
TOTAL_LINE_RE = re.compile(
    r"\b(total\s*(de\s*plata|general|cu\s*tva)?|de\s*plata)\b",
    re.IGNORECASE
)

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
    fee_k = ["taxa", "consumabile", "materiale", "transport", "ecotaxa", "materiale/piese"]
    part_k = ["filtru", "ulei", "placute", "disc", "kit", "buj", "piesa", "garnitura", "curea", "pompa", "senzor", "acumulator", "agrafe", "accesorii"]
    if any(k in desc for k in labor_k):
        return "labor"
    if any(k in desc for k in fee_k):
        return "fee"
    if any(k in desc for k in part_k):
        return "part"
    return "unknown"

def parse_item_line(line: str, default_currency: str) -> Optional[Dict[str, Any]]:
    raw = clean_text(line)
    l = normalize_line(line)

    if is_header_or_noise(l):
        return None

    # filtre tari: date / ani / id-uri (cand nu e linie de item)
    if DATE_RE.search(l) and not any(x in l for x in ["buc", "ore", "ora", "h", "ron", "lei", "eur", "euro"]):
        return None
    if ("hg" in l) or ("km.bord" in l) or ("km bord" in l):
        return None
    if ID_LIKE_RE.search(l) and not any(x in l for x in ["ron", "lei", "eur", "euro"]):
        return None

    currency = default_currency
    if CURRENCY_RE.search(l):
        c = detect_currency(l)
        currency = c if c != "mixed" else default_currency

    unit = None
    um = UNIT_RE.search(l)
    if um:
        unit = um.group(1).lower().replace(".", "")
        if unit == "ora":
            unit = "ore"

    qty = None
    if unit:
        m_qty = re.search(rf"\b(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b", l)
        if m_qty:
            qty = float(m_qty.group(1))

    if qty is None:
        m = QTY_HINT_RE.search(l)
        if m:
            qty = float(m.group(1))

    nums = [float(x) for x in MONEY_NUM_RE.findall(l)]

    unit_price = None
    line_total = None

    if len(nums) >= 1:
        line_total = nums[-1]
    if len(nums) >= 2:
        unit_price = nums[-2]

    # Guardrail: sume absurde (OCR/parse gresit)
    if line_total is not None and line_total > 100000:
        return None

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

    desc = l
    desc = CURRENCY_RE.sub(" ", desc)
    desc = UNIT_RE.sub(" ", desc)
    desc = MONEY_NUM_RE.sub(" ", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" -;:/")

    kind = guess_kind(desc)

    # daca descrierea e practic goala si nu ai bani reali, nu e item
    if len(desc) < 3 and len(nums) <= 1:
        return None

    # daca nu pare item (fara cantitate/unitate/pret) si e unknown, ignora
    if kind == "unknown" and qty is None and unit is None and unit_price is None:
        return None

    # sanity checks
    if qty and unit_price and line_total:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")
    else:
        warnings.append("missing_fields")

    return {
        "raw": raw,
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
            nums = [float(x) for x in MONEY_NUM_RE.findall(ln)]
            if nums:
                # la "total cu tva: 2142.00" ultimul numar e totalul
                candidates.append(nums[-1])
    if not candidates:
        return None
    return {"total": candidates[-1], "currency": currency_default}

@app.post("/parse")
def parse(payload: InputPayload):
    default_currency = detect_currency(payload.ocr_text)

    items: List[Dict[str, Any]] = []
    for line in payload.ocr_text.split("\n"):
        parsed = parse_item_line(line, default_currency)
        if parsed:
            items.append(parsed)

    totals = extract_total(payload.ocr_text, default_currency)

    # sumar: suma doar din valori “plauzibile”
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
