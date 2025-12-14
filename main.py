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
    # 1,50 -> 1.50
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s

def is_header_or_noise(line: str) -> bool:
    # elimina chestii tip "DEVIZ", "CLIENT", "CUI", "DATA", etc
    noise_keywords = [
        "deviz", "factura", "client", "cui", "nr", "data", "adresa",
        "telefon", "email", "subtotal", "tva", "serie", "numar",
    ]

    # footer/header frecvent in OCR-uri (autodeviz, pagini, branduri, etc.)
    footer_noise = [
        "generat cu", "autodeviz", "produs al", "vega", "vega web", "vega webs",
        "web", "www", "http", "https",
        "pagina", "page",
        "semnatura", "stampila",
        "cont", "iban", "banca",
    ]

    # daca linia contine marcatori foarte specifici de footer, o tai direct
    if any(k in line for k in footer_noise):
        return True

    if len(line) < 3:
        return True

    # daca linia are foarte putine litere si multe simboluri, e suspect
    letters = sum(ch.isalpha() for ch in line)
    if letters <= 1 and len(line) < 10:
        return True

    # nu o tai daca pare item (are unitati/preturi)
    if any(u in line for u in ["buc", "ore", "ora", "h", "pcs", "lei", "ron", "eur", "euro"]):
        return False

    # header simplu: are keyword, dar nu are cifre
    if any(k in line for k in noise_keywords) and not any(ch.isdigit() for ch in line):
        return True

    return False

# detectii
CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

# prinde numere care arata ca bani: 35, 35.5, 350.00
MONEY_NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,2})?)(?!\w)")

# qty poate fi si fara unitate (ex: "2 x", "2*")
QTY_HINT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(x|\*)\b", re.IGNORECASE)

TOTAL_LINE_RE = re.compile(r"\b(total\s*(de\s*plata|general)?|de\s*plata)\b", re.IGNORECASE)

def detect_currency(text: str) -> str:
    hits = CURRENCY_RE.findall(text)
    if not hits:
        return "unknown"
    # majoritate
    hits = [h.lower() for h in hits]
    if "eur" in hits or "euro" in hits:
        # daca apar ambele, nu ghicim - dar in practica poti decide
        if "ron" in hits or "lei" in hits:
            return "mixed"
        return "EUR"
    return "RON"

def guess_kind(desc: str) -> str:
    # euristici super simple - se rafineaza cu exemple
    labor_k = ["manopera", "ore", "ora", "labor", "diagnoza", "diagnostic", "verificare"]
    fee_k = ["taxa", "consumabile", "materiale", "transport", "ecotaxa"]
    part_k = ["filtru", "ulei", "placute", "disc", "kit", "buj", "piesa", "garnitura", "curea", "pompa", "senzor"]
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

    # ignora linii foarte "meta" care nu par item
    if is_header_or_noise(l):
        return None

    currency = default_currency
    if CURRENCY_RE.search(l):
        c = detect_currency(l)
        currency = c if c != "mixed" else default_currency

    unit = None
    um = UNIT_RE.search(l)
    if um:
        unit = um.group(1).lower().replace(".", "")
        # normalizare "ora" -> "ore"
        if unit == "ora":
            unit = "ore"

    # qty: daca avem "2 buc" / "2 ore"
    qty = None
    if unit:
        m_qty = re.search(rf"\b(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b", l)
        if m_qty:
            qty = float(m_qty.group(1))
    # alt hint: "2 x"
    if qty is None:
        m = QTY_HINT_RE.search(l)
        if m:
            qty = float(m.group(1))

    # extrage toate numerele din linie
    nums = [float(x) for x in MONEY_NUM_RE.findall(l)]

    unit_price = None
    line_total = None

    # euristica: daca ai 2+ numere, de obicei ultimul e total, penultimul e pret unitar
    if len(nums) >= 1:
        line_total = nums[-1]
    if len(nums) >= 2:
        unit_price = nums[-2]

    # Guardrail: sume absurde (OCR/parse gresit)
    if line_total is not None and line_total > 100000:
        return None

    # daca avem qty si unit_price, putem verifica totalul
    warnings = []
    confidence = {"qty": 0.0, "unit": 0.0, "unit_price": 0.0, "line_total": 0.0}

    if qty is not None:
        confidence["qty"] = 0.7
    if unit is not None:
        confidence["unit"] = 0.7
    if unit_price is not None:
        confidence["unit_price"] = 0.5 if len(nums) >= 2 else 0.3
    if line_total is not None:
        confidence["line_total"] = 0.6 if len(nums) >= 1 else 0.3

    # descriere: scoatem bucati evidente de pret/unitate ca sa ramana text
    desc = l
    desc = CURRENCY_RE.sub(" ", desc)
    desc = UNIT_RE.sub(" ", desc)
    desc = MONEY_NUM_RE.sub(" ", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" -;:/")

    kind = guess_kind(desc)

    # filtreaza fals-positive: daca desc e prea scurt si avem doar 1 numar, probabil nu e item
    if len(desc) < 3 and len(nums) <= 1:
        return None

    # Daca nu pare item (fara cantitate/unitate/pret) si e "unknown", ignora
    if kind == "unknown" and qty is None and unit is None and unit_price is None:
        return None

    # sanity checks
    if qty and unit_price and line_total:
        calc = qty * unit_price
        # toleranta: OCR si rotunjiri
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
    # cauta linii care contin "total" si ia ultimul numar
    lines = [normalize_line(x) for x in text.split("\n")]
    candidates = []
    for ln in lines:
        if TOTAL_LINE_RE.search(ln):
            nums = [float(x) for x in MONEY_NUM_RE.findall(ln)]
            if nums:
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

    # sumar
    line_totals = [x["line_total"] for x in items if isinstance(x.get("line_total"), (int, float))]
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
