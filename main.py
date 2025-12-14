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

# ---------------- helpers ----------------

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

# ---------------- regex ----------------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml)\b", re.IGNORECASE)

# accepta pana la 4 zecimale (1000.0000)
NUM_RE = re.compile(r"(?<!\w)(\d+(?:\.\d{1,4})?)(?!\w)")

# prinde si "total cu tva"
TOTAL_LINE_RE = re.compile(
    r"\b(total\s*(de\s*plata|general|cu\s*tva)?|de\s*plata)\b",
    re.IGNORECASE
)

DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
HG_RE = re.compile(r"\bhg\s*\d+\s*/\s*\d+\b", re.IGNORECASE)

# id-uri / serii (si forme cu linii)
ID_TAG_RE = re.compile(r"\b(id|nr|numar|serie)\s*[:#]", re.IGNORECASE)

# cod piesa tipic: multe litere/cifre fara spatii, minim 6
PART_CODE_RE = re.compile(r"^[a-z0-9]{6,}$", re.IGNORECASE)

# ---------------- logic ----------------

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
    labor_k = ["manopera", "ore", "ora", "labor", "diagnoza", "diagnostic", "verificare", "reparatie", "montare"]
    fee_k = ["taxa", "consumabile", "materiale", "transport", "ecotaxa"]
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
    # IMPORTANT: nu tai linii care incep cu "manopera ..." (alea sunt item-uri)
    if l.startswith("manopera "):
        return False

    footer_noise = [
        "generat cu", "autodeviz", "produs al", "vega", "web", "www", "http", "https",
        "pagina", "page", "semnatura", "stampila",
        "in conformitate cu", "unitatea noastra garanteaza",
    ]
    if any(k in l for k in footer_noise):
        return True
    if HG_RE.search(l):
        return True

    # headere de tabel (dar atentie: doar cele "pure")
    header_exact = [
        "materiale/piese", "materiale", "piese", "manopera",
        "denumire piese si materiale utilizate",
        "denumirea serviciilor prestate",
        "u/m", "cantitate", "pret lista", "pret deviz", "valoare", "red", "subtotal", "tva",
        "beneficiar", "comanda", "observatii", "descriere obiect",
        "km.bord", "km bord", "km. bord", "km bord:"
    ]
    if l.strip() in header_exact:
        return True

    # date izolate
    if DATE_RE.search(l) and not any(x in l for x in ["ron", "lei", "eur", "euro", "buc", "ore", "ora", "h"]):
        return True

    # linii foarte scurte, fara litere
    if len(l) < 3:
        return True
    letters = sum(ch.isalpha() for ch in l)
    if letters <= 1 and len(l) < 10:
        return True

    return False

def looks_like_item_start(line: str) -> bool:
    if not line:
        return False

    # manopera ca item start
    if line.startswith("manopera "):
        return True

    parts = line.split()
    if len(parts) < 2:
        return False

    first = parts[0].strip(" -:;")
    if PART_CODE_RE.match(first) and len(first) >= 6:
        if first in ["total", "subtotal"]:
            return False
        return True

    return False

def _pick_qty_from_nums(nums: List[float]) -> Optional[float]:
    # pentru devize auto: qty tipic e intre 0.001 si 100 (1.000, 4.000, 1.5 etc)
    for n in nums:
        if 0.001 <= n <= 100:
            return n
    return None

def parse_block(block_lines: List[str], default_currency: str) -> Optional[Dict[str, Any]]:
    raw_block = "\n".join([clean_text(x) for x in block_lines]).strip()
    if not raw_block:
        return None

    lblock = normalize_text(raw_block)

    currency = default_currency
    if CURRENCY_RE.search(lblock):
        c = detect_currency(lblock)
        currency = c if c != "mixed" else default_currency

    unit = None
    um = UNIT_RE.search(lblock)
    if um:
        unit = um.group(1).lower().replace(".", "")
        if unit == "ora":
            unit = "ore"

    nums = [float(x) for x in NUM_RE.findall(lblock)]

    # qty: daca nu apare "1.000 buc" pe aceeasi linie, ia primul numar mic ca qty
    qty = None
    if unit:
        m_qty = re.search(rf"\b(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b", lblock)
        if m_qty:
            qty = float(m_qty.group(1))
    if qty is None and unit and nums:
        qty = _pick_qty_from_nums(nums)

    unit_price = None
    line_total = None

    # euristica pentru randuri de tabel:
    # - daca ai 3 numere: [qty, unit_price, line_total] (sau [qty, pret_lista, pret_deviz, valoare] etc)
    # - luam de regula ultimele doua ca pret si total, iar qty din primul "mic"
    if len(nums) >= 1:
        line_total = nums[-1]
    if len(nums) >= 2:
        unit_price = nums[-2]

    # daca avem qty si nu avem unit_price dar avem un total, incearca unit_price = total/qty
    if qty is not None and line_total is not None and unit_price is None and qty != 0:
        unit_price = line_total / qty

    # guardrail: ignora "totaluri" absurde per linie
    if line_total is not None and line_total > 100000:
        return None

    first_line = normalize_line(block_lines[0])
    desc = first_line
    desc = CURRENCY_RE.sub(" ", desc)
    desc = UNIT_RE.sub(" ", desc)
    desc = NUM_RE.sub(" ", desc)
    desc = re.sub(r"\s{2,}", " ", desc).strip(" -;:/")

    kind = guess_kind(desc)

    # nu pastra blocuri care sunt doar text + cifre irelevante
    if kind == "unknown" and qty is None and unit is None and unit_price is None and line_total is None:
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

    if qty and unit_price and line_total:
        calc = qty * unit_price
        if abs(calc - line_total) > max(1.0, 0.05 * line_total):
            warnings.append(f"inconsistent_total: qty*unit_price={calc:.2f} vs total={line_total:.2f}")
    else:
        warnings.append("missing_fields")

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

    # 1) cauta "total..." si ia numarul de pe aceeasi linie sau de pe linia urmatoare
    for i, ln in enumerate(lines):
        if TOTAL_LINE_RE.search(ln):
            nums = [float(x) for x in NUM_RE.findall(ln)]
            if nums:
                return {"total": nums[-1], "currency": currency_default}

            # daca pe linia "Total cu TVA:" nu e numar, cauta 1-3 linii dupa
            for j in range(1, 4):
                if i + j < len(lines):
                    nxt = lines[i + j]
                    nums2 = [float(x) for x in NUM_RE.findall(nxt)]
                    if nums2:
                        return {"total": nums2[-1], "currency": currency_default}

    # 2) fallback: cauta in finalul documentului cel mai mare numar plauzibil,
    # dar EXCLUDE liniile cu ID/serie (ex: ID:62-7304-46037)
    tail = lines[-120:] if len(lines) > 120 else lines
    nums_tail: List[float] = []

    for ln in tail:
        if DATE_RE.search(ln):
            continue
        if HG_RE.search(ln):
            continue
        if "km.bord" in ln or "km bord" in ln:
            continue
        if ID_TAG_RE.search(ln):
            continue  # asta elimina fix cazul 46037 din ID
        if "id:" in ln:
            continue

        nums = [float(x) for x in NUM_RE.findall(ln)]
        for n in nums:
            if 1.0 <= n <= 100000:
                nums_tail.append(n)

    if not nums_tail:
        return None

    return {"total": max(nums_tail), "currency": currency_default}

@app.post("/parse")
def parse(payload: InputPayload):
    default_currency = detect_currency(payload.ocr_text)

    raw_lines = payload.ocr_text.split("\n")

    cleaned_lines: List[str] = []
    for ln in raw_lines:
        nln = normalize_line(ln)
        if not nln:
            continue
        if is_noise_line(nln):
            continue
        cleaned_lines.append(ln)

    # construieste blocuri: cod piesa / manopera ... = inceput de item
    blocks: List[List[str]] = []
    current: List[str] = []

    for ln in cleaned_lines:
        nln = normalize_line(ln)
        if looks_like_item_start(nln):
            if current:
                blocks.append(current)
            current = [ln]
        else:
            if current:
                current.append(ln)

    if current:
        blocks.append(current)

    items: List[Dict[str, Any]] = []
    for b in blocks:
        parsed = parse_block(b, default_currency)
        if parsed:
            items.append(parsed)

    totals = extract_total(payload.ocr_text, default_currency)

    line_totals = [
        x["line_total"]
        for x in items
        if isinstance(x.get("line_total"), (int, float))
        and x["line_total"] is not None
        and x["line_total"] <= 100000
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
