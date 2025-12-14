import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Normalizer (zipper v3 - fixed)")

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
    # 2,142.00 -> 2142.00
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
        "lista materiale necesare:", "lista operatiuni necesare:", "nr.cod produs",
        "nr. cod produs denumire materiale", "denumire materiale", "u.m. cantitate",
        "u/m", "u.m", "cantitate", "pret", "valoare", "pret lista", "pret deviz",
        "fara tva", "red", "%", "tva", "materiale/piese", "manopera",
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
        if nums: return nums[-1]
        for j in range(1, 5):
            if i + j < len(lines):
                nxt = lines[i + j].strip()
                if not nxt: continue
                nums2 = [float(x) for x in NUM_RE.findall(nxt)]
                if nums2: return nums2[-1]
        return None
    candidates: List[Tuple[int, int, float]] = []
    for i, ln in enumerate(lines):
        if not ln: continue
        score = None
        if TOTAL_DEVIZ_RE.search(ln): score = 100
        elif TOTAL_CU_TVA_RE.search(ln): score = 95
        elif TOTAL_DE_PLATA_RE.search(ln): score = 90
        elif TOTAL_GENERIC_RE.search(ln): score = 10
        if score is None: continue
        val = get_number_near(i)
        if val is None or val <= 0 or val > 1_000_000: continue
        candidates.append((score, i, val))
    if not candidates: return None
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return {"total": candidates[0][2], "currency": currency_default}

# ---------------- Section helpers ----------------

def split_section(lines: List[str], start_contains: str, end_contains_any: List
