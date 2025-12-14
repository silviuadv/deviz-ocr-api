import re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Parser Universal 3.0 (Fixed Shift)")

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

class ExtractedItem(BaseModel):
    type: str  # "manopera", "part", "fee", "unknown"
    description: str
    qty: float
    unit: Optional[str] = None
    unit_price: float
    line_total: float
    confidence_score: float
    warnings: List[str] = []

class ParseResponse(BaseModel):
    document_info: Dict[str, Any]
    detected_type: str
    totals: Optional[Dict[str, Any]]
    items: List[ExtractedItem]
    warnings: List[str]

# ---------------- Regex Constants ----------------

CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)
NUM_RE = re.compile(r"(?<!\S)\d+(?:\.\d+)?(?!\S)") # Cifre curate: 100, 100.00

# Regex specific pentru liniile care contin DOAR numere (pentru Deviz 2)
NUM_ONLY_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")

# ---------------- Helpers ----------------

def clean_text(s: str) -> str:
    s = (s or "").replace("\t", " ").replace("\u00a0", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()

def normalize_text(s: str) -> str:
    """Standardizează punctuația (1.200,00 -> 1200.00)."""
    s = clean_text(s).lower()
    # elimina separator mii
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)
    # virgula zecimala -> punct
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s

def extract_floats(line: str) -> List[float]:
    return [float(x) for x in NUM_RE.findall(normalize_text(line))]

def detect_currency(text: str) -> str:
    hits = CURRENCY_RE.findall(text or "")
    if not hits: return "RON"
    if "eur" in [h.lower() for h in hits]: return "EUR"
    return "RON"

# --- FILTRU AGRESIV PENTRU ZGOMOT (Rezolvă decalarea "Service Auto") ---
def is_garbage_line(line: str) -> bool:
    l = line.lower().strip()
    if len(l) < 2: return True
    
    # Cuvinte interzise care apar in OCR intercalate cu tabelul
    junk_phrases = [
        "c.i.f", "j40", "adresa", "telefon", "pagina", "data:", "numar:", "nr.",
        "midsoft", "bucuresti", "valoare", "cantitate", "pret", "u.m.", "cod produs",
        "comanda", "deviz", "semnatura", "lucrarile", "garantie", "total",
        "factura", "aaa 2012", "popescu", "service auto", "denumire materiale"
    ]
    
    if any(k in l for k in junk_phrases):
        return True
        
    # Dacă linia e doar o dată (ex: 27-04-2020)
    if re.search(r"\d{2}-\d{2}-\d{4}", l):
        return True
        
    return False

# ---------------- ENGINE 1: COLT DEVIZ (Zipper Fix) ----------------
# Logică: Colectăm separat descrierile CURATE și numerele, apoi le unim.

def parse_colt_deviz_v3(lines: List[str]) -> List[ExtractedItem]:
    items = []
    
    # 1. Separăm textul în secțiuni (Materiale vs Manoperă)
    # Folosim liste separate pentru a nu amesteca manopera cu piesele
    mat_raw_lines = []
    lab_raw_lines = []
    
    current_section = None
    
    for line in lines:
        l_low = line.lower()
        if "lista materiale necesare" in l_low:
            current_section = "parts"
            continue
        elif "lista operatiuni necesare" in l_low:
            current_section = "labor"
            continue
        elif "total deviz" in l_low: # Stop parsing la final
            current_section = None
            
        if current_section == "parts":
            mat_raw_lines.append(line)
        elif current_section == "labor":
            lab_raw_lines.append(line)

    # 2. Procesor pentru o secțiune
    def process_section_zipper(section_lines, kind):
        descs = []
        nums = []
        
        for line in section_lines:
            if is_garbage_line(line):
                continue
            
            # Curățăm "1.", "2." de la început
            clean_line = re.sub(r"^\d+\.\s*", "", line).strip()
            if not clean_line: continue

            # DECIDEM: E text sau e număr?
            # Verificăm dacă are litere (excluzând 'h' solitar sau unități simple)
            # Regex: caută litere a-z.
            has_letters = bool(re.search(r'[a-zA-Z]', clean_line))
            found_nums = extract_floats(clean_line)

            # Manopera are linii dubioase gen "h" sau "ore". Le ignorăm ca descrieri.
            is_unit_only = clean_line.lower() in ["h", "ore", "buc", "l"]

            if has_letters and not is_unit_only:
                # Este DESCRIERE (ex: "Carcasa Ambreiaj")
                # Chiar dacă conține cifre (1.9 TDI), e text.
                descs.append(clean_line)
            elif found_nums:
                # Este LINIE DE VALORI (ex: "1.00", "100.00")
                nums.extend(found_nums)

        # 3. ZIPPER (Unirea)
        # Regula Deviz 2: 
        # Materiale: 3 numere per item (Cantitate, Preț, Valoare)
        # Manopera: OCR-ul tău scoate 3 numere (3.00, 90.00, 270.00) sau 2.
        
        vals_per_item = 3
        
        # Siguranță: calculăm limita
        limit = min(len(descs), len(nums) // 2) # Măcar 2 numere per item să avem

        final_items = []
        num_idx = 0
        
        for i in range(limit):
            # Luăm următoarea descriere
            desc = descs[i]
            
            # Încercăm să extragem următorul grup de numere.
            # Ne uităm la următorii 3. Dacă al 3-lea e produsul primelor 2, e grup de 3.
            # Altfel e grup de 2.
            
            if num_idx + 2 < len(nums):
                q = nums[num_idx]
                p = nums[num_idx+1]
                val = nums[num_idx+2]
                
                # Verificăm matematica: Q * P ~= Val
                if abs((q * p) - val) < 0.1 * val:
                    # E grup de 3 perfect
                    final_items.append(ExtractedItem(
                        type=kind, description=desc, qty=q, unit="buc" if kind=="part" else "ore",
                        unit_price=p, line_total=val, confidence_score=0.9
                    ))
                    num_idx += 3
                else:
                    # Nu se pupă matematica, sau OCR-ul a spart rândul.
                    # Asumăm grup de 2 (Cantitate, Preț) și calculăm totalul.
                    final_items.append(ExtractedItem(
                        type=kind, description=desc, qty=q, unit="buc" if kind=="part" else "ore",
                        unit_price=p, line_total=q*p, confidence_score=0.8
                    ))
                    # ATENȚIE: La manoperă uneori unitatea '3.00' apare de două ori sau e haos.
                    # Avansăm cu 2 sau 3? Dacă e 3.00 90.00 ... urmatorul e 3.00
                    num_idx += 2
            elif num_idx + 1 < len(nums):
                 # Au rămas doar 2 numere
                q = nums[num_idx]
                p = nums[num_idx+1]
                final_items.append(ExtractedItem(
                    type=kind, description=desc, qty=q, unit="buc" if kind=="part" else "ore",
                    unit_price=p, line_total=q*p, confidence_score=0.8
                ))
                num_idx += 2

        return final_items

    items.extend(process_section_zipper(mat_raw_lines, "part"))
    items.extend(process_section_zipper(lab_raw_lines, "manopera"))
    return items

# ---------------- ENGINE 2: SURUBELNITA (Deviz 3) ----------------
# Rând mixt: Manoperă (stânga) | Piese (dreapta)

def parse_surubelnita_deviz(lines: List[str]) -> List[ExtractedItem]:
    items = []
    active = False
    
    for raw in lines:
        line = normalize_text(raw)
        if "lucrari convenite" in line.lower(): active = True; continue
        if "total manopera" in line.lower(): active = False; break
        if not active or "operatie" in line.lower(): continue
        
        nums = extract_floats(line)
        if len(nums) < 2: continue
        
        # Stanga: Manopera
        match_num = NUM_RE.search(line)
        if match_num:
            text_part = line[:match_num.start()].strip()
            parts = text_part.split()
            desc = text_part
            if parts and parts[0].isdigit(): desc = " ".join(parts[1:])
            
            lab_q = nums[0]
            lab_val = nums[1]
            lab_p = round(lab_val/lab_q, 2) if lab_q else 0
            
            items.append(ExtractedItem(
                type="manopera", description=desc, qty=lab_q, unit="ore",
                unit_price=lab_p, line_total=lab_val, confidence_score=0.9
            ))
            
            # Dreapta: Piese (dacă există Qty, Pret, Total la final)
            if len(nums) >= 5:
                mat_q = nums[-3]
                mat_p = nums[-2]
                mat_t = nums[-1]
                # Extragem textul dintre valoarea manoperei si cantitatea piesei
                patt = re.compile(rf"{re.escape(str(lab_val))}\s+(.+?)\s+{re.escape(str(mat_q))}")
                m = patt.search(line)
                if m:
                    mat_desc = m.group(1).replace("buc", "").replace("ltr", "").strip()
                    items.append(ExtractedItem(
                        type="part", description=mat_desc, qty=mat_q, unit="buc",
                        unit_price=mat_p, line_total=mat_t, confidence_score=0.8
                    ))
    return items

# ---------------- ENGINE 3: GENERIC (Deviz 1) ----------------
# Linie standard: Text ... Cantitate ... Pret ... Total

def parse_generic_deviz(lines: List[str]) -> List[ExtractedItem]:
    items = []
    current_type = "part"
    
    for raw in lines:
        line = normalize_text(raw)
        if "manopera" in line.lower() and "subtotal" not in line.lower(): current_type = "manopera"
        
        nums = extract_floats(line)
        if len(nums) >= 3:
            total = nums[-1]
            price = nums[-2]
            qty = nums[-3]
            
            # Validare matematică
            if abs((qty * price) - total) < 2.0:
                match = re.search(rf"(.*?)\s+{re.escape(str(qty))}", line)
                if match:
                    desc = match.group(1).strip()
                    # Curățare coduri "5NU..."
                    parts = desc.split()
                    if len(parts) > 1 and len(parts[0]) > 4 and any(c.isdigit() for c in parts[0]):
                        desc = " ".join(parts[1:])
                    
                    items.append(ExtractedItem(
                        type=current_type, description=desc, qty=qty, unit="buc" if current_type=="part" else "ore",
                        unit_price=price, line_total=total, confidence_score=0.75
                    ))
    return items

# ---------------- TOTALURI DOCUMENT ----------------

def extract_doc_total(lines: List[str]) -> Optional[float]:
    for line in lines:
        norm = normalize_text(line)
        if "total" in norm and ("deviz" in norm or "general" in norm):
            nums = extract_floats(norm)
            if nums: return nums[-1]
    return None

# ---------------- API PRINCIPAL ----------------

@app.post("/parse", response_model=ParseResponse)
def parse_endpoint(payload: InputPayload):
    text = payload.ocr_text
    lines = text.split('\n')
    currency = detect_currency(text)
    
    # 1. Detectare Tip Document
    detected = "generic"
    items = []
    
    # Priorități detectare
    if "lista materiale necesare" in text.lower():
        detected = "deviz_2_colt"
        items = parse_colt_deviz_v3(lines)
    elif "lucrari convenite" in text.lower():
        detected = "deviz_3_surubelnita"
        items = parse_surubelnita_deviz(lines)
    else:
        detected = "deviz_1_standard"
        items = parse_generic_deviz(lines)
        
    # 2. Totaluri
    doc_total = extract_doc_total(lines)
    calc_sum = sum(i.line_total for i in items)
    
    warnings = []
    if doc_total and abs(calc_sum - doc_total) > 5.0:
        warnings.append(f"Mismatch: Sum Items ({calc_sum:.2f}) != Doc Total ({doc_total})")

    return ParseResponse(
        document_info={"vin": payload.vin, "currency": currency},
        detected_type=detected,
        totals={"amount": doc_total, "currency": currency} if doc_total else None,
        items=items,
        warnings=warnings
    )

@app.get("/health")
def health():
    return {"status": "ok"}
