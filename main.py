import re
from typing import Any, Dict, List, Optional, Tuple
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Parser Universal")

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
    code: Optional[str] = None
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
UNIT_RE = re.compile(r"\b(buc|buc\.|pcs|ore|ora|h|km|l|ml|ltr)\b", re.IGNORECASE)

# Numbers: 100, 100.00
NUM_RE = re.compile(r"(?<!\S)\d+(?:\.\d+)?(?!\S)")

# Specific for Zipper Method (Deviz 2)
# Lines that are ONLY numbers: "1.00 100.00 100.00"
NUM_ONLY_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*$")
# Lines typical for labor in Deviz 2: "h 3.00 90.00"
LABOR_ZIPPER_RE = re.compile(r"^\s*(?:h|ore|ora)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*(\d+(?:\.\d+)?)?.*$")

# Keywords
LABOR_KEYWORDS = ["manopera", "inlocuire", "demontare", "montare", "reparatie", "revizie", "control", "testare", "diagnoza", "vopsitorie"]
PART_KEYWORDS = ["filtru", "ulei", "placute", "disc", "kit", "bujie", "garnitura", "bucsa", "amortizor", "brat", "curea", "pompa", "ambreiaj", "anvelopa"]

# Total Regex
TOTAL_GENERIC_RE = re.compile(r"\b(total|plata|suma)\b", re.IGNORECASE)

# ---------------- Helpers ----------------

def clean_text(s: str) -> str:
    s = (s or "").replace("\t", " ").replace("\u00a0", " ")
    s = re.sub(r"[ ]{2,}", " ", s)
    return s.strip()

def normalize_number_format(s: str) -> str:
    """Standardize 1.200,00 to 1200.00 and 1,50 to 1.50"""
    s = clean_text(s).lower()
    # remove thousands separators (dot or comma lookbehind)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s) # 2,100 -> 2100
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s) # 2.100 -> 2100
    # comma decimals: 1,50 -> 1.50
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s

def extract_floats(line: str) -> List[float]:
    norm = normalize_number_format(line)
    return [float(x) for x in NUM_RE.findall(norm)]

def detect_currency(text: str) -> str:
    hits = CURRENCY_RE.findall(text or "")
    if not hits: return "RON"
    hits = [h.lower() for h in hits]
    if "eur" in hits or "euro" in hits: return "EUR"
    return "RON"

def guess_kind(desc: str) -> str:
    desc = desc.lower()
    if any(k in desc for k in LABOR_KEYWORDS): return "labor"
    if any(k in desc for k in PART_KEYWORDS): return "part"
    return "part" # Default fallback

# ---------------- ENGINE A: Mixed Row (Deviz 3) ----------------
# Handles rows like: "Code | Operation | Time | Val | PartName | Qty | Price"

def parse_mixed_row_deviz(lines: List[str], default_currency: str) -> List[ExtractedItem]:
    items = []
    start_processing = False
    
    for raw_line in lines:
        line = normalize_number_format(raw_line)
        
        # Trigger / Stop
        if "lucrari convenite" in line.lower():
            start_processing = True
            continue
        if "total manopera" in line.lower() or "cost reparatie" in line.lower():
            start_processing = False
            break
            
        if not start_processing:
            continue
            
        # Skip Headers
        if "operatie" in line.lower() or "val." in line.lower():
            continue
            
        nums = extract_floats(line)
        if len(nums) < 2: continue

        # 1. LABOR PART (Left side)
        # Expected structure: Code(maybe) Text Time(num) Price(num)
        # We look for the first number (Time)
        match = NUM_RE.search(line)
        if match:
            text_part = line[:match.start()].strip()
            # Try extract code
            parts = text_part.split()
            code = None
            desc = text_part
            if parts and parts[0].isdigit() and len(parts[0]) >= 3:
                code = parts[0]
                desc = " ".join(parts[1:])
            
            time_qty = nums[0]
            val_labor = nums[1]
            
            # Sanity check for labor
            if time_qty < 20 and val_labor > 0:
                unit_price = round(val_labor / time_qty, 2) if time_qty else 0
                items.append(ExtractedItem(
                    type="labor",
                    code=code,
                    description=desc,
                    qty=time_qty,
                    unit="ore",
                    unit_price=unit_price,
                    line_total=val_labor,
                    confidence_score=0.9,
                    warnings=[]
                ))

        # 2. MATERIAL PART (Right side)
        # If we have 3 more numbers at the end (Qty, UnitPrice, Total)
        if len(nums) >= 5:
            # The last 3 numbers are usually Qty, Price, Total
            qty = nums[-3]
            price = nums[-2]
            total = nums[-1]
            
            # The description is hidden between the Labor Value (nums[1]) and Qty (nums[-3])
            # We use regex to find text between these two values
            try:
                # Construct regex pattern dynamically based on values found
                patt = re.compile(rf"{re.escape(str(nums[1]))}\s+(.+?)\s+{re.escape(str(qty))}")
                m = patt.search(line)
                if m:
                    mat_desc = m.group(1).strip()
                    # Clean up common unit noise
                    mat_desc = re.sub(r"\b(buc|ltr|l)\b", "", mat_desc, flags=re.IGNORECASE).strip()
                    
                    items.append(ExtractedItem(
                        type="part",
                        description=mat_desc,
                        qty=qty,
                        unit="buc", # default
                        unit_price=price,
                        line_total=total,
                        confidence_score=0.85,
                        warnings=[]
                    ))
            except:
                pass # Regex failed to construct or match

    return items

# ---------------- ENGINE B: Zipper Method (Deviz 2) ----------------
# Handles documents where lists of descriptions and lists of prices are disjointed

def parse_zipper_deviz(lines: List[str], default_currency: str) -> List[ExtractedItem]:
    
    # Containers
    part_descs = []
    part_nums = [] # (qty, price, total)
    labor_descs = []
    labor_nums = [] # (qty, price, total)
    
    current_section = "unknown"
    
    for raw_line in lines:
        line = normalize_number_format(raw_line)
        line_lower = line.lower()
        
        # Section Detection
        if "materiale necesare" in line_lower:
            current_section = "parts"
            continue
        if "operatiuni necesare" in line_lower:
            current_section = "labor"
            continue
        if "total deviz" in line_lower:
            current_section = "footer"
            continue
            
        # Garbage Filter
        if len(line) < 3: continue
        if any(x in line_lower for x in ["midsoft", "adresa", "telefon", "c.i.f", "j40", "pagina", "data:", "numar:"]):
            continue

        # 1. Capture pure numbers (Parts usually)
        mat_match = NUM_ONLY_LINE_RE.match(line)
        if mat_match:
            try:
                q, p, t = map(float, mat_match.groups())
                part_nums.append((q, p, t))
            except: pass
            continue
            
        # 2. Capture labor numbers (starts with 'h' or 'ore')
        lab_match = LABOR_ZIPPER_RE.match(line)
        if lab_match:
            try:
                groups = lab_match.groups()
                q = float(groups[0])
                p = float(groups[1])
                t = float(groups[2]) if groups[2] else q*p
                labor_nums.append((q, p, t))
            except: pass
            continue
            
        # 3. Capture Descriptions
        # If it's not a number line, it's text.
        if extract_floats(line) and len(extract_floats(line)) > 2:
            # This line has text AND numbers (standard line), handle separately if needed
            # But for Deviz 2, they are usually separate.
            pass
        
        clean_desc = re.sub(r"^\d+\.\s*", "", raw_line).strip() # Remove "1. "
        
        if "manopera" in clean_desc.lower():
            labor_descs.append(clean_desc)
        elif current_section == "parts" or (current_section == "unknown" and len(clean_desc) > 5):
            # Exclude obvious keywords appearing in text lines
            if "total" not in clean_desc.lower():
                part_descs.append(clean_desc)

    items = []
    
    # ZIP Parts
    limit_p = min(len(part_descs), len(part_nums))
    for i in range(limit_p):
        q, p, t = part_nums[i]
        
        # Code extraction attempt from description
        desc_text = part_descs[i]
        code = None
        parts = desc_text.split()
        if len(parts) > 1 and (parts[0].isdigit() or len(parts[0]) > 5):
             # Basic heuristic for code at start
             pass # keeping full desc for now as fuzzy match is better on full string
        
        items.append(ExtractedItem(
            type="part",
            description=desc_text,
            qty=q,
            unit="buc",
            unit_price=p,
            line_total=t,
            confidence_score=0.9,
            warnings=[]
        ))

    # ZIP Labor
    limit_l = min(len(labor_descs), len(labor_nums))
    for i in range(limit_l):
        q, p, t = labor_nums[i]
        items.append(ExtractedItem(
            type="labor",
            description=labor_descs[i],
            qty=q,
            unit="ore",
            unit_price=p,
            line_total=t,
            confidence_score=0.95,
            warnings=[]
        ))
        
    return items

# ---------------- ENGINE C: Generic Block (Deviz 1) ----------------
# Fallback for standard tables where line = Desc + Qty + Price

def parse_generic_lines(lines: List[str], default_currency: str) -> List[ExtractedItem]:
    items = []
    
    for raw_line in lines:
        line = normalize_number_format(raw_line)
        nums = extract_floats(line)
        
        if len(nums) < 3: continue
        
        # Assume format: Text ... Qty Price Total
        total = nums[-1]
        price = nums[-2]
        qty = nums[-3]
        
        # Validation: Qty * Price ~= Total
        if qty == 0: continue
        calc = qty * price
        if abs(calc - total) > max(1.0, 0.1 * total):
            continue # Math doesn't work, probably noise or phone numbers
            
        # Extract text: Everything before the Qty
        # Regex to find the Qty number in the line
        match = re.search(rf"(.*?)\s+{re.escape(str(qty))}", line)
        if match:
            desc_raw = match.group(1).strip()
            
            # Code logic
            parts = desc_raw.split()
            code = None
            if len(parts) > 0 and len(parts[0]) > 4 and any(c.isdigit() for c in parts[0]):
                code = parts[0]
                desc_raw = " ".join(parts[1:])
                
            kind = guess_kind(desc_raw)
            
            items.append(ExtractedItem(
                type=kind,
                code=code,
                description=desc_raw,
                qty=qty,
                unit="buc" if kind == "part" else "ore",
                unit_price=price,
                line_total=total,
                confidence_score=0.7,
                warnings=[]
            ))
            
    return items

# ---------------- Total Extraction ----------------

def extract_total_doc(text: str) -> Optional[Dict[str, Any]]:
    lines = text.split('\n')
    candidates = []
    
    for line in lines:
        norm = normalize_number_format(line).lower()
        if "total" in norm or "plata" in norm:
            nums = extract_floats(norm)
            if nums:
                val = nums[-1]
                if 10 < val < 100000:
                    score = 100 if "total deviz" in norm or "total general" in norm else 50
                    candidates.append((score, val))
    
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return {"amount": candidates[0][1]}
    return None

# ---------------- Main API ----------------

@app.post("/parse", response_model=ParseResponse)
def parse_deviz(payload: InputPayload):
    ocr_text = payload.ocr_text
    lines = ocr_text.split('\n')
    default_currency = detect_currency(ocr_text)
    
    # --- DETECT TYPE ---
    is_type_3 = any("lucrari convenite" in l.lower() for l in lines)
    is_type_2 = any("lista materiale necesare" in l.lower() for l in lines)
    
    detected = "generic"
    items = []
    
    if is_type_3:
        detected = "mixed_row_surubelnita"
        items = parse_mixed_row_deviz(lines, default_currency)
    elif is_type_2:
        detected = "zipper_colt"
        items = parse_zipper_deviz(lines, default_currency)
    else:
        detected = "standard_block"
        items = parse_generic_lines(lines, default_currency)
        
    # Totals
    total_info = extract_total_doc(ocr_text)
    if total_info:
        total_info["currency"] = default_currency

    # Warnings
    warnings = []
    calc_sum = sum(i.line_total for i in items)
    if total_info and abs(calc_sum - total_info["amount"]) > 5.0:
        warnings.append(f"Sum mismatch: Lines={calc_sum:.2f} vs Doc={total_info['amount']}")

    return ParseResponse(
        document_info={
            "vin": payload.vin,
            "currency": default_currency
        },
        detected_type=detected,
        totals=total_info,
        items=items,
        warnings=warnings
    )

@app.get("/health")
def health():
    return {"status": "ok"}
