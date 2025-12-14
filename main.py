import re
from typing import Any, Dict, List, Optional
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Deviz OCR Parser Universal 2.0")

# ---------------- Models ----------------

class Vehicle(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    engine: Optional[str] = None

class InputPayload(BaseModel):
    vin: str
    vehicle: Optional[Vehicle] = None
    ocr_text: str

class ExtractedItem(BaseModel):
    type: str  # "manopera", "part"
    description: str
    qty: float
    unit: str
    unit_price: float
    line_total: float
    confidence: float

class ParseResponse(BaseModel):
    detected_type: str
    totals: Optional[Dict[str, Any]]
    items: List[ExtractedItem]
    warnings: List[str]

# ---------------- Helpers & Regex ----------------

# Detect numbers (floats)
NUM_RE = re.compile(r"(?<!\S)\d+(?:\.\d+)?(?!\S)") 
CURRENCY_RE = re.compile(r"\b(ron|lei|eur|euro)\b", re.IGNORECASE)

def normalize_text(s: str) -> str:
    """Standardize punctuation (1.200,00 -> 1200.00)."""
    s = (s or "").strip().replace("\t", " ")
    # remove thousands separators
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    s = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", s)
    # comma decimal to dot
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s

def extract_floats(line: str) -> List[float]:
    return [float(x) for x in NUM_RE.findall(normalize_text(line))]

def is_garbage_line(line: str) -> bool:
    """Filters out headers, footers, metadata."""
    l = line.lower().strip()
    if len(l) < 2: return True
    # Keywords found in headers/footers of your examples
    junk = [
        "c.i.f", "j40", "adresa", "telefon", "pagina", "data:", "numar:", 
        "midsoft", "bucuresti", "valoare", "cantitate", "pret", "u.m.", 
        "comanda", "deviz", "semnatura", "lucrarile", "garantie", "total",
        "factura", "aaa 2012", "popescu", "service auto"
    ]
    if any(k in l for k in junk): return True
    return False

# ---------------- ENGINE 1: COLUMN STREAM (Deviz 2 - Colt) ----------------
# Logic: Collect all Descriptions -> Collect all Numbers -> Match them sequentially

def parse_colt_deviz_v2(lines: List[str]) -> List[ExtractedItem]:
    items = []
    
    # 1. Segment text into sections based on headers
    section_map = {"parts": [], "labor": []}
    current_key = None
    
    for line in lines:
        l_low = line.lower()
        if "lista materiale" in l_low:
            current_key = "parts"
            continue
        elif "lista operatiuni" in l_low:
            current_key = "labor"
            continue
        elif "total" in l_low and "deviz" in l_low:
            current_key = None # End parsing
            
        if current_key:
            section_map[current_key].append(line)

    # 2. Processor function for a section
    def process_section(section_lines, kind):
        descs = []
        nums = []
        
        for line in section_lines:
            if is_garbage_line(line):
                continue
                
            # Clean generic index numbers "1.", "2." at start of line
            clean_line = re.sub(r"^\d+\.\s*", "", line).strip()
            
            # Extract numbers from line
            found_nums = extract_floats(clean_line)
            
            if found_nums:
                # If line is JUST numbers (or mostly numbers), add to num queue
                nums.extend(found_nums)
            elif len(clean_line) > 3:
                # Text line
                descs.append(clean_line)

        # 3. Match Logic (The "Zipper")
        # For 'parts', OCR output implies format: Qty, Price, Total (3 nums per item)
        # For 'labor', OCR output implies: Qty, Price (sometimes Total is far away or mixed)
        
        num_step = 3 if kind == "part" else 3 # Trying 3 for labor too based on your OCR
        
        limit = min(len(descs), len(nums) // num_step)
        
        section_items = []
        for i in range(limit):
            # Take next chunk of numbers
            chunk = nums[i*num_step : (i+1)*num_step]
            
            # Logic for Colt format: Qty is usually first small number, Price is bigger
            # Example Material: 1.00, 100.00, 100.00
            qty = chunk[0]
            price = chunk[1]
            total = chunk[2] if len(chunk) > 2 else qty * price
            
            # Correction for Labor if order is flipped or 'h' confused things
            # In your OCR: 3.00 90.00 (Total 270 is elsewhere sometimes). 
            # But let's assume standard triplet if available.
            
            section_items.append(ExtractedItem(
                type=kind,
                description=descs[i],
                qty=qty,
                unit="buc" if kind == "part" else "ore",
                unit_price=price,
                line_total=total,
                confidence=0.85
            ))
        return section_items

    items.extend(process_section(section_map["parts"], "part"))
    items.extend(process_section(section_map["labor"], "manopera"))
    
    return items

# ---------------- ENGINE 2: MIXED ROW (Deviz 3 - Surubelnita) ----------------
# Logic: Split row by detected number clusters

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
        
        # Left Side (Labor): Text ... Time(0.3) ... Price(12)
        match_num = NUM_RE.search(line)
        if match_num:
            text_part = line[:match_num.start()].strip()
            # Clean Code
            parts = text_part.split()
            desc = text_part
            if parts and parts[0].isdigit(): desc = " ".join(parts[1:])
            
            labor_qty = nums[0]
            labor_val = nums[1]
            labor_price = round(labor_val/labor_qty, 2) if labor_qty else 0
            
            items.append(ExtractedItem(
                type="manopera", description=desc, qty=labor_qty, unit="ore",
                unit_price=labor_price, line_total=labor_val, confidence=0.9
            ))
            
            # Right Side (Material): If 3+ numbers remain (Qty, Price, Total)
            if len(nums) >= 5:
                mat_qty = nums[-3]
                mat_price = nums[-2]
                mat_total = nums[-1]
                
                # Regex extract desc between Labor Val and Mat Qty
                patt = re.compile(rf"{re.escape(str(labor_val))}\s+(.+?)\s+{re.escape(str(mat_qty))}")
                m = patt.search(line)
                if m:
                    mat_desc = m.group(1).replace("buc", "").replace("ltr", "").strip()
                    items.append(ExtractedItem(
                        type="part", description=mat_desc, qty=mat_qty, unit="buc",
                        unit_price=mat_price, line_total=mat_total, confidence=0.8
                    ))
    return items

# ---------------- ENGINE 3: GENERIC BLOCK (Deviz 1 - Standard) ----------------
# Logic: Line = Text ... Qty ... Price ... Total

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
            
            # Validation logic
            if abs((qty * price) - total) < 2.0:
                # Valid line
                match = re.search(rf"(.*?)\s+{re.escape(str(qty))}", line)
                if match:
                    desc = match.group(1).strip()
                    # Clean Code "5NU..."
                    parts = desc.split()
                    if len(parts) > 1 and len(parts[0]) > 4 and any(c.isdigit() for c in parts[0]):
                        desc = " ".join(parts[1:])
                    
                    items.append(ExtractedItem(
                        type=current_type, description=desc, qty=qty, unit="buc" if current_type=="part" else "ore",
                        unit_price=price, line_total=total, confidence=0.75
                    ))
    return items

# ---------------- MAIN API ----------------

@app.post("/parse", response_model=ParseResponse)
def parse_deviz_endpoint(payload: InputPayload):
    text = payload.ocr_text
    lines = text.split('\n')
    
    # 1. Detect Type
    if "service auto la colt" in text.lower() or "lista materiale necesare" in text.lower():
        detected = "deviz_2_colt"
        items = parse_colt_deviz_v2(lines)
    elif "lucrari convenite" in text.lower():
        detected = "deviz_3_surubelnita"
        items = parse_surubelnita_deviz(lines)
    else:
        detected = "deviz_1_standard"
        items = parse_generic_deviz(lines)
        
    # 2. Extract Total Document Value
    total_val = 0.0
    for line in lines:
        if "total" in line.lower() and ("deviz" in line.lower() or "general" in line.lower()):
            nums = extract_floats(line)
            if nums: total_val = nums[-1]
            
    warnings = []
    calc_sum = sum(i.line_total for i in items)
    if total_val > 0 and abs(calc_sum - total_val) > 5.0:
        warnings.append(f"Total mismatch: Sum items ({calc_sum}) != Doc Total ({total_val})")

    return ParseResponse(
        detected_type=detected,
        totals={"amount": total_val, "currency": "RON"},
        items=items,
        warnings=warnings
    )
