from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
import math
import re
from statistics import median

# =====================================================
# Airtable / Make SAFE ENUMS — NU SE SCHIMBA
# =====================================================

MathStatus = Literal["OK", "MISMATCH", "INSUFFICIENT_DATA"]
InternalVerdict = Literal["OK", "SUSPICIOUS", "BAD", "INSUFFICIENT_DATA"]
VATStatus = Literal["OK", "INCONSISTENT", "NOT_APPLICABLE"]
LaborStatus = Literal["OK", "WARN", "FAIL"]

# =====================================================
# INPUT MODELS
# =====================================================

class DevizInternalItem(BaseModel):
    desc: str = ""
    kind: Literal["part", "labor"] = "part"
    qty: float = 0.0
    unit: str = ""
    unit_price: float = 0.0
    line_total: float = 0.0
    currency: str = "RON"


class DevizInternalTotals(BaseModel):
    materials: Optional[float] = None
    labor: Optional[float] = None
    vat: Optional[float] = None
    subtotal_no_vat: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "RON"


class DevizInternalInput(BaseModel):
    items: List[DevizInternalItem] = Field(default_factory=list)
    totals: DevizInternalTotals = Field(default_factory=DevizInternalTotals)

# =====================================================
# OUTPUT MODEL (Make/Airtable READY)
# =====================================================

class DevizInternalOutput(BaseModel):
    MathConsistency: MathStatus
    InternalVerdict: InternalVerdict
    VATCheck: VATStatus
    LaborSanityResult: LaborStatus

    InternalScore: int
    InternalFlags: str
    confidence: Literal["LOW", "MED", "HIGH"]

    debug: Dict[str, Any] = Field(default_factory=dict)

# =====================================================
# HELPERS
# =====================================================

def _f(x: Optional[float]) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def _sum(items: List[DevizInternalItem], kind: Optional[str] = None) -> float:
    return sum(float(i.line_total or 0.0) for i in items if not kind or i.kind == kind)

def _approx(a: Optional[float], b: Optional[float], abs_tol: float = 5.0, rel_tol: float = 0.02) -> bool:
    if a is None or b is None:
        return False
    diff = abs(a - b)
    if diff <= abs_tol:
        return True
    return diff / max(1.0, abs(b)) <= rel_tol

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())

# =====================================================
# 1. MATH CONSISTENCY — STRICT, FARA INVENTII
# =====================================================

def _math_consistency(payload: DevizInternalInput) -> (MathStatus, List[str], Dict[str, Any]):
    items = payload.items
    t = payload.totals

    gt = _f(t.grand_total)
    mat = _f(t.materials)
    lab = _f(t.labor)
    vat = _f(t.vat)
    net = _f(t.subtotal_no_vat)

    sum_all = _sum(items)
    sum_parts = _sum(items, "part")
    sum_labor = _sum(items, "labor")

    checks = 0
    mismatches = 0
    reasons: List[str] = []

    if gt is not None and mat is not None and lab is not None:
        checks += 1
        if not _approx(mat + lab, gt, 10.0):
            mismatches += 1
            reasons.append("grand_total_vs_materials_plus_labor")

    if mat is not None and sum_parts > 0:
        checks += 1
        if not _approx(sum_parts, mat):
            mismatches += 1
            reasons.append("materials_total_mismatch")

    if lab is not None and sum_labor > 0:
        checks += 1
        if not _approx(sum_labor, lab):
            mismatches += 1
            reasons.append("labor_total_mismatch")

    if gt is not None and sum_all > 0:
        checks += 1
        if not _approx(sum_all, gt, 15.0):
            mismatches += 1
            reasons.append("grand_total_vs_items_sum")

    if checks == 0:
        return "INSUFFICIENT_DATA", ["not_enough_numeric_data"], {}

    if mismatches > 0:
        return "MISMATCH", reasons, {}

    return "OK", [], {}

# =====================================================
# 2. VAT CHECK
# =====================================================

def _vat_check(payload: DevizInternalInput) -> VATStatus:
    t = payload.totals
    gt = _f(t.grand_total)
    net = _f(t.subtotal_no_vat)
    vat = _f(t.vat)

    if gt is None or net is None or net <= 0:
        return "NOT_APPLICABLE"

    if vat is not None:
        expected = net * 0.19
        return "OK" if _approx(vat, expected, 3.0) else "INCONSISTENT"

    implied = gt - net
    rate = implied / net if implied > 0 else None

    if rate and 0.16 <= rate <= 0.22:
        return "OK"

    return "INCONSISTENT"

# =====================================================
# 3. LABOR SANITY
# =====================================================

def _labor_sanity(payload: DevizInternalInput) -> LaborStatus:
    labor = [i for i in payload.items if i.kind == "labor" and i.line_total > 0]

    if not labor:
        return "OK"

    hours = sum(i.qty for i in labor if i.qty > 0)
    value = sum(i.line_total for i in labor)

    if hours <= 0:
        return "FAIL"

    rate = value / hours

    if rate < 30 or rate > 450:
        return "WARN"

    small_chunks = [i for i in labor if 0 < i.qty <= 0.4]
    if len(small_chunks) >= max(3, int(0.4 * len(labor))):
        return "WARN"

    return "OK"

# =====================================================
# 4. TYPOLOGII DE TEPARI (AICI E GREUL)
# =====================================================

def _typologies(payload: DevizInternalInput, math_status: MathStatus) -> (int, List[str]):
    items = payload.items
    t = payload.totals

    gt = _f(t.grand_total)
    sum_all = _sum(items)

    score = 0
    flags: List[str] = []

    # T1 — matematica nu iese
    if math_status == "MISMATCH":
        score += 25
        flags.append("math_mismatch")

    # T2 — ascunde valoare in manopera vaga
    labor = [i for i in items if i.kind == "labor"]
    vague_labor = [i for i in labor if i.qty == 0 or i.unit_price == 0]
    if len(vague_labor) >= 2:
        score += 20
        flags.append("labor_value_hidden")

    # T3 — total mare, linii putine
    if gt and len(items) <= 3 and gt >= 500:
        score += 15
        flags.append("too_few_lines_for_total")

    # T4 — fragmentare artificiala
    hours = [i.qty for i in labor if i.qty > 0]
    if hours:
        if median(hours) <= 0.4 and len(hours) >= 5:
            score += 15
            flags.append("labor_fragmentation")

    # T5 — sume rotunde suspect
    prices = [i.unit_price for i in items if i.unit_price > 0]
    round_prices = [p for p in prices if p % 10 == 0]
    if prices and len(round_prices) / len(prices) >= 0.7:
        score += 10
        flags.append("suspicious_round_prices")

    return min(100, score), flags

# =====================================================
# 5. VERDICT FINAL INTERN
# =====================================================

def _internal_verdict(math_status: MathStatus, score: int) -> InternalVerdict:
    if math_status == "INSUFFICIENT_DATA" and score < 20:
        return "INSUFFICIENT_DATA"

    if score >= 70:
        return "BAD"

    if score >= 40 or math_status == "MISMATCH":
        return "SUSPICIOUS"

    return "OK"

# =====================================================
# PUBLIC API
# =====================================================

def internal_deviz_check(payload: DevizInternalInput) -> DevizInternalOutput:
    math_status, math_reasons, _ = _math_consistency(payload)
    vat_status = _vat_check(payload)
    labor_status = _labor_sanity(payload)

    score, flags = _typologies(payload, math_status)
    verdict = _internal_verdict(math_status, score)

    confidence = "LOW"
    if score >= 60:
        confidence = "HIGH"
    elif score >= 30:
        confidence = "MED"

    return DevizInternalOutput(
        MathConsistency=math_status,
        InternalVerdict=verdict,
        VATCheck=vat_status,
        LaborSanityResult=labor_status,
        InternalScore=score,
        InternalFlags=", ".join(flags),
        confidence=confidence,
        debug={
            "sum_all_lines": _sum(payload.items),
            "sum_parts": _sum(payload.items, "part"),
            "sum_labor": _sum(payload.items, "labor"),
        }
    )
