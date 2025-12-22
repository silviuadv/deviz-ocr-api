from typing import List, Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


# =========================
# INPUT MODELS (compatibile cu main.py)
# =========================

class ExtractedItem(BaseModel):
    desc: str = ""
    kind: str                 # "part" | "labor"
    qty: float = 0.0
    unit: str = ""
    unit_price: float = 0.0
    line_total: float = 0.0
    currency: str = "RON"


class Totals(BaseModel):
    materials: Optional[float] = None
    labor: Optional[float] = None
    vat: Optional[float] = None
    subtotal_no_vat: Optional[float] = None
    grand_total: Optional[float] = None
    currency: str = "RON"


class DevizInternalInput(BaseModel):
    items: List[ExtractedItem] = []
    totals: Totals = Totals()


# =========================
# OUTPUT MODEL (Airtable-ready)
# =========================

class DevizInternalResult(BaseModel):
    InternalScore: int                               # 1–100
    InternalVerdict: str                            # OK / SUSPICIOUS / BAD / INSUFFICIENT_DATA
    InternalFlags: str                              # text
    LaborSanityResult: str                          # OK / WARN / FAIL / SKIPPED
    MathConsistency: str                            # OK / MISMATCH / INSUFFICIENT_DATA
    VATCheck: str                                   # OK / MISSING / INCONSISTENT / NOT_APPLICABLE
    debug: Dict[str, Any] = {}


# =========================
# HELPERS
# =========================

def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _clamp_score(x: float) -> int:
    return max(1, min(100, int(round(x))))


# =========================
# CORE LOGIC
# =========================

def internal_deviz_check(payload: DevizInternalInput) -> DevizInternalResult:
    flags: List[str] = []
    score = 100.0
    dbg: Dict[str, Any] = {}

    items = payload.items or []
    totals = payload.totals or Totals()

    parts = [i for i in items if (i.kind or "").lower() == "part"]
    labor = [i for i in items if (i.kind or "").lower() == "labor"]

    # =========================
    # 1. Math consistency (linie)
    # =========================

    mismatches = 0
    for it in items:
        if (it.qty or 0) > 0 and (it.unit_price or 0) > 0:
            expected = float(it.qty) * float(it.unit_price)
            tol = max(2.0, expected * 0.05)  # 2 RON sau 5%
            if abs(expected - float(it.line_total or 0.0)) > tol:
                mismatches += 1

    if len(items) < 2:
        math_consistency = "INSUFFICIENT_DATA"
        score -= 10
    else:
        if mismatches == 0:
            math_consistency = "OK"
        elif mismatches <= 2:
            math_consistency = "MISMATCH"
            score -= mismatches * 5
            flags.append("line_math_mismatch")
        else:
            math_consistency = "MISMATCH"
            score -= 15
            flags.append("multiple_line_math_mismatches")

    dbg["line_math_mismatches"] = mismatches

    # =========================
    # 2. Totaluri vs suma linii
    # =========================

    sum_parts = sum(float(i.line_total or 0.0) for i in parts)
    sum_labor = sum(float(i.line_total or 0.0) for i in labor)
    sum_all = sum(float(i.line_total or 0.0) for i in items)

    if totals.materials is not None:
        tol = max(5.0, float(totals.materials) * 0.05)
        if abs(sum_parts - float(totals.materials)) > tol:
            flags.append("materials_total_mismatch")
            score -= 10

    if totals.labor is not None:
        tol = max(5.0, float(totals.labor) * 0.05)
        if abs(sum_labor - float(totals.labor)) > tol:
            flags.append("labor_total_mismatch")
            score -= 10

    if totals.grand_total is not None:
        tol = max(10.0, float(totals.grand_total) * 0.05)
        if abs(sum_all - float(totals.grand_total)) > tol:
            flags.append("grand_total_mismatch")
            score -= 15

    dbg["sum_parts"] = sum_parts
    dbg["sum_labor"] = sum_labor
    dbg["sum_all"] = sum_all

    # =========================
    # 3. Labor sanity
    # =========================

    if len(labor) == 0:
        labor_sanity = "SKIPPED"
        dbg["labor_hours"] = 0.0
        dbg["labor_value"] = 0.0
        dbg["labor_hourly_rate"] = None
    else:
        labor_hours = sum(float(i.qty or 0.0) for i in labor if (i.qty or 0.0) > 0)
        labor_value = sum(float(i.line_total or 0.0) for i in labor if (i.line_total or 0.0) > 0)

        hourly_rate = _safe_ratio(labor_value, labor_hours)

        dbg["labor_hours"] = labor_hours
        dbg["labor_value"] = labor_value
        dbg["labor_hourly_rate"] = hourly_rate

        if labor_hours == 0 or hourly_rate is None:
            labor_sanity = "FAIL"
            score -= 15
            flags.append("labor_missing_hours_or_rate")
        elif hourly_rate < 30 or hourly_rate > 400:
            labor_sanity = "WARN"
            score -= 5
            flags.append("labor_hourly_rate_suspicious")
        else:
            labor_sanity = "OK"

    # =========================
    # 4. TVA logic (doar daca exista date)
    # =========================

    if totals.vat is None:
        if totals.grand_total is None or totals.subtotal_no_vat is None:
            vat_check = "NOT_APPLICABLE"
        else:
            vat_check = "MISSING"
            score -= 5
            flags.append("vat_missing")
    else:
        if totals.subtotal_no_vat is not None:
            expected_vat = float(totals.subtotal_no_vat) * 0.19
            tol = max(2.0, expected_vat * 0.05)
            if abs(expected_vat - float(totals.vat)) > tol:
                vat_check = "INCONSISTENT"
                score -= 10
                flags.append("vat_inconsistent")
            else:
                vat_check = "OK"
        else:
            vat_check = "INSUFFICIENT_DATA"

    # =========================
    # 5. Structura deviz (sanity)
    # =========================

    if len(parts) == 0:
        flags.append("no_parts")
        score -= 20

    if len(items) < 3:
        flags.append("too_few_items")
        score -= 10

    # =========================
    # 6. Verdict final
    # =========================

    score_i = _clamp_score(score)

    if score_i >= 80:
        verdict = "OK"
    elif score_i >= 55:
        verdict = "SUSPICIOUS"
    elif score_i >= 30:
        verdict = "BAD"
    else:
        verdict = "INSUFFICIENT_DATA"

    return DevizInternalResult(
        InternalScore=score_i,
        InternalVerdict=verdict,
        InternalFlags=", ".join(flags) if flags else "",
        LaborSanityResult=labor_sanity,
        MathConsistency=math_consistency,
        VATCheck=vat_check,
        debug=dbg
    )


# =========================
# FASTAPI ENDPOINT (ce face sa apara in /docs)
# =========================

@router.post("/deviz_internal_check", response_model=DevizInternalResult)
def deviz_internal_check_endpoint(payload: DevizInternalInput) -> DevizInternalResult:
    return internal_deviz_check(payload)
