from typing import List, Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field
import math

router = APIRouter()

# =========================
# INPUT MODELS (compatibile cu main.py)
# =========================

class ExtractedItem(BaseModel):
    desc: str = ""
    kind: str = "part"            # "part" | "labor"
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
    items: List[ExtractedItem] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)

# =========================
# OUTPUT MODEL (Airtable-ready)
# =========================

class DevizInternalResult(BaseModel):
    InternalScore: int
    InternalVerdict: str
    InternalFlags: str
    LaborSanityResult: str
    MathConsistency: str
    VATCheck: str
    debug: Dict[str, Any] = Field(default_factory=dict)

# =========================
# HELPERS
# =========================

def _safe_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        if b == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None

def _clamp_score(x: float) -> int:
    return max(1, min(100, int(round(x))))

def _approx(a: float, b: float, tol_abs: float, tol_rel: float) -> bool:
    # True daca |a-b| <= max(tol_abs, |b|*tol_rel)
    return abs(a - b) <= max(tol_abs, abs(b) * tol_rel)

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

    dbg["items_count"] = len(items)
    dbg["parts_count"] = len(parts)
    dbg["labor_count"] = len(labor)

    # =========================
    # 0) Detect partial payload (foarte important pt Make logs per-line)
    # =========================
    partial_payload = False
    partial_reasons: List[str] = []

    if totals.labor is not None and totals.labor > 0 and len(labor) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_labor_but_no_labor_items")

    if totals.materials is not None and totals.materials > 0 and len(parts) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_materials_but_no_part_items")

    # daca ai foarte putine items, e foarte probabil ca rulezi check-ul pe o singura linie
    if len(items) <= 1:
        partial_payload = True
        partial_reasons.append("too_few_items_for_totals_checks")

    dbg["partial_payload"] = partial_payload
    dbg["partial_reasons"] = partial_reasons

    # =========================
    # 1) Math consistency (linie)
    # =========================
    mismatches = 0
    checked = 0

    for it in items:
        q = float(it.qty or 0.0)
        up = float(it.unit_price or 0.0)
        lt = float(it.line_total or 0.0)
        if q > 0 and up > 0:
            checked += 1
            expected = q * up
            tol = max(2.0, expected * 0.05)  # 2 RON sau 5%
            if abs(expected - lt) > tol:
                mismatches += 1

    if len(items) < 2:
        math_consistency = "INSUFFICIENT_DATA"
        score -= 10
    else:
        if checked == 0:
            math_consistency = "INSUFFICIENT_DATA"
            score -= 5
        elif mismatches == 0:
            math_consistency = "OK"
        elif mismatches <= 2:
            math_consistency = "MISMATCH"
            score -= mismatches * 5
            flags.append("line_math_mismatch")
        else:
            math_consistency = "MISMATCH"
            score -= 15
            flags.append("multiple_line_math_mismatches")

    dbg["line_math_checked"] = checked
    dbg["line_math_mismatches"] = mismatches

    # =========================
    # 2) Totaluri vs suma linii
    #    IMPORTANT: daca payload e partial, nu penalizam mismatch-urile astea
    # =========================
    sum_parts = sum(float(i.line_total or 0.0) for i in parts)
    sum_labor = sum(float(i.line_total or 0.0) for i in labor)
    sum_all = sum(float(i.line_total or 0.0) for i in items)

    dbg["sum_parts"] = sum_parts
    dbg["sum_labor"] = sum_labor
    dbg["sum_all"] = sum_all

    if not partial_payload:
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
    else:
        # doar semnalam, fara penalty
        if totals.materials is not None and totals.materials > 0 and len(parts) == 0:
            flags.append("partial_payload_skipped_materials_total_check")
        if totals.labor is not None and totals.labor > 0 and len(labor) == 0:
            flags.append("partial_payload_skipped_labor_total_check")
        if totals.grand_total is not None and totals.grand_total > 0 and len(items) <= 1:
            flags.append("partial_payload_skipped_grand_total_check")

    # =========================
    # 3) Labor sanity
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
    # 4) TVA logic (+ infer daca putem)
    # =========================
    vat_check = "NOT_APPLICABLE"

    if totals.vat is not None and totals.subtotal_no_vat is not None:
        expected_vat = float(totals.subtotal_no_vat) * 0.19
        tol = max(2.0, expected_vat * 0.05)
        if abs(expected_vat - float(totals.vat)) > tol:
            vat_check = "INCONSISTENT"
            score -= 10
            flags.append("vat_inconsistent")
        else:
            vat_check = "OK"

    elif totals.vat is None:
        # daca avem net + total, TVA lipseste
        if totals.grand_total is not None and totals.subtotal_no_vat is not None:
            vat_check = "MISSING"
            score -= 5
            flags.append("vat_missing")
        else:
            # incearca infer: daca avem (materials+labor) si grand_total
            net = None
            if totals.materials is not None or totals.labor is not None:
                net = float(totals.materials or 0.0) + float(totals.labor or 0.0)

            if net is not None and totals.grand_total is not None and net > 0 and float(totals.grand_total) > 0:
                inferred_vat = float(totals.grand_total) - net
                dbg["vat_inferred_from_grand_minus_net"] = inferred_vat

                # valid doar daca seamana cu 19% din net (cu toleranta)
                if inferred_vat > 0 and _approx(inferred_vat, net * 0.19, tol_abs=5.0, tol_rel=0.08):
                    vat_check = "OK"
                    flags.append("vat_inferred")
                else:
                    vat_check = "NOT_APPLICABLE"
            else:
                vat_check = "NOT_APPLICABLE"

    else:
        # totals.vat exista dar nu avem net
        vat_check = "INSUFFICIENT_DATA"

    # =========================
    # 5) Structura deviz (sanity)
    # =========================
    # aici iar: daca payload e partial, nu penalizam "no_parts/too_few_items"
    if not partial_payload:
        if len(parts) == 0:
            flags.append("no_parts")
            score -= 20
        if len(items) < 3:
            flags.append("too_few_items")
            score -= 10
    else:
        flags.append("partial_payload")

    # =========================
    # 6) Verdict final
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
# FASTAPI ENDPOINT (apare in /docs)
# =========================

@router.post("/deviz_internal_check", response_model=DevizInternalResult)
def deviz_internal_check_endpoint(payload: DevizInternalInput) -> DevizInternalResult:
    return internal_deviz_check(payload)
