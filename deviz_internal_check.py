from typing import List, Dict, Any, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

# =========================
# INPUT MODELS (compatibile cu main.py)
# =========================

class ExtractedItem(BaseModel):
    desc: str = ""
    kind: str = "part"  # "part" | "labor"
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
        b = float(b)
        if b == 0:
            return None
        return float(a) / b
    except Exception:
        return None

def _clamp_score(x: float) -> int:
    return max(1, min(100, int(round(x))))

def _tol(value: float, abs_min: float, rel: float) -> float:
    # toleranta = max(abs_min, rel * |value|)
    return max(abs_min, abs(value) * rel)

def _approx(a: float, b: float, abs_min: float, rel: float) -> bool:
    return abs(a - b) <= _tol(b, abs_min=abs_min, rel=rel)

def _f(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)

# =========================
# CORE LOGIC
# =========================

def internal_deviz_check(payload: DevizInternalInput) -> DevizInternalResult:
    flags: List[str] = []
    score = 100.0
    dbg: Dict[str, Any] = {}

    items = payload.items or []
    totals = payload.totals or Totals()

    # normalize kinds
    for it in items:
        it.kind = (it.kind or "part").lower().strip()

    parts = [i for i in items if (i.kind or "").lower() == "part"]
    labor = [i for i in items if (i.kind or "").lower() == "labor"]

    dbg["items_count"] = len(items)
    dbg["parts_count"] = len(parts)
    dbg["labor_count"] = len(labor)

    # sums (always)
    sum_parts = sum(float(i.line_total or 0.0) for i in parts)
    sum_labor = sum(float(i.line_total or 0.0) for i in labor)
    sum_all = sum(float(i.line_total or 0.0) for i in items)

    dbg["sum_parts"] = sum_parts
    dbg["sum_labor"] = sum_labor
    dbg["sum_all"] = sum_all

    tm = _f(totals.materials)
    tl = _f(totals.labor)
    tg = _f(totals.grand_total)
    tv = _f(totals.vat)
    tnet = _f(totals.subtotal_no_vat)

    # =========================
    # 0) Detect partial payload (Make iterates / only parts / only labor)
    # =========================
    partial_payload = False
    partial_reasons: List[str] = []

    # classic: totals say there is labor/materials but items list doesn't contain those
    if (tl is not None and tl > 0) and len(labor) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_labor_but_no_labor_items")
    if (tm is not None and tm > 0) and len(parts) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_materials_but_no_part_items")

    # if too few items, totals checks are usually meaningless (common when called per-line from Make)
    if len(items) <= 1:
        partial_payload = True
        partial_reasons.append("too_few_items_for_totals_checks")

    # extra: if grand total exists but sum_all is way smaller, likely you are not sending all items
    if (tg is not None and tg > 0) and (sum_all > 0) and (sum_all < 0.65 * tg):
        partial_payload = True
        partial_reasons.append("sum_all_much_smaller_than_grand_total_likely_missing_items")

    dbg["partial_payload"] = partial_payload
    dbg["partial_reasons"] = partial_reasons

    # =========================
    # 1) Math consistency (per line)
    # =========================
    mismatches = 0
    checked = 0
    missing_price_lines = 0

    for it in items:
        q = float(it.qty or 0.0)
        up = float(it.unit_price or 0.0)
        lt = float(it.line_total or 0.0)

        if q > 0 and up > 0:
            checked += 1
            expected = q * up
            tol_line = max(2.0, expected * 0.05)  # 2 RON sau 5%
            if abs(expected - lt) > tol_line:
                mismatches += 1
        else:
            # lines without usable price inputs
            if (lt or 0.0) > 0 and (q == 0 or up == 0):
                missing_price_lines += 1

    if len(items) < 2:
        math_consistency = "INSUFFICIENT_DATA"
        score -= 5
    else:
        if checked == 0:
            math_consistency = "INSUFFICIENT_DATA"
            score -= 3
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

    if missing_price_lines >= max(2, int(0.35 * max(1, len(items)))):
        flags.append("many_lines_missing_qty_or_unit_price")
        score -= 5

    dbg["line_math_checked"] = checked
    dbg["line_math_mismatches"] = mismatches
    dbg["missing_price_lines"] = missing_price_lines

    # =========================
    # 2) Totals vs sums (IMPORTANT: skip penalties if partial payload)
    # =========================
    if not partial_payload:
        # materials
        if tm is not None and tm > 0 and sum_parts > 0:
            tol_m = _tol(tm, abs_min=5.0, rel=0.05)
            dbg["materials_tol"] = tol_m
            dbg["materials_diff"] = (sum_parts - tm)
            if abs(sum_parts - tm) > tol_m:
                flags.append("materials_total_mismatch")
                score -= 10

        # labor (ONLY penalize if you actually have labor items)
        if tl is not None and tl > 0 and len(labor) > 0:
            tol_l = _tol(tl, abs_min=5.0, rel=0.05)
            dbg["labor_tol"] = tol_l
            dbg["labor_diff"] = (sum_labor - tl)
            if abs(sum_labor - tl) > tol_l:
                flags.append("labor_total_mismatch")
                score -= 10

        # grand total (only if you have enough items to represent whole invoice)
        if tg is not None and tg > 0 and len(items) >= 3 and sum_all > 0:
            tol_g = _tol(tg, abs_min=10.0, rel=0.05)
            dbg["grand_tol"] = tol_g
            dbg["grand_diff"] = (sum_all - tg)
            if abs(sum_all - tg) > tol_g:
                flags.append("grand_total_mismatch")
                score -= 12

        # net sanity: if we have subtotal_no_vat + vat + grand_total, check arithmetic
        if (tnet is not None and tnet > 0) and (tv is not None and tv >= 0) and (tg is not None and tg > 0):
            expected_grand = tnet + tv
            tol_ng = _tol(expected_grand, abs_min=5.0, rel=0.03)
            dbg["net_plus_vat_expected_grand"] = expected_grand
            dbg["net_plus_vat_tol"] = tol_ng
            if abs(expected_grand - tg) > tol_ng:
                flags.append("grand_total_vs_net_plus_vat_mismatch")
                score -= 8
    else:
        # signal only, no penalty
        flags.append("partial_payload")
        if tm is not None and tm > 0 and len(parts) == 0:
            flags.append("partial_payload_skipped_materials_total_check")
        if tl is not None and tl > 0 and len(labor) == 0:
            flags.append("partial_payload_skipped_labor_total_check")
        if tg is not None and tg > 0 and len(items) <= 1:
            flags.append("partial_payload_skipped_grand_total_check")

    # =========================
    # 3) Labor sanity (rate)
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
            score -= 12
            flags.append("labor_missing_hours_or_rate")
        elif hourly_rate < 30 or hourly_rate > 450:
            labor_sanity = "WARN"
            score -= 5
            flags.append("labor_hourly_rate_suspicious")
        else:
            labor_sanity = "OK"

    # =========================
    # 4) VAT logic (incl. inference)
    # =========================
    vat_check = "NOT_APPLICABLE"

    # Case A: explicit net + vat -> validate 19%
    if tv is not None and tnet is not None and tnet > 0:
        expected_vat = tnet * 0.19
        tol_v = _tol(expected_vat, abs_min=2.0, rel=0.05)
        dbg["vat_expected_from_net"] = expected_vat
        dbg["vat_tol"] = tol_v
        dbg["vat_diff"] = (tv - expected_vat)
        if abs(tv - expected_vat) > tol_v:
            vat_check = "INCONSISTENT"
            score -= 10
            flags.append("vat_inconsistent")
        else:
            vat_check = "OK"

    # Case B: net + grand but missing vat -> infer vat = grand - net, validate 19%
    elif tv is None and tnet is not None and tg is not None and tnet > 0 and tg > 0:
        inferred_vat = tg - tnet
        dbg["vat_inferred_from_grand_minus_net"] = inferred_vat
        if inferred_vat > 0:
            expected_vat = tnet * 0.19
            if _approx(inferred_vat, expected_vat, abs_min=5.0, rel=0.08):
                vat_check = "OK"
                flags.append("vat_inferred")
            else:
                vat_check = "INCONSISTENT"
                score -= 6
                flags.append("vat_grand_minus_net_not_19pct")

    # Case C: we only have materials/labor/grand; sometimes materials/labor are already gross (include VAT)
    else:
        net_from_totals = None
        if tm is not None or tl is not None:
            net_from_totals = float(tm or 0.0) + float(tl or 0.0)

        # If net_from_totals ~= grand_total => totals likely already include VAT (common in devize)
        if net_from_totals is not None and tg is not None and net_from_totals > 0 and tg > 0:
            dbg["net_from_materials_plus_labor"] = net_from_totals
            # close enough -> VAT is embedded, can't validate without true net
            if _approx(tg, net_from_totals, abs_min=10.0, rel=0.03):
                vat_check = "NOT_APPLICABLE"
                flags.append("vat_probably_included_in_materials_and_labor_totals")
            else:
                # If grand is ~1.19 * (materials+labor), then materials+labor might be net
                if _approx(tg, net_from_totals * 1.19, abs_min=15.0, rel=0.05):
                    vat_check = "OK"
                    flags.append("vat_inferred_from_grand_vs_net_approx_19pct")
                else:
                    vat_check = "NOT_APPLICABLE"
        else:
            vat_check = "NOT_APPLICABLE"

    # =========================
    # 5) Structure sanity (only if not partial)
    # =========================
    if not partial_payload:
        if len(parts) == 0 and len(items) >= 3:
            flags.append("no_parts")
            score -= 12
        if len(items) < 3:
            flags.append("too_few_items")
            score -= 6

        # suspicious proportions: almost all value is labor or almost all is parts
        total_value = sum_all
        if total_value > 0:
            labor_share = _safe_ratio(sum_labor, total_value)
            dbg["labor_share"] = labor_share
            if labor_share is not None:
                if labor_share > 0.92:
                    flags.append("labor_share_extremely_high")
                    score -= 8
                if labor_share < 0.03 and len(labor) > 0:
                    flags.append("labor_share_extremely_low")
                    score -= 4

    # =========================
    # 6) Verdict
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
