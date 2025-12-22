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
        b = float(b)
        if b == 0:
            return None
        return float(a) / b
    except Exception:
        return None

def _clamp_score(x: float) -> int:
    return max(1, min(100, int(round(x))))

def _approx(a: float, b: float, tol_abs: float, tol_rel: float) -> bool:
    return abs(a - b) <= max(tol_abs, abs(b) * tol_rel)

def _sum_if_present(vals: List[Optional[float]]) -> Optional[float]:
    # returneaza suma doar daca exista cel putin o valoare nenula
    ok = [v for v in vals if v is not None]
    if not ok:
        return None
    return float(sum(float(v) for v in ok))

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
    # 0) Detect partial payload (important pt Make logs per-line)
    # =========================
    partial_payload = False
    partial_reasons: List[str] = []

    # semnal puternic: totals cer comparatie dar items lipsesc pe categoria respectiva
    if (totals.labor is not None and float(totals.labor) > 0) and len(labor) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_labor_but_no_labor_items")

    if (totals.materials is not None and float(totals.materials) > 0) and len(parts) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_materials_but_no_part_items")

    # foarte putine items => probabil rulezi check-ul pe o singura linie
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

    # tolerante un pic mai stricte ca sa nu fie totul "OK" (dar tot safe pt OCR)
    def _tol_total(x: float, base_abs: float, base_rel: float) -> float:
        return max(base_abs, abs(float(x)) * base_rel)

    if not partial_payload:
        if totals.materials is not None:
            tv = float(totals.materials)
            if abs(sum_parts - tv) > _tol_total(tv, base_abs=5.0, base_rel=0.04):
                flags.append("materials_total_mismatch")
                score -= 12

        if totals.labor is not None:
            tv = float(totals.labor)
            if abs(sum_labor - tv) > _tol_total(tv, base_abs=5.0, base_rel=0.04):
                flags.append("labor_total_mismatch")
                score -= 12

        if totals.grand_total is not None:
            tv = float(totals.grand_total)
            if abs(sum_all - tv) > _tol_total(tv, base_abs=10.0, base_rel=0.03):
                flags.append("grand_total_mismatch")
                score -= 18
    else:
        # doar semnalam, fara penalty
        if totals.materials is not None and float(totals.materials) > 0 and len(parts) == 0:
            flags.append("partial_payload_skipped_materials_total_check")
        if totals.labor is not None and float(totals.labor) > 0 and len(labor) == 0:
            flags.append("partial_payload_skipped_labor_total_check")
        if totals.grand_total is not None and float(totals.grand_total) > 0 and len(items) <= 1:
            flags.append("partial_payload_skipped_grand_total_check")

    # =========================
    # 2b) Net vs Total sanity (intra si cand nu ai subtotal_no_vat)
    # =========================
    # net_from_totals = materials + labor (daca exista macar una)
    net_from_totals = _sum_if_present([totals.materials, totals.labor])
    dbg["net_from_totals"] = net_from_totals

    if not partial_payload and totals.grand_total is not None and net_from_totals is not None:
        gt = float(totals.grand_total)
        net = float(net_from_totals)
        if gt > 0 and net > 0:
            ratio = gt / net
            dbg["grand_over_net_ratio"] = ratio
            # cu TVA (19%) ar trebui sa fie ~1.19; fara TVA ~1.00
            if ratio < 0.93 or ratio > 1.35:
                flags.append("grand_total_vs_net_suspicious")
                score -= 10

    # =========================
    # 3) Labor sanity
    # =========================
    if len(labor) == 0:
        labor_sanity = "SKIPPED"
        dbg["labor_hours"] = 0.0
        dbg["labor_value"] = 0.0
        dbg["labor_hourly_rate"] = None
    else:
        labor_hours = sum(float(i.qty or 0.0) for i in labor if float(i.qty or 0.0) > 0)
        labor_value = sum(float(i.line_total or 0.0) for i in labor if float(i.line_total or 0.0) > 0)
        hourly_rate = _safe_ratio(labor_value, labor_hours)

        dbg["labor_hours"] = labor_hours
        dbg["labor_value"] = labor_value
        dbg["labor_hourly_rate"] = hourly_rate

        if labor_hours == 0 or hourly_rate is None:
            labor_sanity = "FAIL"
            score -= 18
            flags.append("labor_missing_hours_or_rate")
        elif hourly_rate < 40 or hourly_rate > 450:
            # praguri un pic mai realiste (service-uri sar usor de 30)
            labor_sanity = "WARN"
            score -= 8
            flags.append("labor_hourly_rate_suspicious")
        else:
            labor_sanity = "OK"

    # =========================
    # 4) TVA logic (+ infer daca putem)
    # =========================
    vat_check = "NOT_APPLICABLE"

    # (A) avem TVA si net
    if totals.vat is not None and totals.subtotal_no_vat is not None:
        expected_vat = float(totals.subtotal_no_vat) * 0.19
        tol = max(2.0, expected_vat * 0.05)
        if abs(expected_vat - float(totals.vat)) > tol:
            vat_check = "INCONSISTENT"
            score -= 12
            flags.append("vat_inconsistent")
        else:
            vat_check = "OK"

    # (B) n-avem TVA explicit
    elif totals.vat is None:
        # daca avem net + total, TVA lipseste in campuri (dar putem valida)
        if totals.grand_total is not None and totals.subtotal_no_vat is not None:
            vat_check = "MISSING"
            score -= 6
            flags.append("vat_missing")
        else:
            # infer: daca avem grand_total si un "net" plauzibil (prefer subtotal_no_vat, altfel materials+labor)
            base_net = None
            if totals.subtotal_no_vat is not None and float(totals.subtotal_no_vat) > 0:
                base_net = float(totals.subtotal_no_vat)
                dbg["vat_infer_base"] = "subtotal_no_vat"
            elif net_from_totals is not None and float(net_from_totals) > 0:
                base_net = float(net_from_totals)
                dbg["vat_infer_base"] = "materials_plus_labor"
            else:
                base_net = None

            if base_net is not None and totals.grand_total is not None and float(totals.grand_total) > 0:
                inferred_vat = float(totals.grand_total) - base_net
                dbg["vat_inferred_amount"] = inferred_vat

                # acceptam TVA daca seamana cu 19% din baza
                if inferred_vat > 0 and _approx(inferred_vat, base_net * 0.19, tol_abs=5.0, tol_rel=0.08):
                    vat_check = "OK"
                    flags.append("vat_inferred")
                else:
                    vat_check = "NOT_APPLICABLE"
            else:
                vat_check = "NOT_APPLICABLE"

    # (C) totals.vat exista dar n-avem baza net
    else:
        vat_check = "INSUFFICIENT_DATA"

    # =========================
    # 5) Structura deviz (sanity)
    # =========================
    # daca payload e partial, nu penalizam "no_parts/too_few_items"
    if not partial_payload:
        if len(parts) == 0:
            flags.append("no_parts")
            score -= 22
        if len(items) < 3:
            flags.append("too_few_items")
            score -= 12
        # si un semnal simplu: daca ai doar 1 tip de items, e dubios pt devize reale
        if len(parts) == 0 and len(labor) > 0:
            flags.append("only_labor_items")
            score -= 8
        if len(labor) == 0 and len(parts) > 0 and (totals.labor is not None and float(totals.labor) > 0):
            flags.append("labor_totals_present_but_no_labor_items")
            score -= 10
    else:
        flags.append("partial_payload")

    # =========================
    # 6) Verdict final
    # =========================
    score_i = _clamp_score(score)

    if score_i >= 82:
        verdict = "OK"
    elif score_i >= 60:
        verdict = "SUSPICIOUS"
    elif score_i >= 35:
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
