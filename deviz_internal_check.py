# deviz_internal_check.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# Airtable-safe fixed enums (DO NOT CHANGE STRINGS)
MathStatus = Literal["OK", "MISMATCH", "INSUFFICIENT_DATA"]
InternalVerdict = Literal["OK", "SUSPICIOUS", "BAD", "INSUFFICIENT_DATA"]


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


class DevizInternalOutput(BaseModel):
    # Airtable-safe fixed enums
    math_consistency: MathStatus = "INSUFFICIENT_DATA"
    internal_verdict: InternalVerdict = "INSUFFICIENT_DATA"

    # tipologii
    typology_flags: List[str] = Field(default_factory=list)
    typology_score: float = 0.0  # 0..100
    confidence: Literal["LOW", "MED", "HIGH"] = "LOW"

    # details
    reasons: List[str] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)


# -------------------------
# helpers
# -------------------------

def _nz(x: Optional[float]) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v
    except Exception:
        return None

def _sum_items(items: List[DevizInternalItem], kind: Optional[str] = None) -> float:
    s = 0.0
    for it in items or []:
        if kind and it.kind != kind:
            continue
        try:
            s += float(it.line_total or 0.0)
        except Exception:
            pass
    return float(s)

def _count_items(items: List[DevizInternalItem], kind: Optional[str] = None) -> int:
    if not items:
        return 0
    if not kind:
        return len(items)
    return sum(1 for it in items if it.kind == kind)

def _abs(x: float) -> float:
    return x if x >= 0 else -x

def _approx(a: Optional[float], b: Optional[float], tol_abs: float = 2.0, tol_rel: float = 0.02) -> bool:
    if a is None or b is None:
        return False
    da = _abs(float(a) - float(b))
    if da <= tol_abs:
        return True
    denom = max(1.0, _abs(float(b)))
    return (da / denom) <= tol_rel

def _ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    try:
        if a is None or b is None:
            return None
        b = float(b)
        if b == 0:
            return None
        return float(a) / b
    except Exception:
        return None

def _norm_desc(s: str) -> str:
    s = (s or "").strip().lower()
    # normalizeaza spatii
    s = " ".join(s.split())
    return s


# -------------------------
# math consistency (uses main.py totals/items, no parsing)
# -------------------------

def _math_consistency(payload: DevizInternalInput) -> Dict[str, Any]:
    items = payload.items or []
    t = payload.totals or DevizInternalTotals()

    gt = _nz(t.grand_total)
    mat = _nz(t.materials)
    lab = _nz(t.labor)
    vat = _nz(t.vat)
    sub_no_vat = _nz(t.subtotal_no_vat)

    sum_all = _sum_items(items)
    sum_parts = _sum_items(items, "part")
    sum_labor = _sum_items(items, "labor")

    checks: List[Dict[str, Any]] = []
    reasons: List[str] = []
    status: MathStatus = "INSUFFICIENT_DATA"

    # we define "have_lines" as: at least one non-zero line_total
    have_lines = _abs(sum_all) > 0.01

    # check 1: m + l = gt
    if gt is not None and mat is not None and lab is not None:
        expected = float(mat) + float(lab)
        ok = _approx(expected, gt, tol_abs=5.0, tol_rel=0.01)
        checks.append({
            "name": "grand_total_vs_materials_plus_labor",
            "expected": expected,
            "actual": gt,
            "ok": ok,
        })
        if not ok:
            reasons.append("totals_mismatch_gt_vs_m_plus_l")
            status = "MISMATCH"
        else:
            status = "OK"

    # check 2: totals.materials vs sum(part lines)
    if mat is not None and have_lines:
        ok = _approx(sum_parts, mat, tol_abs=5.0, tol_rel=0.02)
        checks.append({
            "name": "materials_total_vs_sum_part_lines",
            "expected": sum_parts,
            "actual": mat,
            "ok": ok,
        })
        if not ok:
            reasons.append("materials_total_mismatch")
            status = "MISMATCH" if status != "OK" else "MISMATCH"
        else:
            if status == "INSUFFICIENT_DATA":
                status = "OK"

    # check 3: totals.labor vs sum(labor lines)
    if lab is not None and have_lines:
        ok = _approx(sum_labor, lab, tol_abs=5.0, tol_rel=0.02)
        checks.append({
            "name": "labor_total_vs_sum_labor_lines",
            "expected": sum_labor,
            "actual": lab,
            "ok": ok,
        })
        if not ok:
            reasons.append("labor_total_mismatch")
            status = "MISMATCH" if status != "OK" else "MISMATCH"
        else:
            if status == "INSUFFICIENT_DATA":
                status = "OK"

    # check 4: sub_no_vat + vat = grand_total
    if gt is not None and sub_no_vat is not None:
        if vat is not None:
            expected = float(sub_no_vat) + float(vat)
            ok = _approx(expected, gt, tol_abs=3.0, tol_rel=0.01)
            checks.append({
                "name": "grand_total_vs_subtotal_plus_vat",
                "expected": expected,
                "actual": gt,
                "ok": ok,
            })
            if not ok:
                reasons.append("grand_total_mismatch_subtotal_plus_vat")
                status = "MISMATCH"
            else:
                if status == "INSUFFICIENT_DATA":
                    status = "OK"
        else:
            # we can still sanity-check implied vat
            implied_vat = float(gt) - float(sub_no_vat)
            checks.append({
                "name": "implied_vat_from_grand_minus_subtotal",
                "implied_vat": implied_vat,
                "ok": True,  # informational
            })
            if status == "INSUFFICIENT_DATA":
                status = "OK"

    # check 5: grand_total vs sum(lines) if both present
    if gt is not None and have_lines:
        ok = _approx(sum_all, gt, tol_abs=10.0, tol_rel=0.02)
        checks.append({
            "name": "grand_total_vs_sum_all_lines",
            "expected": sum_all,
            "actual": gt,
            "ok": ok,
        })
        if not ok:
            reasons.append("grand_total_vs_items_sum_suspicious")
            status = "MISMATCH"

    # if we did zero meaningful checks => insufficient data
    meaningful = [c for c in checks if c.get("name") in (
        "grand_total_vs_materials_plus_labor",
        "materials_total_vs_sum_part_lines",
        "labor_total_vs_sum_labor_lines",
        "grand_total_vs_subtotal_plus_vat",
        "grand_total_vs_sum_all_lines",
    )]
    if len(meaningful) == 0:
        status = "INSUFFICIENT_DATA"
        if not reasons:
            reasons.append("not_enough_data_for_math_checks")

    # severity hint (used by verdict)
    severity = 0.0
    if gt is not None and have_lines:
        delta = _abs(float(gt) - float(sum_all))
        severity = max(severity, delta)

    return {
        "status": status,
        "reasons": reasons,
        "checks": checks,
        "stats": {
            "sum_all_lines": sum_all,
            "sum_part_lines": sum_parts,
            "sum_labor_lines": sum_labor,
            "have_lines": have_lines,
            "grand_total": gt,
            "materials": mat,
            "labor": lab,
            "vat": vat,
            "subtotal_no_vat": sub_no_vat,
            "max_abs_delta_hint": severity,
        }
    }


# -------------------------
# typologies (fraud patterns)
# -------------------------

def _typologies(payload: DevizInternalInput, math_info: Dict[str, Any]) -> Dict[str, Any]:
    items = payload.items or []
    t = payload.totals or DevizInternalTotals()

    gt = _nz(t.grand_total)
    mat = _nz(t.materials)
    lab = _nz(t.labor)
    vat = _nz(t.vat)
    sub_no_vat = _nz(t.subtotal_no_vat)

    sum_all = float(math_info.get("stats", {}).get("sum_all_lines") or 0.0)
    sum_parts = float(math_info.get("stats", {}).get("sum_part_lines") or 0.0)
    sum_labor = float(math_info.get("stats", {}).get("sum_labor_lines") or 0.0)
    have_lines = bool(math_info.get("stats", {}).get("have_lines"))

    flags: List[str] = []
    reasons: List[str] = []
    score = 0.0

    # T1: labor value hidden (lots of labor lines with zero pricing, but labor total exists)
    labor_items = [it for it in items if it.kind == "labor"]
    labor_zero_pricing = [
        it for it in labor_items
        if (float(it.line_total or 0.0) <= 0.0) or (float(it.unit_price or 0.0) <= 0.0)
    ]
    if len(labor_items) >= 2 and len(labor_zero_pricing) / max(1, len(labor_items)) >= 0.6 and (lab or 0.0) >= 100.0:
        flags.append("labor_value_hidden_in_vague_descriptions")
        score += 18.0
        reasons.append("labor lines missing unit_price/line_total while labor total is significant")

    # T2: totals mismatch is itself a typology (rely on math_info reasons)
    if "grand_total_vs_items_sum_suspicious" in (math_info.get("reasons") or []):
        flags.append("grand_total_vs_items_sum_suspicious")
        score += 22.0

    if "labor_total_mismatch" in (math_info.get("reasons") or []):
        flags.append("labor_total_mismatch")
        score += 18.0

    if "materials_total_mismatch" in (math_info.get("reasons") or []):
        flags.append("materials_total_mismatch")
        score += 14.0

    # T3: VAT inferred from grand vs net approx 19% (only if we can infer)
    # If we have (grand_total and subtotal_no_vat), check implied vat rate.
    if gt is not None and sub_no_vat is not None and sub_no_vat > 0:
        implied_vat = float(gt) - float(sub_no_vat)
        rate = _ratio(implied_vat, sub_no_vat)
        if rate is not None and 0.16 <= rate <= 0.22 and vat is None:
            flags.append("vat_inferred_from_grand_vs_net_approx_19pct")
            score += 8.0

    # T4: too many round numbers (unit prices / line totals)
    # Heuristic: if a lot of unit_price are multiples of 10 or 100, its suspicious-ish (low weight)
    def is_round(v: float) -> bool:
        # multiples of 10 or 100 (treat decimals)
        if v <= 0:
            return False
        if abs(v - round(v)) < 1e-6:  # integer
            iv = int(round(v))
            return (iv % 10 == 0) or (iv % 100 == 0)
        return False

    prices = [float(it.unit_price or 0.0) for it in items if float(it.unit_price or 0.0) > 0]
    if len(prices) >= 4:
        round_cnt = sum(1 for p in prices if is_round(p))
        if round_cnt / max(1, len(prices)) >= 0.75:
            flags.append("suspicious_rounding_pattern")
            score += 6.0

    # T5: duplicates / near duplicates
    seen: Dict[str, int] = {}
    for it in items:
        k = f"{it.kind}:{_norm_desc(it.desc)}"
        if len(k) < 8:
            continue
        seen[k] = seen.get(k, 0) + 1
    dup = [k for k, c in seen.items() if c >= 2]
    if len(dup) >= 2:
        flags.append("duplicate_or_repeated_lines")
        score += 10.0

    # T6: missing detail density
    # many parts without qty/unit_price or line_total (line_total==0)
    part_items = [it for it in items if it.kind == "part"]
    part_missing = [
        it for it in part_items
        if float(it.qty or 0.0) <= 0.0 or float(it.unit_price or 0.0) <= 0.0 or float(it.line_total or 0.0) <= 0.0
    ]
    if len(part_items) >= 3 and (len(part_missing) / max(1, len(part_items)) >= 0.5):
        flags.append("missing_unit_prices_or_totals_for_parts")
        score += 10.0

    # normalize score to 0..100
    score = max(0.0, min(100.0, float(score)))

    # confidence heuristic
    # higher confidence if we have totals + enough lines
    conf = "LOW"
    if (gt is not None or (mat is not None and lab is not None)) and _count_items(items) >= 6:
        conf = "HIGH" if score >= 40.0 else "MED"
    elif (gt is not None or have_lines) and _count_items(items) >= 3:
        conf = "MED"

    return {
        "flags": flags,
        "score": score,
        "confidence": conf,
        "reasons": reasons,
        "stats": {
            "items_count": _count_items(items),
            "parts_count": _count_items(items, "part"),
            "labor_count": _count_items(items, "labor"),
            "sum_all_lines": sum_all,
            "sum_part_lines": sum_parts,
            "sum_labor_lines": sum_labor,
            "grand_total": gt,
            "materials": mat,
            "labor": lab,
            "vat": vat,
            "subtotal_no_vat": sub_no_vat,
        }
    }


# -------------------------
# verdict logic (combines math + typologies)
# -------------------------

def _internal_verdict(math_status: MathStatus, math_info: Dict[str, Any], ty: Dict[str, Any]) -> Dict[str, Any]:
    score = float(ty.get("score") or 0.0)
    flags = ty.get("flags") or []
    max_abs_delta = float((math_info.get("stats") or {}).get("max_abs_delta_hint") or 0.0)

    # decide data sufficiency
    # if math is insufficient AND typologies weak => insufficient
    if math_status == "INSUFFICIENT_DATA" and score < 20.0:
        return {
            "verdict": "INSUFFICIENT_DATA",
            "reasons": ["insufficient_data_math_and_typologies_weak"]
        }

    # strong mismatch -> BAD
    # severe mismatch: delta >= 50 OR relative big mismatch signal
    severe_mismatch = False
    if math_status == "MISMATCH":
        if max_abs_delta >= 50.0:
            severe_mismatch = True
        # also if we have explicit grand_total mismatch in reasons, treat as severe if score also high
        if "grand_total_vs_items_sum_suspicious" in (math_info.get("reasons") or []) and max_abs_delta >= 25.0:
            severe_mismatch = True

    if severe_mismatch and score >= 30.0:
        return {
            "verdict": "BAD",
            "reasons": ["severe_math_mismatch_and_suspicious_typologies"]
        }
    if severe_mismatch:
        return {
            "verdict": "BAD",
            "reasons": ["severe_math_mismatch"]
        }

    # SUSPICIOUS rules
    if math_status == "MISMATCH":
        return {
            "verdict": "SUSPICIOUS",
            "reasons": ["math_mismatch"]
        }

    # typology-driven suspicion
    if score >= 55.0:
        return {
            "verdict": "SUSPICIOUS",
            "reasons": ["typology_score_high"]
        }

    # some flags are stronger triggers even with moderate score
    strong_flags = {
        "labor_value_hidden_in_vague_descriptions",
        "grand_total_vs_items_sum_suspicious",
        "labor_total_mismatch",
        "materials_total_mismatch",
    }
    if any(f in strong_flags for f in flags) and score >= 35.0:
        return {
            "verdict": "SUSPICIOUS",
            "reasons": ["strong_typology_flags_present"]
        }

    return {
        "verdict": "OK",
        "reasons": []
    }


# -------------------------
# public API
# -------------------------

def internal_deviz_check(payload: DevizInternalInput) -> DevizInternalOutput:
    # 1) math
    math_info = _math_consistency(payload)
    math_status: MathStatus = math_info["status"]

    # 2) typologies
    ty = _typologies(payload, math_info)

    # 3) internal verdict
    v = _internal_verdict(math_status, math_info, ty)

    out = DevizInternalOutput(
        math_consistency=math_status,
        internal_verdict=v["verdict"],

        typology_flags=list(ty.get("flags") or []),
        typology_score=float(ty.get("score") or 0.0),
        confidence=ty.get("confidence") or "LOW",

        reasons=(math_info.get("reasons") or []) + (ty.get("reasons") or []) + (v.get("reasons") or []),
        debug={
            "math": {
                "checks": math_info.get("checks") or [],
                "stats": math_info.get("stats") or {},
            },
            "typologies": {
                "stats": ty.get("stats") or {},
            },
        }
    )
    return out
