# deviz_internal_check.py
# Goal:
# - no "SKIPPED" anywhere
# - Airtable-friendly single select values: OK | MISMATCH | INSUFFICIENT_DATA
# - do NOT re-invent totals when we already have them from main.py
# - correctly handle net vs gross (subtotal_no_vat + vat = grand_total)

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field


# =========================
# Models (compatible with main.py)
# =========================

SelectMath = Literal["OK", "MISMATCH", "INSUFFICIENT_DATA"]
ItemKind = Literal["part", "labor"]

class DevizInternalItem(BaseModel):
    desc: str = ""
    kind: ItemKind = "part"
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

class DevizInternalCheckResult(BaseModel):
    # Airtable single-select (exact)
    math_consistency: SelectMath = "INSUFFICIENT_DATA"

    # Optional: keep your existing pipeline fields if you already use them
    internal_status: str = "OK"          # OK / SUSPICIOUS / etc (free text if you want)
    internal_score: float = 0.0          # 0..100 (heuristic)
    reasons: List[str] = Field(default_factory=list)  # typologies / flags
    debug: Dict[str, Any] = Field(default_factory=dict)


# =========================
# Helpers
# =========================

def _is_num(x: Optional[float]) -> bool:
    return x is not None

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if not s:
            return None
        s = s.replace(" ", "")
        # allow "1,234.56"
        if "," in s and "." in s:
            s = s.replace(",", "")
            return float(s)
        # allow "123,45"
        if "," in s and "." not in s:
            return float(s.replace(",", "."))
        return float(s)
    except Exception:
        return None

def _sum_items(items: List[DevizInternalItem]) -> Tuple[float, float, float]:
    mat = 0.0
    lab = 0.0
    total = 0.0
    for it in items or []:
        lt = float(it.line_total or 0.0)
        total += lt
        if it.kind == "labor":
            lab += lt
        else:
            mat += lt
    return mat, lab, total

def _approx_equal(a: Optional[float], b: Optional[float], tol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= float(tol)

def _tol_for_amount(x: float) -> float:
    # practical tolerance for OCR/calc
    # <= 1k: 2 lei, <= 10k: 5 lei, else: 10 lei
    ax = abs(float(x))
    if ax <= 1000:
        return 2.0
    if ax <= 10000:
        return 5.0
    return 10.0

def _maybe_infer_vat(subtotal_no_vat: Optional[float], grand_total: Optional[float]) -> Optional[float]:
    if subtotal_no_vat is None or grand_total is None:
        return None
    if subtotal_no_vat <= 0 or grand_total <= 0:
        return None
    vat = grand_total - subtotal_no_vat
    # if negative, ignore
    if vat < -1.0:
        return None
    # if tiny negative due to rounding, clamp
    if vat < 0:
        vat = 0.0
    return float(vat)

def _pick_best_net_total(
    materials: Optional[float],
    labor: Optional[float],
    subtotal_no_vat: Optional[float],
    sum_items_total: float
) -> Tuple[Optional[float], str]:
    """
    Decide what is the best representation of NET total (fara TVA).
    Priority:
      1) subtotal_no_vat (explicit)
      2) materials + labor (explicit)
      3) sum_items_total (from lines)
    """
    if _is_num(subtotal_no_vat) and (subtotal_no_vat or 0) > 0:
        return float(subtotal_no_vat), "subtotal_no_vat"
    if _is_num(materials) and _is_num(labor) and (materials or 0) + (labor or 0) > 0:
        return float(materials or 0) + float(labor or 0), "materials_plus_labor"
    if sum_items_total > 0:
        return float(sum_items_total), "sum_items_total"
    return None, "none"


# =========================
# Core check
# =========================

def internal_deviz_check(payload: DevizInternalInput) -> DevizInternalCheckResult:
    totals = payload.totals or DevizInternalTotals()
    items = payload.items or []

    # normalize totals (in case dict was passed in)
    if isinstance(totals, dict):
        totals = DevizInternalTotals(**totals)
    if items and isinstance(items[0], dict):
        items = [DevizInternalItem(**x) for x in items]

    # sum items
    sum_mat, sum_lab, sum_items_total = _sum_items(items)

    # extract totals
    materials = _safe_float(totals.materials)
    labor = _safe_float(totals.labor)
    subtotal_no_vat = _safe_float(totals.subtotal_no_vat)
    vat = _safe_float(totals.vat)
    grand_total = _safe_float(totals.grand_total)

    reasons: List[str] = []
    debug: Dict[str, Any] = {
        "totals_in": {
            "materials": materials,
            "labor": labor,
            "subtotal_no_vat": subtotal_no_vat,
            "vat": vat,
            "grand_total": grand_total,
            "currency": totals.currency or "RON",
        },
        "items_sum": {
            "materials_from_items": round(sum_mat, 2),
            "labor_from_items": round(sum_lab, 2),
            "sum_items_total": round(sum_items_total, 2),
            "items_count": len(items),
        },
    }

    # basic sufficiency
    any_total_present = any(_is_num(x) and (x or 0) > 0 for x in [materials, labor, subtotal_no_vat, vat, grand_total])
    any_items_present = len(items) > 0 and (sum_items_total > 0)

    if not any_total_present and not any_items_present:
        return DevizInternalCheckResult(
            math_consistency="INSUFFICIENT_DATA",
            internal_status="OK",
            internal_score=0.0,
            reasons=["insufficient_data_no_totals_no_items"],
            debug=debug
        )

    # Pick best NET total (fara TVA)
    net_total, net_source = _pick_best_net_total(materials, labor, subtotal_no_vat, sum_items_total)
    debug["net_total"] = {"value": net_total, "source": net_source}

    # Determine VAT:
    # - prefer explicit vat
    # - else infer if we have grand_total and net_total
    vat_final = vat if (_is_num(vat) and (vat or 0) >= 0) else _maybe_infer_vat(net_total, grand_total)
    debug["vat_final"] = vat_final

    # Determine expected gross:
    gross_expected = None
    gross_source = "none"
    if net_total is not None and vat_final is not None:
        gross_expected = net_total + vat_final
        gross_source = "net_plus_vat"
    debug["gross_expected"] = {"value": gross_expected, "source": gross_source}

    # -------------------------
    # Consistency checks
    # -------------------------

    mismatches = 0

    # 1) If we have explicit materials and labor, verify they add up to explicit subtotal_no_vat (if present)
    if _is_num(materials) and _is_num(labor) and _is_num(subtotal_no_vat):
        tol = _tol_for_amount(subtotal_no_vat or 0)
        if not _approx_equal((materials or 0) + (labor or 0), subtotal_no_vat, tol=tol):
            mismatches += 1
            reasons.append("net_mismatch_materials_plus_labor_vs_subtotal_no_vat")

    # 2) If we have items, verify their sums roughly match declared materials/labor (when declared)
    if len(items) > 0:
        if _is_num(materials) and (materials or 0) > 0:
            tol = _tol_for_amount(materials or 0)
            if not _approx_equal(sum_mat, materials, tol=tol):
                mismatches += 1
                reasons.append("materials_mismatch_vs_items_sum")
        if _is_num(labor) and (labor or 0) > 0:
            tol = _tol_for_amount(labor or 0)
            if not _approx_equal(sum_lab, labor, tol=tol):
                mismatches += 1
                reasons.append("labor_mismatch_vs_items_sum")

    # 3) If we have grand_total and we can compute expected gross, verify it
    if _is_num(grand_total) and (grand_total or 0) > 0 and gross_expected is not None:
        tol = _tol_for_amount(grand_total or 0)
        if not _approx_equal(gross_expected, grand_total, tol=tol):
            mismatches += 1
            reasons.append("gross_mismatch_net_plus_vat_vs_grand_total")

    # 4) If we have grand_total but no vat/subtotal to reconcile,
    #    then ONLY compare net_total with grand_total when VAT is clearly absent (we cannot know here),
    #    so treat as insufficient (not mismatch) unless difference is tiny.
    if _is_num(grand_total) and (grand_total or 0) > 0 and gross_expected is None:
        # If net_total exists and is close, OK. If far, we just don't know (could be VAT included).
        if net_total is not None:
            tol = _tol_for_amount(grand_total or 0)
            if _approx_equal(net_total, grand_total, tol=tol):
                pass
            else:
                reasons.append("cannot_reconcile_grand_total_without_vat_or_subtotal")
                # do not count as mismatch, data is ambiguous

    # -------------------------
    # Decide Airtable value (no SKIPPED)
    # -------------------------

    # If we only have ambiguous info (grand_total but no vat/subtotal and no declared m/l),
    # mark INSUFFICIENT_DATA instead of mismatch.
    ambiguous_only = (
        mismatches == 0
        and any(r == "cannot_reconcile_grand_total_without_vat_or_subtotal" for r in reasons)
        and not any(r.endswith("_mismatch_vs_items_sum") or r.startswith("net_mismatch") or r.startswith("gross_mismatch") for r in reasons)
    )

    if mismatches > 0:
        math_consistency: SelectMath = "MISMATCH"
    else:
        if ambiguous_only:
            math_consistency = "INSUFFICIENT_DATA"
        else:
            # if we have at least one strong reconciliation path, consider OK
            has_strong_path = (
                (_is_num(subtotal_no_vat) and (subtotal_no_vat or 0) > 0)
                or (_is_num(materials) and _is_num(labor) and (materials or 0) + (labor or 0) > 0)
                or (len(items) > 0 and sum_items_total > 0)
            )
            math_consistency = "OK" if has_strong_path else "INSUFFICIENT_DATA"

    # score (simple, stable)
    # start from 100, subtract per mismatch; clamp
    score = 100.0
    score -= 35.0 * float(mismatches)
    if math_consistency == "INSUFFICIENT_DATA":
        score = min(score, 40.0)
    score = max(0.0, min(100.0, score))

    debug["result"] = {
        "mismatches": mismatches,
        "math_consistency": math_consistency,
        "score": score,
        "reasons": reasons[:],
    }

    # internal_status: keep lightweight
    internal_status = "OK" if math_consistency == "OK" else "SUSPICIOUS"

    return DevizInternalCheckResult(
        math_consistency=math_consistency,
        internal_status=internal_status,
        internal_score=round(score, 2),
        reasons=reasons,
        debug=debug
    )
