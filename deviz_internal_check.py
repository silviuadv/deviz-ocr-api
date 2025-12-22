from typing import List, Dict, Any, Optional, Tuple
from fastapi import APIRouter
from pydantic import BaseModel, Field
import math
import re
from statistics import median

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
        if b <= 0:
            return None
        return float(a) / b
    except Exception:
        return None

def _clamp_score(x: float) -> int:
    return max(1, min(100, int(round(x))))

def _f(x: Optional[float]) -> Optional[float]:
    return None if x is None else float(x)

def _tol(value: float, abs_min: float, rel: float) -> float:
    return max(abs_min, abs(value) * rel)

def _approx(a: float, b: float, abs_min: float, rel: float) -> bool:
    return abs(a - b) <= _tol(b, abs_min=abs_min, rel=rel)

def _norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9ăâîșț\-\._/ ]+", " ", s)  # ok si cu/sau fara diacritice
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _tokenize(s: str) -> List[str]:
    s = _norm_text(s)
    if not s:
        return []
    toks = s.split()
    # scoate tokenuri foarte scurte / “umplutura”
    stop = {
        "si","sau","cu","din","de","la","pe","pt","ptr","pentru","in","la","nr","cod",
        "buc","buc.","ore","h","um","u.m","um.","ron","lei","tva","total","subtotal",
        "operatie","operatiune","lucrari","lucrare"
    }
    out = []
    for t in toks:
        if t in stop:
            continue
        if len(t) <= 2:
            continue
        out.append(t)
    return out

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    uni = len(sa | sb)
    return inter / max(1, uni)

def _is_vague_desc(desc: str) -> bool:
    d = _norm_text(desc)
    if not d:
        return True
    vague_markers = [
        "alte operatiuni", "alte operatii", "diverse", "diverse materiale",
        "materiale auxiliare", "consumabile", "consumabil", "consumabile diverse",
        "operatiuni conexe", "operatii conexe", "manopera generala", "manopera diversa",
        "verificare generala", "constatare", "control general", "diagnoza", "diagnostic",
        "taxa", "taxare", "fee"
    ]
    if any(m in d for m in vague_markers):
        return True
    # prea scurt + generic
    if len(d) <= 10:
        return True
    return False

def _top_similar_groups(items: List[ExtractedItem], kind: str, sim_threshold: float = 0.78) -> List[Tuple[str, int]]:
    """
    Grupeaza descrieri similare (aprox) ca sa prinda “fragmentare”.
    Returneaza (repr_desc, count) doar pt grupuri cu count>=2.
    """
    group_reprs: List[Tuple[str, List[str], int]] = []  # (repr, tokens, count)
    for it in items:
        if (it.kind or "").lower() != kind:
            continue
        toks = _tokenize(it.desc or "")
        if not toks:
            continue
        placed = False
        for gi in range(len(group_reprs)):
            rep, rep_toks, cnt = group_reprs[gi]
            if _jaccard(toks, rep_toks) >= sim_threshold:
                group_reprs[gi] = (rep, rep_toks, cnt + 1)
                placed = True
                break
        if not placed:
            rep = _norm_text(it.desc or "")[:80] or "item"
            group_reprs.append((rep, toks, 1))
    out = [(rep, cnt) for (rep, _toks, cnt) in group_reprs if cnt >= 2]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:5]

# =========================
# CORE LOGIC
# =========================

def internal_deviz_check(payload: DevizInternalInput) -> DevizInternalResult:
    flags: List[str] = []
    dbg: Dict[str, Any] = {}

    score = 100.0

    items = payload.items or []
    totals = payload.totals or Totals()

    # normalize kinds
    for it in items:
        it.kind = (it.kind or "part").lower().strip()
        if it.kind not in ("part", "labor"):
            it.kind = "part"

    parts = [i for i in items if i.kind == "part"]
    labor = [i for i in items if i.kind == "labor"]

    dbg["items_count"] = len(items)
    dbg["parts_count"] = len(parts)
    dbg["labor_count"] = len(labor)

    # totals
    tm = _f(totals.materials)
    tl = _f(totals.labor)
    tv = _f(totals.vat)
    tnet = _f(totals.subtotal_no_vat)
    tg = _f(totals.grand_total)

    # sums (always)
    sum_parts = sum(float(i.line_total or 0.0) for i in parts)
    sum_labor = sum(float(i.line_total or 0.0) for i in labor)
    sum_all = sum(float(i.line_total or 0.0) for i in items)

    dbg["sum_parts"] = sum_parts
    dbg["sum_labor"] = sum_labor
    dbg["sum_all"] = sum_all

    # =========================
    # 0) partial payload detection (Make per-line / filtrari)
    # =========================
    partial_payload = False
    partial_reasons: List[str] = []

    if (tl is not None and tl > 0) and len(labor) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_labor_but_no_labor_items")
    if (tm is not None and tm > 0) and len(parts) == 0:
        partial_payload = True
        partial_reasons.append("totals_has_materials_but_no_part_items")

    if len(items) <= 1:
        partial_payload = True
        partial_reasons.append("too_few_items_for_deviz_level_checks")

    # daca exista tg si suma liniilor e mult mai mica, probabil nu ai toate liniile
    if (tg is not None and tg > 0) and (sum_all > 0) and (sum_all < 0.65 * tg):
        partial_payload = True
        partial_reasons.append("sum_all_much_smaller_than_grand_total_likely_missing_items")

    dbg["partial_payload"] = partial_payload
    dbg["partial_reasons"] = partial_reasons

    # daca payload e partial, nu “inventam” verdict OK – dam insuficient_data si ne oprim elegant
    if partial_payload:
        # in continuare calculez VATCheck “daca se poate” si MathConsistency pt o linie, dar verdictul ramane INSUFFICIENT_DATA
        # =========================
        # 1) Math consistency (per line)
        # =========================
        mismatches = 0
        checked = 0
        for it in items:
            q = float(it.qty or 0.0)
            up = float(it.unit_price or 0.0)
            lt = float(it.line_total or 0.0)
            if q > 0 and up > 0:
                checked += 1
                exp = q * up
                if abs(exp - lt) > max(2.0, exp * 0.05):
                    mismatches += 1

        if checked == 0:
            math_consistency = "INSUFFICIENT_DATA"
        elif mismatches == 0:
            math_consistency = "OK"
        else:
            math_consistency = "MISMATCH"

        # =========================
        # 4) VAT logic (best-effort)
        # =========================
        vat_check = "NOT_APPLICABLE"
        if tv is not None and tnet is not None and tnet > 0:
            exp_v = tnet * 0.19
            if abs(tv - exp_v) <= _tol(exp_v, abs_min=2.0, rel=0.05):
                vat_check = "OK"
            else:
                vat_check = "INCONSISTENT"
        elif tv is None and tnet is not None and tg is not None and tnet > 0 and tg > 0:
            inferred = tg - tnet
            exp_v = tnet * 0.19
            if inferred > 0 and _approx(inferred, exp_v, abs_min=5.0, rel=0.08):
                vat_check = "OK"
                flags.append("vat_inferred")
            else:
                vat_check = "NOT_APPLICABLE"

        # labor sanity
        if len(labor) == 0:
            labor_sanity = "SKIPPED"
        else:
            hours = sum(float(i.qty or 0.0) for i in labor if (i.qty or 0.0) > 0)
            val = sum(float(i.line_total or 0.0) for i in labor if (i.line_total or 0.0) > 0)
            rate = _safe_ratio(val, hours)
            dbg["labor_hours"] = hours
            dbg["labor_value"] = val
            dbg["labor_hourly_rate"] = rate
            if hours <= 0 or rate is None:
                labor_sanity = "FAIL"
            elif rate < 30 or rate > 450:
                labor_sanity = "WARN"
            else:
                labor_sanity = "OK"

        flags = ["partial_payload"] + flags
        return DevizInternalResult(
            InternalScore=50,
            InternalVerdict="INSUFFICIENT_DATA",
            InternalFlags=", ".join(flags),
            LaborSanityResult=labor_sanity,
            MathConsistency=math_consistency,
            VATCheck=vat_check,
            debug=dbg
        )

    # =========================
    # 1) Math consistency (per line) – deviz full
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
            if abs(expected - lt) > max(2.0, expected * 0.05):
                mismatches += 1
        else:
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
        score -= 6

    dbg["line_math_checked"] = checked
    dbg["line_math_mismatches"] = mismatches
    dbg["missing_price_lines"] = missing_price_lines

    # =========================
    # 2) Totals vs sums (deviz full)
    # =========================
    if tm is not None and tm > 0 and sum_parts > 0:
        tol_m = _tol(tm, abs_min=5.0, rel=0.05)
        dbg["materials_tol"] = tol_m
        dbg["materials_diff"] = (sum_parts - tm)
        if abs(sum_parts - tm) > tol_m:
            flags.append("materials_total_mismatch")
            score -= 10

    # labor totals mismatch e util DOAR daca ai labor items si total labor e definit
    if tl is not None and tl > 0 and len(labor) > 0:
        tol_l = _tol(tl, abs_min=5.0, rel=0.05)
        dbg["labor_tol"] = tol_l
        dbg["labor_diff"] = (sum_labor - tl)
        if abs(sum_labor - tl) > tol_l:
            flags.append("labor_total_mismatch")
            score -= 10

    if tg is not None and tg > 0 and len(items) >= 3 and sum_all > 0:
        tol_g = _tol(tg, abs_min=10.0, rel=0.05)
        dbg["grand_tol"] = tol_g
        dbg["grand_diff"] = (sum_all - tg)
        if abs(sum_all - tg) > tol_g:
            flags.append("grand_total_mismatch")
            score -= 12

    # extra: grand vs net-from-items (daca ai totals lipsa / sau suspect)
    net_from_items = sum_all
    if tg is not None and tg > 0 and net_from_items > 0:
        r = _safe_ratio(tg, net_from_items)  # daca tg e “cu tva” iar items sunt “fara”, r tinde spre ~1.19
        dbg["grand_over_sum_all_ratio"] = r
        # daca ratio e prea mic/mare, suspect
        if r is not None and (r < 0.90 or r > 1.35):
            flags.append("grand_total_vs_items_sum_suspicious")
            score -= 6

    # =========================
    # 3) Labor sanity + “mintea teparului”
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

        if labor_hours <= 0 or hourly_rate is None:
            labor_sanity = "FAIL"
            flags.append("labor_missing_hours_or_rate")
            score -= 14
        elif hourly_rate < 30 or hourly_rate > 450:
            labor_sanity = "WARN"
            flags.append("labor_hourly_rate_suspicious")
            score -= 6
        else:
            labor_sanity = "OK"

        # --- fraud-ish signals (deviz-level) ---
        labor_lines = [it for it in labor if (it.line_total or 0.0) > 0]
        hours_list = [float(it.qty or 0.0) for it in labor if (it.qty or 0.0) > 0]
        val_list = [float(it.line_total or 0.0) for it in labor if (it.line_total or 0.0) > 0]

        if len(labor_lines) >= 6:
            # fragmentare: multe linii mici
            small_hours = [h for h in hours_list if h > 0 and h <= 0.4]
            dbg["labor_small_hours_count"] = len(small_hours)
            if len(small_hours) >= max(3, int(0.45 * len(hours_list))):
                flags.append("labor_fragmentation_many_small_time_entries")
                score -= 8

        if hours_list:
            try:
                dbg["labor_hours_median"] = median(hours_list)
                dbg["labor_hours_max"] = max(hours_list)
            except Exception:
                pass

        # repetitii: operatii similare care par sparte intentionat
        sim_groups = _top_similar_groups(items, kind="labor", sim_threshold=0.78)
        dbg["labor_similar_groups_top"] = sim_groups
        if sim_groups and sim_groups[0][1] >= 3:
            flags.append("labor_repeated_similar_operations")
            score -= 7

        # descrieri vagi pe labor
        vague_labor = [it for it in labor_lines if _is_vague_desc(it.desc or "")]
        vague_labor_value = sum(float(it.line_total or 0.0) for it in vague_labor)
        dbg["vague_labor_count"] = len(vague_labor)
        dbg["vague_labor_value"] = vague_labor_value
        if labor_value > 0:
            vague_share = vague_labor_value / labor_value
            dbg["vague_labor_share"] = vague_share
            if vague_share >= 0.35 and vague_labor_value >= 150:
                flags.append("labor_value_hidden_in_vague_descriptions")
                score -= 10

    # =========================
    # 4) VAT logic (incl. inference)
    # =========================
    vat_check = "NOT_APPLICABLE"

    # A: explicit net + vat -> validate 19%
    if tv is not None and tnet is not None and tnet > 0:
        expected_vat = tnet * 0.19
        tol_v = _tol(expected_vat, abs_min=2.0, rel=0.05)
        dbg["vat_expected_from_net"] = expected_vat
        dbg["vat_tol"] = tol_v
        dbg["vat_diff"] = (tv - expected_vat)
        if abs(tv - expected_vat) > tol_v:
            vat_check = "INCONSISTENT"
            flags.append("vat_inconsistent")
            score -= 10
        else:
            vat_check = "OK"

    # B: net + grand missing vat -> infer
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
                flags.append("vat_grand_minus_net_not_19pct")
                score -= 6
        else:
            vat_check = "NOT_APPLICABLE"

    # C: materials/labor/grand only (devize uneori au totals “brut” deja)
    else:
        net_from_totals = None
        if tm is not None or tl is not None:
            net_from_totals = float(tm or 0.0) + float(tl or 0.0)

        if net_from_totals is not None and tg is not None and net_from_totals > 0 and tg > 0:
            dbg["net_from_materials_plus_labor"] = net_from_totals
            if _approx(tg, net_from_totals, abs_min=10.0, rel=0.03):
                vat_check = "NOT_APPLICABLE"
                flags.append("vat_probably_included_in_materials_and_labor_totals")
            elif _approx(tg, net_from_totals * 1.19, abs_min=15.0, rel=0.05):
                vat_check = "OK"
                flags.append("vat_inferred_from_grand_vs_net_approx_19pct")
            else:
                vat_check = "NOT_APPLICABLE"
        else:
            vat_check = "NOT_APPLICABLE"

    # =========================
    # 5) Structure sanity + “semnatura tepei”
    # =========================
    if len(items) < 3:
        flags.append("too_few_items")
        score -= 8

    # no parts in a “deviz auto” e suspect (dar lasa o toleranta: uneori e doar manopera)
    if len(parts) == 0 and len(items) >= 4:
        flags.append("no_parts")
        score -= 10

    total_value = sum_all if sum_all > 0 else (tg or 0.0)
    if total_value and total_value > 0:
        labor_share = _safe_ratio(sum_labor, total_value)
        parts_share = _safe_ratio(sum_parts, total_value)
        dbg["labor_share"] = labor_share
        dbg["parts_share"] = parts_share

        # semne extreme
        if labor_share is not None and labor_share > 0.92 and sum_labor >= 300:
            flags.append("labor_share_extremely_high")
            score -= 10

    # vagi pe total (labor+parts)
    vague_all = [it for it in items if _is_vague_desc(it.desc or "")]
    vague_all_value = sum(float(it.line_total or 0.0) for it in vague_all)
    dbg["vague_all_count"] = len(vague_all)
    dbg["vague_all_value"] = vague_all_value
    if total_value and total_value > 0:
        vague_share_total = vague_all_value / total_value
        dbg["vague_share_total"] = vague_share_total
        if vague_share_total >= 0.22 and vague_all_value >= 200:
            flags.append("value_hidden_in_vague_lines")
            score -= 10

    # repetitii pe parts (de ex “ulei motor” de multe ori, “filtru” repetat etc) – nu e automat teapa, dar semnal
    part_groups = _top_similar_groups(items, kind="part", sim_threshold=0.80)
    dbg["part_similar_groups_top"] = part_groups
    if part_groups and part_groups[0][1] >= 3:
        flags.append("parts_repeated_similar_items")
        score -= 4

    # =========================
    # 6) Verdict
    # =========================
    score_i = _clamp_score(score)

    # praguri un pic mai dure ca sa nu iasa “OK” prea usor
    if score_i >= 86:
        verdict = "OK"
    elif score_i >= 65:
        verdict = "SUSPICIOUS"
    elif score_i >= 40:
        verdict = "BAD"
    else:
        verdict = "INSUFFICIENT_DATA"

    # curateaza flags duplicat
    flags_unique = list(dict.fromkeys([f for f in flags if f]))

    return DevizInternalResult(
        InternalScore=score_i,
        InternalVerdict=verdict,
        InternalFlags=", ".join(flags_unique) if flags_unique else "",
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
