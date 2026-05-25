"""
ESA CDSS Inference Engine
=========================
Core prediction engine that integrates:
  - CatBoost risk prediction model (ElasticNet-selected 16 features)
  - K-Means phenotype classification (K=2)
  - SHAP-based explainability
  - Patient-level clinical action recommendations

Updated 2025-05-25: ElasticNet + CatBoost pipeline, 2 phenotypes.
"""

import os
import json
import joblib
import shap
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    MODEL_PATH, CLUSTER_META_PATH, SUBTYPE_NAMES, SUBTYPE_SHORT_NAMES,
    SUBTYPE_DESCRIPTIONS, SUBTYPE_CHECKLISTS, FEATURE_LABELS, SHAP_HIDE,
    RISK_THRESHOLDS, RISK_COLORS, RISK_DESCRIPTIONS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fval(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Asset loading
# ---------------------------------------------------------------------------

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
PREPROCESSOR_CONFIG_PATH = ASSETS_DIR / "preprocessor_config.json"


class CDSSAssets:
    """Lazy-loading singleton for all model assets."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return

        # --- Load CatBoost pipeline (sklearn Pipeline with preprocessor + CatBoost) ---
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        self.pipeline = joblib.load(MODEL_PATH)
        self.model = self.pipeline.named_steps['model']
        self.ct = self.pipeline.named_steps['preprocessor']

        # --- Load preprocessor config (for sklearn-independent fallback) ---
        if not PREPROCESSOR_CONFIG_PATH.exists():
            raise FileNotFoundError(f"Preprocessor config not found: {PREPROCESSOR_CONFIG_PATH}")
        with open(PREPROCESSOR_CONFIG_PATH, "r", encoding="utf-8") as f:
            preproc_cfg = json.load(f)
        self.feat_names_out = preproc_cfg.get("feature_names_out")
        self._preproc_cfg = preproc_cfg

        # Required raw input features for the CatBoost pipeline
        self.required_features = list(self.pipeline.feature_names_in_)

        # --- Load cluster metadata ---
        if not CLUSTER_META_PATH.exists():
            raise FileNotFoundError(f"Cluster metadata not found: {CLUSTER_META_PATH}")
        with open(CLUSTER_META_PATH, "r", encoding="utf-8") as f:
            self.cluster_meta = json.load(f)

        # --- SHAP explainer ---
        self.explainer = shap.TreeExplainer(self.model)

        self._loaded = True

    def transform(self, df):
        """Apply pipeline preprocessing via the sklearn ColumnTransformer."""
        try:
            return self.ct.transform(df)
        except Exception:
            return self._manual_transform(df)

    def _manual_transform(self, df):
        """Fallback manual preprocessing (sklearn-independent)."""
        cfg = self._preproc_cfg
        num_parts = {}
        cat_parts = {}

        for t_info in cfg["transformers"]:
            name = t_info["name"]
            columns = t_info["columns"]
            steps = t_info["steps"]

            if name == "num":
                step0 = steps[0].get("imputer", steps[0][list(steps[0].keys())[0]])
                step1 = steps[1].get("scaler", steps[1][list(steps[1].keys())[0]])
                fill_values = step0["statistics_"]
                means = step1["mean_"]
                scales = step1["scale_"]

                for i, col in enumerate(columns):
                    series = df[col].apply(pd.to_numeric, errors="coerce")
                    series = series.fillna(fill_values[i])
                    num_parts[f"num__{col}"] = (series - means[i]) / scales[i]

            elif name == "cat":
                step0 = steps[0].get("imputer", steps[0][list(steps[0].keys())[0]])
                step1 = steps[1].get("encoder", steps[1][list(steps[1].keys())[0]])
                categories = step1["categories_"]

                for i, col in enumerate(columns):
                    series = df[col].astype(str).fillna("")
                    for cat_val in categories[i]:
                        cat_parts[f"cat__{col}_{cat_val}"] = (series == cat_val).astype(float)

        all_parts = {**num_parts, **cat_parts}
        result = pd.DataFrame(all_parts, index=df.index)

        if self.feat_names_out:
            for col in self.feat_names_out:
                if col not in result.columns:
                    result[col] = 0.0
            result = result[self.feat_names_out]

        return result.values


def get_assets() -> CDSSAssets:
    assets = CDSSAssets()
    assets.load()
    return assets


# ---------------------------------------------------------------------------
# Input processing
# ---------------------------------------------------------------------------

def build_input_dataframe(values: Dict[str, Any]) -> pd.DataFrame:
    """Build a feature DataFrame from user input for the CatBoost pipeline.

    The pipeline expects 16 features:
    hb, esa_dose, esa_route, dry_weight, center_creator, dialysis_hours,
    age, sodium, dialysis_age, pre_sbp_mean, tsat_current, creatinine,
    ferritin_current, pth, pre_dbp_mean, idh_count
    """
    row = values.copy()

    # Map user-facing names to pipeline feature names
    row.setdefault("age", _fval(values.get("age")))
    row.setdefault("dialysis_age", _fval(values.get("dialysis_age")))
    row.setdefault("hb", _fval(values.get("hb")))
    row.setdefault("esa_dose", _fval(values.get("esa_dose")))
    row.setdefault("dry_weight", _fval(values.get("dry_weight")))
    row.setdefault("dialysis_hours", _fval(values.get("dialysis_hours", 4.0)))
    row.setdefault("sodium", _fval(values.get("sodium")))
    row.setdefault("creatinine", _fval(values.get("creatinine")))
    row.setdefault("ferritin_current", _fval(values.get("ferritin_current")))
    row.setdefault("tsat_current", _fval(values.get("tsat_current")))
    row.setdefault("pth", _fval(values.get("pth")))
    row.setdefault("pre_sbp_mean", _fval(values.get("sbp")))
    row.setdefault("pre_dbp_mean", _fval(values.get("dbp")))
    row.setdefault("idh_count", _fval(values.get("idh_count", values.get("idh_any", 0))))

    # ESA route mapping
    route = str(row.get("esa_route", "Subcutaneous"))
    if route in ("Subcutaneous", "SC", "subcutaneous"):
        row["esa_route"] = "皮下"
    elif route in ("Intravenous", "IV", "intravenous"):
        row["esa_route"] = "静脉"

    # Default categorical values
    row.setdefault("center_creator", "河源市紫金县中医院血透中心")

    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Risk stratification
# ---------------------------------------------------------------------------

def assign_risk_level(prob: float) -> str:
    if prob <= 0.20:
        return "Low"
    if prob <= 0.50:
        return "Intermediate"
    if prob <= 0.80:
        return "High"
    return "Very High"


# ---------------------------------------------------------------------------
# Phenotype classification
# ---------------------------------------------------------------------------

def assign_phenotype(values: Dict[str, Any], cluster_meta: Dict) -> Dict:
    """Classify patient into a phenotype cluster (K=2)."""
    centroids = np.array(
        cluster_meta.get("centroids_scaled", cluster_meta.get("centroids")),
        dtype=float,
    )
    feature_order = cluster_meta["cluster_features"]
    imp = cluster_meta.get("imputation_values", {})
    mean = cluster_meta.get("scaler_mean", {})
    scale = cluster_meta.get("scaler_scale", {})

    raw_input_map = cluster_meta.get("raw_input_features", {
        "age": "age", "dialysis_age": "dialysis_age", "hb": "hb",
        "esa_dose": "esa_dose", "dry_weight": "dry_weight",
        "dialysis_hours": "dialysis_hours", "sodium": "sodium",
        "creatinine": "creatinine", "phosphorus": "phosphorus",
        "tsat_current": "tsat_current", "pre_sbp_mean": "sbp",
        "pre_dbp_mean": "dbp", "idh_count": "idh_count",
    })

    prepared = values.copy()
    crp = _fval(prepared.get("crp"))
    pth = _fval(prepared.get("pth"))
    ferritin = _fval(prepared.get("ferritin_current"))

    # Compute log-transformed clustering features
    if np.isfinite(crp):
        cap = _fval(cluster_meta.get("crp_p99_cap", 91.27))
        capped = min(max(crp, 0.0), cap) if np.isfinite(cap) else max(crp, 0.0)
        prepared["log_crp_w99"] = np.log1p(capped)
    if np.isfinite(pth) and pth > 0:
        prepared["log_pth_w99"] = np.log1p(pth)
    if np.isfinite(ferritin) and ferritin > 0:
        prepared["log_ferritin_w99"] = np.log1p(ferritin)

    arr = []
    for col in feature_order:
        # Map cluster feature name to user input name
        input_key = raw_input_map.get(col, col)
        v = _fval(prepared.get(input_key, prepared.get(col)))
        if not np.isfinite(v):
            v = _fval(imp.get(col, 0))
        if mean and scale:
            s = _fval(scale.get(col, 1))
            v = (v - _fval(mean.get(col, 0))) / (s if s else 1)
        arr.append(v)

    vec = np.array(arr, dtype=float)
    dists = np.linalg.norm(centroids - vec, axis=1)
    exp_neg = np.exp(-dists)
    probs = exp_neg / exp_neg.sum()

    idx = int(np.argmin(dists))
    name_map = SUBTYPE_NAMES

    distances = {}
    for i, name in name_map.items():
        distances[name] = {
            "distance": round(float(dists[i]), 3),
            "similarity": round(float(probs[i]), 3),
        }

    return {
        "assigned_index": idx,
        "assigned": name_map[idx],
        "assigned_short": SUBTYPE_SHORT_NAMES[idx],
        "description": SUBTYPE_DESCRIPTIONS[idx],
        "checklist": SUBTYPE_CHECKLISTS[idx],
        "distances": distances,
    }


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------

def compute_shap(assets: CDSSAssets, df: pd.DataFrame) -> Dict:
    """Compute SHAP values and extract risk drivers/protectors."""
    X_processed = assets.transform(df)
    X_processed = np.atleast_2d(X_processed)

    sv = assets.explainer.shap_values(X_processed)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]

    fnames = (assets.feat_names_out
              if assets.feat_names_out
              else [f"f{i}" for i in range(len(sv))])

    def _strip(name):
        if name.startswith("num__"):
            return name[5:]
        if name.startswith("cat__"):
            return name[5:]
        return name

    contributions = []
    for i, fname in enumerate(fnames):
        bare = _strip(fname)
        if bare in SHAP_HIDE:
            continue
        if fname.startswith("cat__"):
            continue
        contributions.append({
            "feature": bare,
            "label": FEATURE_LABELS.get(bare, bare),
            "shap": float(sv[i]),
            "abs_shap": abs(float(sv[i])),
        })
    contributions.sort(key=lambda x: x["abs_shap"], reverse=True)

    drivers = [c for c in contributions if c["shap"] > 0][:10]
    protective = [c for c in contributions if c["shap"] < 0][:5]

    return {
        "contributions": contributions,
        "drivers": drivers,
        "protective": protective,
        "base_value": float(assets.explainer.expected_value),
    }


# ---------------------------------------------------------------------------
# Decision recommendation engine
# ---------------------------------------------------------------------------

TIMEFRAME_RANK = {
    "Immediate / same day": 0,
    "Within 1 week": 1,
    "2-4 weeks": 2,
    "4-8 week reassessment": 3,
}

SEVERITY_RANK = {
    "emergency": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}

TIER_RANK = {
    "urgent": 0,
    "primary": 1,
    "phenotype": 2,
    "supportive": 3,
}

CLINICAL_SAFETY_NOTE = (
    "Apply within local protocols, medication contraindications, and the responsible clinician's judgment."
)


def _domain(feat: str) -> str:
    if feat in ("ferritin_current", "tsat_current", "ferritin", "tsat",
                "ferritin_mean", "tsat_mean"):
        return "iron_status"
    if feat in ("esa_dose", "esa_route"):
        return "esa"
    if feat in ("pre_sbp_mean", "sbp"):
        return "bp"
    if feat in ("pre_dbp_mean", "dbp"):
        return "bp"
    if feat in ("idh_count", "idh_any"):
        return "idh"
    if feat in ("pth", "phosphorus"):
        return "mbd"
    return feat


def _fmt(value, unit="", digits=1, missing="not available"):
    if not np.isfinite(value):
        return missing
    if digits == 0:
        text = f"{value:.0f}"
    elif digits == 2:
        text = f"{value:.2f}"
    else:
        text = f"{value:.1f}"
    return f"{text} {unit}".strip()


def _yes_no(flag: bool) -> str:
    return "present" if flag else "absent"


def _patient_context(values: Dict[str, Any]) -> Dict[str, Any]:
    hb = _fval(values.get("hb"))
    dry_weight = _fval(values.get("dry_weight"))
    esa_dose = _fval(values.get("esa_dose"))
    route = str(values.get("esa_route", "Subcutaneous"))
    eq_dose = esa_dose if route in ("Subcutaneous", "SC", "subcutaneous", "皮下") else esa_dose * 2 / 3
    eri = eq_dose / dry_weight / (hb / 10.0) if (np.isfinite(eq_dose) and np.isfinite(dry_weight) and np.isfinite(hb) and dry_weight > 0 and hb > 0) else np.nan

    crp = _fval(values.get("crp"))
    albumin = _fval(values.get("albumin"))
    ktv = _fval(values.get("ktv"))
    urr = _fval(values.get("urr"))
    pth = _fval(values.get("pth"))
    calcium = _fval(values.get("calcium"))
    phosphorus = _fval(values.get("phosphorus"))
    potassium = _fval(values.get("potassium"))
    sodium = _fval(values.get("sodium"))
    sbp = _fval(values.get("sbp", values.get("pre_sbp_mean")))
    dbp = _fval(values.get("dbp", values.get("pre_dbp_mean")))
    idh_raw = _fval(values.get("idh_count", values.get("idh_any", 0)))
    idh = bool(np.isfinite(idh_raw) and int(idh_raw) > 0)
    ferritin = _fval(values.get("ferritin_current", values.get("ferritin")))
    tsat = _fval(values.get("tsat_current", values.get("tsat")))

    iron_pattern = "not classifiable"
    if not np.isfinite(ferritin) or not np.isfinite(tsat):
        iron_pattern = "incomplete iron assessment"
    elif tsat < 20 and ferritin < 200:
        iron_pattern = "absolute iron deficiency"
    elif tsat < 20 and ferritin >= 200:
        iron_pattern = "functional iron restriction"
    elif ferritin < 200:
        iron_pattern = "depleted iron stores"
    elif ferritin > 800 and tsat < 25:
        iron_pattern = "high ferritin with limited circulating iron"
    else:
        iron_pattern = "no major iron restriction signal"

    labels = []
    if np.isfinite(hb) and hb < 100:
        labels.append(f"low Hb {_fmt(hb, 'g/L')}")
    if np.isfinite(eri) and eri > 12:
        labels.append(f"high ERI {_fmt(eri)}")
    if np.isfinite(crp) and crp > 5:
        labels.append(f"inflammation CRP {_fmt(crp, 'mg/L')}")
    if np.isfinite(albumin) and albumin < 35:
        labels.append(f"low albumin {_fmt(albumin, 'g/L')}")
    if np.isfinite(ktv) and ktv < 1.2:
        labels.append(f"low Kt/V {_fmt(ktv, digits=2)}")
    if np.isfinite(pth) and pth > 600:
        labels.append(f"marked PTH elevation {_fmt(pth, 'pg/mL', 0)}")
    elif np.isfinite(pth) and pth > 300:
        labels.append(f"PTH elevation {_fmt(pth, 'pg/mL', 0)}")
    if np.isfinite(phosphorus) and phosphorus > 1.78:
        labels.append(f"hyperphosphatemia {_fmt(phosphorus, 'mmol/L', 2)}")
    if idh:
        labels.append("intradialytic hypotension")

    return {
        "hb": hb, "dry_weight": dry_weight, "esa_dose": esa_dose,
        "eq_dose": eq_dose, "eri": eri, "crp": crp, "albumin": albumin,
        "ktv": ktv, "urr": urr, "pth": pth, "calcium": calcium,
        "phosphorus": phosphorus, "potassium": potassium, "sodium": sodium,
        "sbp": sbp, "dbp": dbp, "idh": idh, "ferritin": ferritin, "tsat": tsat,
        "iron_pattern": iron_pattern, "labels": labels,
    }


def _sg(tier: str, severity: str, timeframe: str, title: str, rationale: str,
        actions: List[str], avoid: Optional[str] = None, monitoring: Optional[str] = None,
        detail: Optional[str] = None, feature: Optional[str] = None,
        evidence: str = "Clinical best practice", priority: int = 0,
        shap: Optional[float] = None) -> Dict[str, Any]:
    if detail is None:
        detail_parts = [rationale]
        if actions:
            detail_parts.append("Actions: " + "; ".join(actions))
        if avoid:
            detail_parts.append("Avoid/caution: " + avoid)
        if monitoring:
            detail_parts.append("Monitoring: " + monitoring)
        if detail_parts:
            detail_parts.append(CLINICAL_SAFETY_NOTE)
        detail = "\n".join(detail_parts)
    return {
        "tier": tier, "severity": severity, "timeframe": timeframe,
        "title": title, "detail": detail, "rationale": rationale,
        "actions": actions,
        "avoid": avoid or "None specific beyond standard contraindication review.",
        "monitoring": monitoring or "Reassess after the selected intervention interval.",
        "feature": feature, "evidence": evidence, "priority": priority, "shap": shap,
    }


def _driver_suggestion(feat, sv, values):
    ctx = _patient_context(values)
    tag = f"SHAP +{sv:.3f}"
    hb = ctx["hb"]
    eri = ctx["eri"]
    crp = ctx["crp"]
    alb = ctx["albumin"]
    ktv = ctx["ktv"]
    urr = ctx["urr"]
    pth = ctx["pth"]
    phos = ctx["phosphorus"]
    sbp = ctx["sbp"]
    dbp = ctx["dbp"]
    esa_dose = ctx["esa_dose"]
    ferritin = ctx["ferritin"]
    tsat = ctx["tsat"]

    if feat == "hb" and np.isfinite(hb) and hb < 110:
        return _sg(
            "primary", "medium", "2-4 weeks",
            f"Low hemoglobin requires cause-directed anemia review ({tag})",
            f"Hb is {_fmt(hb, 'g/L')}; the patient's iron pattern is {ctx['iron_pattern']}, CRP is {_fmt(crp, 'mg/L')}, and ERI is {_fmt(eri)}.",
            [
                "Confirm iron status with ferritin and TSAT if not current",
                "Check for occult blood loss, access bleeding, recent hospitalization, and hemolysis when clinically indicated",
                "Optimize inflammation, iron availability, dialysis adequacy, and CKD-MBD before medication intensification",
            ],
            "Do not increase ESA reflexively when inflammation, iron restriction, underdialysis, or uncontrolled MBD is present.",
            "Repeat Hb, ESA dose, ERI, CRP, ferritin, and TSAT in 2-4 weeks if high-risk drivers are active.",
            feature="hb", evidence="KDIGO anemia guidance; clinical workflow", shap=sv,
        )

    if feat in ("esa_dose",):
        if np.isfinite(esa_dose) and esa_dose > 12000:
            return _sg(
                "primary", "high", "Within 1 week",
                f"Avoid reflexive ESA escalation; complete low-response workup ({tag})",
                f"ESA dose is {_fmt(esa_dose, 'IU/week', 0)} and ERI is {_fmt(eri)}, suggesting possible ESA hyporesponsiveness.",
                [
                    "Prioritize iron restriction, inflammation, dialysis adequacy, CKD-MBD, bleeding, and hemolysis review",
                    "Document the active reversible drivers before any ESA dose change",
                    "Use local anemia protocol for any modest dose adjustment after reversible drivers are addressed",
                ],
                "Avoid supratherapeutic ESA escalation while Hb remains low.",
                "Track Hb, ESA dose, ERI, blood pressure, and adverse events over the next 2-4 weeks.",
                feature=feat, evidence="KDIGO anemia guidance; ESA safety communications", shap=sv,
            )

    if _domain(feat) == "iron_status":
        if not np.isfinite(ferritin) or not np.isfinite(tsat):
            return _sg(
                "primary", "high", "Within 1 week",
                f"Complete iron assessment before anemia treatment changes ({tag})",
                "Missing ferritin or TSAT prevents distinction between absolute iron deficiency and functional iron restriction.",
                ["Order ferritin and TSAT together", "Review recent iron exposure, infection signs, blood loss, and transfusion history"],
                "Avoid attributing low Hb to ESA resistance until iron availability is known.",
                "Recheck iron indices 4-8 weeks after any iron intervention.",
                feature=feat, evidence="KDIGO anemia guidance", shap=sv,
            )
        if tsat < 20 and ferritin < 200:
            return _sg(
                "primary", "high", "Within 1 week",
                f"Absolute iron deficiency is a priority reversible driver ({tag})",
                f"TSAT is {_fmt(tsat, '%')} and ferritin is {_fmt(ferritin, 'ng/mL', 0)}, consistent with depleted iron stores.",
                [
                    "Assess recent blood loss, access bleeding, gastrointestinal symptoms, and iron adherence",
                    "Optimize iron repletion according to local hemodialysis anemia protocol",
                    "Reassess ESA response after iron availability improves",
                ],
                "Avoid increasing ESA before iron repletion unless there is a separate urgent indication.",
                "Repeat Hb, ferritin, and TSAT in 4-8 weeks after iron optimization.",
                feature=feat, evidence="KDIGO anemia guidance", shap=sv,
            )
        if tsat < 20 and ferritin >= 200:
            severity = "high" if (np.isfinite(crp) and crp > 5) or ferritin > 800 else "medium"
            return _sg(
                "primary", severity, "Within 1 week" if severity == "high" else "2-4 weeks",
                f"Functional iron restriction needs inflammation-aware management ({tag})",
                f"TSAT is {_fmt(tsat, '%')} with ferritin {_fmt(ferritin, 'ng/mL', 0)} and CRP {_fmt(crp, 'mg/L')}.",
                [
                    "Look for active infection or inflammatory disease",
                    "Decide whether to continue, hold, or adjust iron using local protocol",
                    "Address inflammation before interpreting ESA failure",
                ],
                "Avoid adding iron or escalating ESA automatically when ferritin is high or active infection is suspected.",
                "Repeat CRP, ferritin, TSAT, Hb, and ERI in 2-4 weeks if inflammation is active.",
                feature=feat, evidence="KDIGO anemia guidance", shap=sv,
            )

    if feat in ("pre_sbp_mean", "pre_dbp_mean", "sbp", "dbp") and np.isfinite(sbp):
        if sbp < 110:
            return _sg(
                "primary", "high", "Immediate / same day" if ctx["idh"] else "Within 1 week",
                f"Low pre-dialysis blood pressure may limit dialysis delivery ({tag})",
                f"Pre-dialysis BP is {_fmt(sbp, 'mmHg', 0)}/{_fmt(dbp, 'mmHg', 0)} and IDH is {_yes_no(ctx['idh'])}.",
                [
                    "Assess dry weight, ultrafiltration rate, interdialytic weight gain, and antihypertensive timing",
                    "Evaluate cardiac function or autonomic dysfunction if instability persists",
                    "Stabilize hemodynamics so dialysis adequacy and anemia management can be delivered safely",
                ],
                "Avoid aggressive ultrafiltration or pre-dialysis antihypertensive dosing when recurrent IDH is present.",
                "Track pre-/intra-dialysis BP, IDH events, achieved treatment time, and Hb over the next month.",
                feature="pre_sbp_mean", evidence="KDOQI hemodialysis practice guidance", shap=sv,
            )

    if feat in ("idh_count", "idh_any") and ctx["idh"]:
        return _sg(
            "primary", "high", "Within 1 week",
            f"Intradialytic hypotension should be corrected to protect dialysis delivery ({tag})",
            f"IDH is present with pre-dialysis BP {_fmt(sbp, 'mmHg', 0)}/{_fmt(dbp, 'mmHg', 0)}.",
            [
                "Re-evaluate dry weight and ultrafiltration rate; target UF rate below 10 mL/kg/h when feasible",
                "Review pre-dialysis antihypertensive timing and interdialytic weight gain",
                "Consider cool dialysate, sodium profiling, or midodrine for recurrent IDH",
                "Assess cardiac function if IDH is persistent or unexplained",
            ],
            "Avoid shortening dialysis repeatedly as the only response to IDH.",
            "Review IDH events each treatment; repeat monthly adequacy and reassess Hb/ERI after stabilization.",
            feature="idh_count", evidence="KDOQI hemodialysis practice guidance", shap=sv,
        )

    label = FEATURE_LABELS.get(feat, feat)
    return _sg(
        "primary", "medium", "2-4 weeks",
        f"Review {label} in the full clinical context ({tag})",
        f"{label} contributes to the model's predicted risk; interpret alongside Hb, iron status, inflammation, and hemodynamics.",
        ["Confirm data accuracy", "Assess whether the factor is modifiable", "Integrate with the higher-priority clinical action list"],
        "Avoid acting on model attribution alone without clinical confirmation.",
        "Reassess after correcting higher-priority reversible drivers.",
        feature=feat, evidence="Model attribution plus clinical assessment", shap=sv,
    )


def _urgent_rules(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    suggestions = []

    if np.isfinite(ctx["crp"]) and ctx["crp"] > 50:
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Screen for active infection or severe inflammation before anemia escalation",
            f"CRP is {_fmt(ctx['crp'], 'mg/L')}; this pattern can cause inflammation-mediated ESA resistance and functional iron restriction.",
            [
                "Assess vascular access for tenderness, erythema, drainage, or dysfunction",
                "Ask about fever, chills, respiratory, urinary, skin, wound, gastrointestinal, and dental symptoms",
                "Obtain cultures or imaging when clinical findings support infection workup",
            ],
            "Avoid simply increasing ESA during uncontrolled inflammation.",
            "Recheck CRP, Hb, ESA dose, ERI, ferritin, and TSAT within 2-4 weeks.",
            feature="crp", evidence="KDIGO anemia guidance",
        ))

    if ctx["idh"] and np.isfinite(ctx["sbp"]) and ctx["sbp"] < 110:
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Stabilize low pre-dialysis BP with intradialytic hypotension",
            f"IDH is present with pre-dialysis BP {_fmt(ctx['sbp'], 'mmHg', 0)}/{_fmt(ctx['dbp'], 'mmHg', 0)}.",
            [
                "Review dry weight, interdialytic weight gain, ultrafiltration rate",
                "Adjust antihypertensive timing and consider holding pre-dialysis doses",
                "Reduce UF rate when feasible; consider cool dialysate or midodrine for recurrent IDH",
            ],
            "Avoid repeated treatment shortening as the only response to hypotension.",
            "Track IDH every treatment and reassess delivered Kt/V after hemodynamic stabilization.",
            feature="idh_count", evidence="KDOQI hemodialysis practice guidance",
        ))

    if np.isfinite(ctx["potassium"]) and ctx["potassium"] >= 6.0:
        suggestions.append(_sg(
            "urgent", "emergency", "Immediate / same day",
            "Manage hyperkalemia before routine anemia optimization",
            f"Potassium is {_fmt(ctx['potassium'], 'mmol/L')}.",
            [
                "Confirm sample validity and assess ECG or symptoms",
                "Review recent dialysis adequacy, missed treatment, diet, medications",
                "Use the unit's hyperkalemia management protocol",
            ],
            "Avoid delaying hyperkalemia management to address ESA hyporesponsiveness first.",
            "Recheck potassium according to local protocol.",
            feature="potassium", evidence="Clinical safety workflow",
        ))

    return suggestions


def _reversible_cause_rules(ctx: Dict[str, Any], shap_result: Dict) -> List[Dict[str, Any]]:
    suggestions = []
    ferritin = ctx["ferritin"]
    tsat = ctx["tsat"]
    crp = ctx["crp"]
    albumin = ctx["albumin"]
    pth = ctx["pth"]
    phosphorus = ctx["phosphorus"]

    if not np.isfinite(ferritin) or not np.isfinite(tsat):
        suggestions.append(_sg(
            "primary", "high", "Within 1 week",
            "Complete iron-status testing before classifying ESA low response",
            "Ferritin and TSAT are both required to distinguish absolute iron deficiency from functional iron restriction.",
            ["Order ferritin and TSAT together", "Review recent iron administration, transfusion, infection, and blood loss"],
            "Avoid increasing ESA or giving empiric iron without current iron indices.",
            "Repeat iron indices 4-8 weeks after any iron intervention.",
            feature="iron_status", evidence="KDIGO anemia guidance",
        ))
    elif tsat < 20 and ferritin < 200:
        suggestions.append(_sg(
            "primary", "high", "Within 1 week",
            "Prioritize absolute iron deficiency correction",
            f"TSAT is {_fmt(tsat, '%')} and ferritin is {_fmt(ferritin, 'ng/mL', 0)}.",
            ["Assess recent blood loss, access bleeding, and iron adherence", "Optimize iron repletion", "Review Hb response before additional ESA escalation"],
            "Avoid labeling the patient ESA-resistant before iron stores are corrected.",
            "Repeat Hb, ferritin, TSAT, and ESA dose in 4-8 weeks.",
            feature="tsat_current", evidence="KDIGO anemia guidance",
        ))
    elif tsat < 20 and ferritin >= 200:
        suggestions.append(_sg(
            "primary", "high" if (np.isfinite(crp) and crp > 5) else "medium",
            "Within 1 week" if (np.isfinite(crp) and crp > 5) else "2-4 weeks",
            "Treat functional iron restriction as inflammation-aware ESA resistance",
            f"TSAT is {_fmt(tsat, '%')} with ferritin {_fmt(ferritin, 'ng/mL', 0)} and CRP {_fmt(crp, 'mg/L')}.",
            ["Screen for infection or chronic inflammation", "Address inflammation first when CRP is elevated"],
            "Avoid additional iron or ESA escalation when ferritin is high or infection is suspected.",
            "Repeat CRP, ferritin, TSAT, Hb, and ERI in 2-4 weeks.",
            feature="tsat_current", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(crp) and crp > 5:
        suggestions.append(_sg(
            "primary", "high" if crp >= 20 else "medium",
            "Within 1 week" if crp >= 20 else "2-4 weeks",
            "Identify and treat inflammation driving ESA hyporesponsiveness",
            f"CRP is {_fmt(crp, 'mg/L')}, albumin is {_fmt(albumin, 'g/L')}, and iron pattern is {ctx['iron_pattern']}.",
            ["Review vascular access, recent infection, heart failure, autoimmune disease", "Use targeted cultures, imaging, or referral when indicated"],
            "Avoid escalating ESA as the sole response to inflammatory anemia.",
            "Repeat CRP, Hb, ferritin, TSAT, and ERI in 2-4 weeks.",
            feature="crp", evidence="KDIGO anemia guidance",
        ))

    mbd_signal = ((np.isfinite(pth) and pth > 300) or
                  (np.isfinite(phosphorus) and phosphorus > 1.78))
    if mbd_signal:
        suggestions.append(_sg(
            "primary", "medium", "2-4 weeks",
            "Address CKD-MBD contributors to impaired erythropoiesis",
            f"PTH is {_fmt(pth, 'pg/mL', 0)}, phosphorus is {_fmt(phosphorus, 'mmol/L', 2)}.",
            [
                "Review dietary phosphate sources and phosphate binder timing",
                "Assess active vitamin D or analog therapy, calcimimetic suitability",
                "Use KDIGO CKD G5D guidance for PTH targets",
            ],
            "Avoid over-suppression of PTH without reviewing calcium-phosphorus balance.",
            "Repeat calcium and phosphorus in 4-8 weeks, and PTH per local protocol.",
            feature="pth", evidence="KDIGO CKD-MBD 2017",
        ))

    return suggestions


def _esa_strategy_rules(ctx: Dict[str, Any], risk_level: str, risk_score: float) -> List[Dict[str, Any]]:
    suggestions = []
    hb = ctx["hb"]
    eri = ctx["eri"]
    esa_dose = ctx["esa_dose"]

    if (np.isfinite(eri) and eri > 12) or (np.isfinite(esa_dose) and esa_dose > 12000):
        suggestions.append(_sg(
            "supportive", "medium", "2-4 weeks",
            "ESA strategy: complete reversible-cause checklist before dose escalation",
            f"ERI is {_fmt(eri)} and ESA dose is {_fmt(esa_dose, 'IU/week', 0)}.",
            [
                "Document the active reversible drivers and whether each has a plan",
                "Use local anemia protocol for any conservative ESA adjustment",
                "Discuss risk-benefit when ESA exposure is already high",
            ],
            "Avoid repeated ESA increases when Hb response is poor and reversible drivers remain active.",
            "Reassess Hb, ESA dose, ERI in 2-4 weeks.",
            feature="esa_dose", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(hb) and hb < 100 and not ((np.isfinite(eri) and eri > 12) or (np.isfinite(esa_dose) and esa_dose > 12000)):
        suggestions.append(_sg(
            "supportive", "medium", "2-4 weeks",
            "ESA strategy: consider modest protocol-based adjustment only after checks",
            f"Hb is {_fmt(hb, 'g/L')}; reversible contributors are being assessed.",
            ["Confirm iron status, inflammation, dialysis adequacy, and CKD-MBD status", "If checks are addressed, consider a modest ESA adjustment"],
            "Avoid adjusting ESA based on Hb alone.",
            "Repeat Hb and ESA dose response in 2-4 weeks.",
            feature="hb", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(hb) and 100 <= hb <= 115:
        suggestions.append(_sg(
            "supportive", "low", "4-8 week reassessment",
            "ESA strategy: Hb is in the usual maintenance range",
            f"Hb is {_fmt(hb, 'g/L')}; model risk should address risk factors rather than push Hb higher.",
            ["Maintain anemia therapy unless local protocol or symptoms indicate otherwise"],
            "Avoid increasing ESA solely because model risk is intermediate or high when Hb is acceptable.",
            "Repeat Hb, ESA dose, and ERI at routine interval.",
            feature="hb", evidence="KDIGO anemia guidance",
        ))

    if np.isfinite(hb) and hb > 115:
        suggestions.append(_sg(
            "supportive", "medium", "2-4 weeks",
            "ESA strategy: avoid hemoglobin overcorrection",
            f"Hb is {_fmt(hb, 'g/L')}; anemia treatment intensity should be reviewed.",
            ["Review ESA dose trajectory and cardiovascular risk", "Consider dose reduction per local protocol if Hb continues to rise"],
            "Avoid intensifying ESA when Hb is above the maintenance range.",
            "Repeat Hb within 2-4 weeks if therapy is adjusted.",
            feature="hb", evidence="KDIGO anemia guidance",
        ))

    return suggestions


def _phenotype_rules(ctx: Dict[str, Any], phenotype: str) -> List[Dict[str, Any]]:
    idx = None
    for k, name in SUBTYPE_NAMES.items():
        if name == phenotype:
            idx = k
            break
    if idx is None:
        return []

    if idx == 0:  # Mineral-metabolism phenotype
        title = "Phenotype focus: Mineral-metabolism phenotype"
        rationale = (f"Assigned phenotype is Mineral-metabolism phenotype; "
                     f"PTH is {_fmt(ctx['pth'], 'pg/mL', 0)}, phosphorus is {_fmt(ctx['phosphorus'], 'mmol/L', 2)}, "
                     f"creatinine is {_fmt(ctx['sbp'], 'umol/L', 0)}.")
        actions = [
            "Optimize phosphate binder use with meals and dietary phosphate intake",
            "Review active vitamin D or analog therapy, calcimimetic eligibility",
            "Use the KDIGO CKD G5D PTH range of approximately 2-9 times the assay upper limit",
        ]
        avoid = "Avoid using a fixed 150-300 pg/mL PTH target as a universal CKD G5D treatment goal."
        monitoring = "Repeat calcium/phosphorus in 4-8 weeks and PTH in 4-8 or 8-12 weeks depending on intervention."
    else:  # Older short-vintage phenotype
        title = "Phenotype focus: Older short-vintage phenotype"
        rationale = (f"Assigned phenotype is Older short-vintage phenotype; "
                     f"age is {_fmt(ctx['hb'], 'g/L')}, dry weight is {_fmt(ctx['dry_weight'], 'kg')}, "
                     f"pre-dialysis BP is {_fmt(ctx['sbp'], 'mmHg', 0)}/{_fmt(ctx['dbp'], 'mmHg', 0)}, "
                     f"IDH is {_yes_no(ctx['idh'])}.")
        actions = [
            "Prioritize dry-weight, UF-rate, antihypertensive timing review",
            "Assess nutritional status and body composition in older patients",
            "Use cool dialysate, sodium profiling, or midodrine for recurrent IDH per local protocol",
            "Recheck delivered dialysis adequacy after hemodynamic stabilization",
        ]
        avoid = "Avoid sacrificing dialysis time repeatedly without fixing the reason for instability."
        monitoring = "Track IDH every session; repeat Kt/V/URR monthly and Hb/ERI after stability improves."

    return [_sg(
        "phenotype", "medium", "2-4 weeks",
        title, rationale, actions, avoid, monitoring,
        feature="phenotype", evidence="Phenotype clustering analysis integrated with clinical rules",
    )]


def _followup_plan(ctx: Dict[str, Any], risk_level: str) -> List[Dict[str, Any]]:
    if risk_level in ("High", "Very High"):
        timeframe = "2-4 weeks"
        severity = "medium"
        actions = [
            "Create a problem list ordered as emergency issues, reversible causes, anemia medication strategy, then routine monitoring",
            "Repeat Hb, ESA dose, ERI, CRP, ferritin, TSAT, and any abnormal safety labs",
        ]
        rationale = f"Risk level is {risk_level}; short-interval reassessment is needed."
    elif risk_level == "Intermediate":
        timeframe = "4-8 week reassessment"
        severity = "low"
        actions = [
            "Recheck Hb, ESA dose, ERI, iron indices, CRP, and phenotype-specific drivers",
            "Escalate if Hb falls, ERI rises, CRP increases, or dialysis adequacy worsens",
        ]
        rationale = "Risk level is Intermediate; monitoring should verify that modifiable risks do not progress."
    else:
        timeframe = "4-8 week reassessment"
        severity = "low"
        actions = ["Continue routine Hb, iron-status, dialysis adequacy, and CKD-MBD monitoring"]
        rationale = "Risk level is Low; routine surveillance is appropriate."

    return [_sg(
        "supportive", severity, timeframe,
        "Follow-up and reassessment plan",
        rationale, actions,
        "Avoid changing anemia therapy without a documented response assessment interval.",
        "Use the reassessment interval above.",
        feature=None, evidence="Clinical best practice",
    )]


def _rank_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        suggestions,
        key=lambda s: (
            SEVERITY_RANK.get(s.get("severity", "low"), 9),
            TIMEFRAME_RANK.get(s.get("timeframe", "4-8 week reassessment"), 9),
            TIER_RANK.get(s.get("tier", "supportive"), 9),
            s.get("priority", 999),
            -abs(s.get("shap") or 0),
            s.get("title", ""),
        )
    )
    for i, sg in enumerate(ranked, 1):
        sg["priority"] = i
    return ranked


def _dedupe_suggestions(suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for sg in suggestions:
        key = (sg.get("tier"), sg.get("feature"), sg.get("title"))
        broad_key = (sg.get("feature"), sg.get("severity"), sg.get("timeframe"))
        if key in seen or broad_key in seen:
            continue
        seen.add(key)
        seen.add(broad_key)
        deduped.append(sg)
    return deduped


def generate_suggestions(values: Dict, shap_result: Dict, phenotype: str,
                         risk_level: str, risk_score: float) -> List[Dict]:
    ctx = _patient_context(values)
    suggestions = []
    priority_counter = 0

    def add(items):
        nonlocal priority_counter
        for item in items:
            priority_counter += 1
            item["priority"] = priority_counter
            suggestions.append(item)

    add(_urgent_rules(ctx))
    add(_reversible_cause_rules(ctx, shap_result))

    seen_domains = {s.get("feature") for s in suggestions if s.get("feature")}
    seen_shap_domains = set()
    for d in shap_result.get("drivers", [])[:10]:
        feat = d["feature"]
        dom = _domain(feat)
        if dom in seen_shap_domains:
            continue
        if feat in seen_domains and dom in ("iron_status", "mbd", "crp"):
            seen_shap_domains.add(dom)
            continue
        sg = _driver_suggestion(feat, d["shap"], values)
        if sg:
            priority_counter += 1
            sg["priority"] = priority_counter
            suggestions.append(sg)
            seen_shap_domains.add(dom)

    add(_esa_strategy_rules(ctx, risk_level, risk_score))
    add(_phenotype_rules(ctx, phenotype))
    add(_followup_plan(ctx, risk_level))

    suggestions = _dedupe_suggestions(suggestions)
    return _rank_suggestions(suggestions)


# ---------------------------------------------------------------------------
# Main prediction entry point
# ---------------------------------------------------------------------------

def predict_case(values: Dict[str, Any]) -> Dict[str, Any]:
    """Full CDSS prediction pipeline: risk + phenotype + SHAP + recommendations."""
    assets = get_assets()

    df = build_input_dataframe(values)

    # Ensure all required features are present
    for feat in assets.required_features:
        if feat not in df.columns:
            df[feat] = np.nan

    prob = float(assets.pipeline.predict_proba(df)[0, 1])
    risk_level = assign_risk_level(prob)

    row = df.iloc[0].to_dict()
    pheno = assign_phenotype(row, assets.cluster_meta)
    shap_result = compute_shap(assets, df)
    suggestions = generate_suggestions(
        row, shap_result, pheno["assigned"], risk_level, prob
    )

    return {
        "risk_score": prob,
        "risk_level": risk_level,
        "risk_color": RISK_COLORS.get(risk_level, "#95a5a6"),
        "risk_description": RISK_DESCRIPTIONS.get(risk_level, ""),
        "phenotype": pheno["assigned"],
        "phenotype_short": pheno["assigned_short"],
        "phenotype_index": pheno["assigned_index"],
        "phenotype_description": pheno["description"],
        "phenotype_checklist": pheno["checklist"],
        "phenotype_distances": pheno["distances"],
        "shap": shap_result,
        "suggestions": suggestions,
        "input_values": values,
    }
