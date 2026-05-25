"""
ESA CDSS Configuration
======================
Central configuration file for the Clinical Decision Support System.
Contains all constants, thresholds, feature definitions, and phenotype names.

Updated 2025-05-25: ElasticNet feature selection + CatBoost model, 2 phenotypes.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
FIGURES_DIR = BASE_DIR / "figures"
VALIDATION_DIR = BASE_DIR / "validation"
OUTPUT_DIR = BASE_DIR / "output"

MODEL_PATH = ASSETS_DIR / "best_model_CatBoost.joblib"
CLUSTER_META_PATH = ASSETS_DIR / "cluster_metadata.json"

# ---------------------------------------------------------------------------
# Phenotype names (05-21 clustering, K=2, ElasticNet/CatBoost risk model)
# ---------------------------------------------------------------------------

SUBTYPE_NAMES = {
    0: "Long-Vintage Mineral-Metabolic Phenotype",
    1: "Advanced-Age Inflammatory-Nutritional Phenotype",
}

SUBTYPE_SHORT_NAMES = {
    0: "Long-Vintage Mineral-Metabolic",
    1: "Advanced-Age Inflammatory-Nutritional",
}

SUBTYPE_DESCRIPTIONS = {
    0: (
        "Characterized by longer dialysis vintage, higher creatinine, higher PTH, "
        "higher phosphorus, higher pre-dialysis blood pressure, and mineral metabolism "
        "burden. This phenotype prioritizes CKD-MBD optimization and mineral metabolism "
        "management as contributors to ESA hyporesponsiveness."
    ),
    1: (
        "Characterized by older age, shorter dialysis vintage, lower body weight, "
        "lower creatinine, lower phosphorus, lower pre-dialysis blood pressure, "
        "and slightly higher IDH count. This phenotype prioritizes hemodynamic "
        "stability, nutritional support, and volume management."
    ),
}

SUBTYPE_CHECKLISTS = {
    0: [
        "Optimize PTH toward the KDIGO CKD G5D range of approximately 2-9 times the assay upper limit",
        "Improve phosphate control with dietary review and binder optimization",
        "Review vitamin D status and active vitamin D or analog therapy",
        "Evaluate calcium-phosphorus balance and vascular calcification risk",
        "Consider calcimimetic therapy for refractory secondary hyperparathyroidism",
        "Reassess hemoglobin and ESA response after CKD-MBD optimization",
    ],
    1: [
        "Re-evaluate target dry weight using clinical assessment or bioimpedance if available",
        "Reduce ultrafiltration rate to < 10 mL/kg/h when feasible",
        "Review interdialytic weight gain and reinforce fluid restriction counseling",
        "Adjust antihypertensive timing, including withholding pre-dialysis doses when appropriate",
        "Consider cool dialysate (35-35.5 C) or sodium profiling for recurrent IDH",
        "Assess nutritional status with dietary intake, body composition, and albumin",
    ],
}

# ---------------------------------------------------------------------------
# Risk stratification thresholds
# ---------------------------------------------------------------------------

RISK_THRESHOLDS = {
    "Low": (0, 0.20),
    "Intermediate": (0.20, 0.50),
    "High": (0.50, 0.80),
    "Very High": (0.80, 1.01),
}

RISK_COLORS = {
    "Low": "#27ae60",
    "Intermediate": "#f39c12",
    "High": "#e74c3c",
    "Very High": "#8e44ad",
}

RISK_DESCRIPTIONS = {
    "Low": (
        "Low predicted risk of ESA hyporesponsiveness in the next quarter. "
        "Continue current management and routine monitoring."
    ),
    "Intermediate": (
        "Moderate predicted risk. Review modifiable risk factors "
        "before routine ESA adjustment."
    ),
    "High": (
        "High predicted risk. Systematically evaluate reversible "
        "drivers before ESA dose escalation."
    ),
    "Very High": (
        "Very high predicted risk. Urgent comprehensive workup "
        "required. Address all reversible contributors."
    ),
}

# ---------------------------------------------------------------------------
# Feature definitions (user-facing input for the CDSS)
# ---------------------------------------------------------------------------

INPUT_FEATURES = {
    "demographics": {
        "age": {"label": "Age", "unit": "years", "min": 18, "max": 100, "default": 60,
                "help": "Patient age in years"},
        "dialysis_age": {"label": "Dialysis Vintage", "unit": "months", "min": 0.0,
                         "max": 300.0, "default": 24.0, "help": "Time on dialysis"},
    },
    "anemia_esa": {
        "hb": {"label": "Hemoglobin", "unit": "g/L", "min": 30.0, "max": 180.0,
               "default": 100.0, "ref": "100-115 g/L (KDIGO target)"},
        "esa_dose": {"label": "ESA Weekly Dose", "unit": "IU", "min": 0.0,
                     "max": 50000.0, "default": 10000.0, "step": 500.0},
        "esa_route": {"label": "ESA Route", "options": ["Subcutaneous", "Intravenous"],
                      "default": "Subcutaneous"},
        "dry_weight": {"label": "Dry Weight", "unit": "kg", "min": 20.0, "max": 150.0,
                       "default": 60.0},
    },
    "dialysis": {
        "dialysis_hours": {"label": "Dialysis Session Length", "unit": "hours", "min": 2.0,
                           "max": 6.0, "default": 4.0, "step": 0.25,
                           "help": "Typical treatment duration per session"},
    },
    "iron_status": {
        "ferritin_current": {"label": "Current Ferritin", "unit": "ng/mL", "min": 0.0, "max": 5000.0,
                             "default": 200.0, "step": 10.0,
                             "ref": "Common HD target: >=200 ng/mL; interpret with inflammation"},
        "tsat_current": {"label": "Current TSAT", "unit": "%", "min": 0.0, "max": 100.0,
                         "default": 25.0, "step": 0.5,
                         "ref": "Common HD target: >=20%; low TSAT suggests iron-restricted erythropoiesis"},
    },
    "ckd_mbd": {
        "pth": {"label": "PTH", "unit": "pg/mL", "min": 0.0, "max": 3000.0,
                "default": 300.0, "ref": "KDIGO CKD G5D: approximately 2-9x the assay upper limit"},
        "phosphorus": {"label": "Phosphorus", "unit": "mmol/L", "min": 0.0, "max": 4.0,
                       "default": 1.8, "step": 0.01, "ref": "Target: 1.13-1.78"},
    },
    "electrolytes": {
        "sodium": {"label": "Sodium", "unit": "mmol/L", "min": 100.0, "max": 160.0,
                   "default": 138.0},
        "creatinine": {"label": "Creatinine", "unit": "umol/L", "min": 0.0,
                       "max": 2000.0, "default": 800.0},
    },
    "hemodynamics": {
        "sbp": {"label": "Pre-dialysis SBP", "unit": "mmHg", "min": 50.0, "max": 250.0,
                "default": 145.0},
        "dbp": {"label": "Pre-dialysis DBP", "unit": "mmHg", "min": 30.0, "max": 150.0,
                "default": 80.0},
        "idh_count": {"label": "IDH Count (recent quarter)", "unit": "", "min": 0,
                      "max": 50, "default": 0, "step": 1,
                      "help": "Number of intradialytic hypotension episodes in the current quarter"},
    },
}

# SHAP display labels
FEATURE_LABELS = {
    "age": "Age",
    "dialysis_age": "Dialysis Vintage",
    "hb": "Hemoglobin",
    "esa_dose": "ESA Weekly Dose",
    "esa_route": "ESA Route",
    "dry_weight": "Dry Weight",
    "dialysis_hours": "Dialysis Session Length",
    "sodium": "Sodium",
    "creatinine": "Creatinine",
    "tsat_current": "Current TSAT",
    "ferritin_current": "Current Ferritin",
    "pth": "PTH",
    "pre_sbp_mean": "Pre-dialysis SBP",
    "pre_dbp_mean": "Pre-dialysis DBP",
    "idh_count": "IDH Count",
    "center_creator": "Dialysis Center",
    "crp": "CRP",
    "phosphorus": "Phosphorus",
    "albumin": "Albumin",
    "ktv": "Kt/V",
    "urr": "URR",
    "potassium": "Potassium",
    "calcium": "Calcium",
}

# Features to hide from SHAP display
SHAP_HIDE = {
    "center_creator", "receiving_center", "patient_status",
    "esa_use", "esa_type", "iron_use", "hif_use", "esa_unit",
    "sex", "primary_disease",
}

# ---------------------------------------------------------------------------
# Decision rule definitions
# ---------------------------------------------------------------------------

DECISION_TIERS = {
    "urgent": {
        "label": "URGENT",
        "color": "#c62828",
        "bg_color": "#ffebee",
        "css_class": "tier-urgent",
        "description": "Requires immediate clinical attention",
    },
    "primary": {
        "label": "PRIMARY OPTIMIZATION",
        "color": "#e65100",
        "bg_color": "#fff3e0",
        "css_class": "tier-primary",
        "description": "SHAP-driven, address key risk contributors",
    },
    "phenotype": {
        "label": "PHENOTYPE-ALIGNED",
        "color": "#1565c0",
        "bg_color": "#e3f2fd",
        "css_class": "tier-phenotype",
        "description": "Specific to patient's clinical phenotype",
    },
    "supportive": {
        "label": "SUPPORTIVE",
        "color": "#2e7d32",
        "bg_color": "#e8f5e9",
        "css_class": "tier-supportive",
        "description": "Additional optimization opportunities",
    },
}

# Evidence levels
EVIDENCE_LEVELS = {
    "1A": "Strong recommendation, high-quality evidence",
    "1B": "Strong recommendation, moderate-quality evidence",
    "1C": "Strong recommendation, low-quality evidence",
    "2A": "Weak recommendation, high-quality evidence",
    "2B": "Weak recommendation, moderate-quality evidence",
    "2C": "Weak recommendation, low-quality evidence",
    "Expert": "Expert opinion / clinical experience",
}

# Publication figure settings
FIGURE_DPI = 300
FIGURE_FORMATS = ["png", "pdf"]
FIGURE_FONT_FAMILY = "Arial"
FIGURE_FONT_SIZE = 12
FIGURE_TITLE_SIZE = 14
FIGURE_SIZE_SINGLE = (8, 6)
FIGURE_SIZE_DOUBLE = (16, 8)
FIGURE_SIZE_TRIPLE = (18, 6)
