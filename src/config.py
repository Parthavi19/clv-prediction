from pathlib import Path
import os

# === Data ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Auto-detect CSV
if "CLV_DATA" in os.environ and os.environ["CLV_DATA"]:
    CSV_PATH = Path(os.environ["CLV_DATA"])
else:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found in 'data/' and CLV_DATA not set.")
    CSV_PATH = csv_files[0]

# Updated column names to match your dataset
DATE_COL  = "transaction_date"
CUST_COL  = "customer_id"
INV_COL   = "transaction_id"
QTY_COL   = "quantity_purchased"
UP_COL    = "unit_price"
TOTAL_COL = "total_sale_amount"

# Additional columns available in your dataset
LOYALTY_COL = "loyalty_status"
AGE_COL = "age"
GENDER_COL = "gender"
LOCATION_COL = "location"
PRODUCT_ID_COL = "product_id"
PRODUCT_CAT_COL = "product_category"
BRAND_COL = "brand"
STOCK_QTY_COL = "stock_quantity"
PAYMENT_METHOD_COL = "payment_method"
DESCRIPTION_COL = "Description"

# === Artifacts ===
ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

XGB_MODEL       = ART_DIR / "xgb_model.joblib"
XGB_JSON        = ART_DIR / "xgb_model.json"
SCALER_F        = ART_DIR / "scaler.joblib"
FEATURES_F      = ART_DIR / "feature_cols.joblib"
KM_MODEL_F      = ART_DIR / "kmeans_model.joblib"
KM_SCALER_F     = ART_DIR / "kmeans_scaler.joblib"
SEG_LABELS_F    = ART_DIR / "segment_label_map.joblib"
CUSTOMER_FEATS  = ART_DIR / "customer_features.csv"
SAMPLE_FEATS    = ART_DIR / "sample_features.csv"
SHAP_EXPLAINER  = ART_DIR / "shap_explainer.joblib"

# Training config
RANDOM_STATE = 42
K_CLUSTERS = 4
TARGET_HORIZON_DAYS = 365
VAL_SIZE = 0.2
