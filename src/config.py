from pathlib import Path
import os

# === Data ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Auto-detect CSV - support both local and GCS paths
if "CLV_DATA" in os.environ and os.environ["CLV_DATA"]:
    CSV_PATH = os.environ["CLV_DATA"]  # Can be GCS path like gs://bucket/file.csv
else:
    csv_files = list(DATA_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("❌ No CSV file found in 'data/' and CLV_DATA not set.")
    CSV_PATH = str(csv_files[0])  # Convert to string for consistency

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
# Support both local and GCS paths
GCS_BUCKET = os.environ.get("MODEL_BUCKET", "dataset-clv")
IS_GCS_DEPLOYMENT = os.environ.get("GOOGLE_CLOUD_PROJECT") is not None

if IS_GCS_DEPLOYMENT:
    # Use GCS paths when deployed
    ART_BASE = f"gs://{GCS_BUCKET}/artifacts"
    XGB_MODEL       = f"{ART_BASE}/xgb_model.joblib"
    XGB_JSON        = f"{ART_BASE}/xgb_model.json"
    SCALER_F        = f"{ART_BASE}/scaler.joblib"
    FEATURES_F      = f"{ART_BASE}/feature_cols.joblib"
    KM_MODEL_F      = f"{ART_BASE}/kmeans_model.joblib"
    KM_SCALER_F     = f"{ART_BASE}/kmeans_scaler.joblib"
    SEG_LABELS_F    = f"{ART_BASE}/segment_label_map.joblib"
    CUSTOMER_FEATS  = f"{ART_BASE}/customer_features.csv"
    SAMPLE_FEATS    = f"{ART_BASE}/sample_features.csv"
    SHAP_EXPLAINER  = f"{ART_BASE}/shap_explainer.joblib"
else:
    # Use local paths for development
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
