import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

# Force GCS mode if MODEL_BUCKET is set
FORCE_GCS_MODE = os.environ.get("MODEL_BUCKET") is not None

# Import with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    if FORCE_GCS_MODE:
        st.error("Google Cloud Storage client required but not available. Install with: pip install google-cloud-storage")
        st.stop()

# Override config for GCS deployment
if FORCE_GCS_MODE:
    st.info("🌐 Forced GCS mode - MODEL_BUCKET detected")
    GCS_BUCKET = os.environ.get("MODEL_BUCKET", "dataset-clv")
    ART_BASE = f"gs://{GCS_BUCKET}/artifacts"
    
    CUSTOMER_FEATS = f"{ART_BASE}/customer_features.csv"
    SHAP_EXPLAINER = f"{ART_BASE}/shap_explainer.joblib"
    XGB_MODEL = f"{ART_BASE}/xgb_model.joblib"
    SCALER_F = f"{ART_BASE}/scaler.joblib"
    FEATURES_F = f"{ART_BASE}/feature_cols.joblib"
    IS_GCS_DEPLOYMENT = True
else:
    try:
        from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F, IS_GCS_DEPLOYMENT
    except ImportError:
        st.error("Configuration not found. Make sure src/config.py exists.")
        st.stop()

def load_from_gcs_or_local(file_path):
    """Load file from GCS or local filesystem based on path."""
    if isinstance(file_path, str) and file_path.startswith("gs://"):
        if not GCS_AVAILABLE:
            st.error("Google Cloud Storage client not available")
            return None
        
        try:
            # Parse GCS path
            parts = file_path[5:].split("/", 1)  # Remove gs:// prefix
            bucket_name = parts[0]
            blob_name = parts[1]
            
            # Download from GCS
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            
            if file_path.endswith('.csv'):
                # For CSV files, return as pandas DataFrame
                content = blob.download_as_text()
                return pd.read_csv(StringIO(content))
            else:
                # For joblib files, download to bytes and load
                content = blob.download_as_bytes()
                return joblib.load(BytesIO(content))
                
        except Exception as e:
            st.error(f"Error loading from GCS {file_path}: {e}")
            return None
    else:
        # Local file
        try:
            if str(file_path).endswith('.csv'):
                return pd.read_csv(file_path)
            else:
                return joblib.load(file_path)
        except Exception as e:
            st.error(f"Error loading local file {file_path}: {e}")
            return None

def file_exists(file_path):
    """Check if file exists in GCS or locally."""
    if isinstance(file_path, str) and file_path.startswith("gs://"):
        if not GCS_AVAILABLE:
            return False
        try:
            parts = file_path[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1]
            
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            return blob.exists()
        except Exception as e:
            st.error(f"Error checking GCS file existence: {e}")
            return False
    else:
        return os.path.exists(file_path)

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value Dashboard with Explainability")

# Show environment info for debugging
with st.expander("🔧 Debug Information"):
    st.write("Environment Variables:")
    env_vars = {
        "MODEL_BUCKET": os.environ.get("MODEL_BUCKET", "Not Set"),
        "GOOGLE_CLOUD_PROJECT": os.environ.get("GOOGLE_CLOUD_PROJECT", "Not Set"),
        "K_SERVICE": os.environ.get("K_SERVICE", "Not Set"),
        "GAE_ENV": os.environ.get("GAE_ENV", "Not Set"),
    }
    st.json(env_vars)
    st.write(f"Force GCS Mode: {FORCE_GCS_MODE}")
    st.write(f"GCS Available: {GCS_AVAILABLE}")
    st.write(f"Customer Features Path: {CUSTOMER_FEATS}")

# Check deployment mode
if FORCE_GCS_MODE or IS_GCS_DEPLOYMENT:
    st.info("🌐 Running in GCS mode - loading artifacts from cloud storage")
else:
    st.info("💻 Running in local mode - loading artifacts from local files")

# Check if training has been run
st.write("Checking file existence...")
file_exists_result = file_exists(CUSTOMER_FEATS)
st.write(f"File exists: {file_exists_result}")

if not file_exists_result:
    st.error("❌ No customer features found!")
    st.markdown(f"""
    **Current file path:** `{CUSTOMER_FEATS}`
    
    **Steps to fix:**
    1. Make sure you have training data in the `data/` directory
    2. Update column names in `src/config.py` to match your CSV
    3. Run the training pipeline: `python -m src.train_pipeline`
    4. For GCS deployment, make sure artifacts are uploaded to: `{CUSTOMER_FEATS}`
    5. Check that your GCS bucket `dataset-clv` contains the artifacts folder
    6. Redeploy the service
    """)
    
    # Try to list files in the bucket for debugging
    if FORCE_GCS_MODE and GCS_AVAILABLE:
        try:
            st.write("Attempting to list files in GCS bucket...")
            client = storage.Client()
            bucket = client.bucket(os.environ.get("MODEL_BUCKET", "dataset-clv"))
            blobs = list(bucket.list_blobs(prefix="artifacts/"))
            st.write("Files found in artifacts/ folder:")
            for blob in blobs:
                st.write(f"- {blob.name}")
        except Exception as e:
            st.error(f"Error listing GCS files: {e}")
    
    st.stop()

# Load data
try:
    st.write("Loading customer data...")
    df = load_from_gcs_or_local(CUSTOMER_FEATS)
    if df is not None:
        st.success(f"✅ Loaded {len(df)} customer records")
    else:
        st.error("Failed to load customer data")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.exception(e)
    st.stop()

# --- Display table with Predicted CLV + Segment Name ---
st.subheader("Customer Segmentation with Predicted CLV")
display_cols = []
for col in ["customer_id", "predicted_clv", "segment_label", "segment_name"]:
    if col in df.columns:
        display_cols.append(col)

remaining_cols = [c for c in df.columns if c not in display_cols]
final_cols = display_cols + remaining_cols[:5]  # Show first 5 additional columns

st.dataframe(df[final_cols].head(25), use_container_width=True)

# --- Average CLV per segment with segment size coloring & count in labels ---
st.subheader("💰 Average CLV per Segment")
if "segment_name" in df.columns and "predicted_clv" in df.columns:
    avg_clv = df.groupby("segment_name")["predicted_clv"].mean().reset_index()
    segment_sizes = df["segment_name"].value_counts().reset_index()
    segment_sizes.columns = ["segment_name", "count"]
    avg_clv = avg_clv.merge(segment_sizes, on="segment_name")

    # Create labels with counts
    x_labels = [f"{name}\n({count} customers)" for name, count in zip(avg_clv["segment_name"], avg_clv["count"])]

    fig, ax = plt.subplots(figsize=(10, 6))
    # Color intensity by segment size
    colors = plt.cm.Blues(avg_clv["count"] / avg_clv["count"].max())
    bars = ax.bar(range(len(x_labels)), avg_clv["predicted_clv"], color=colors)

    ax.set_xlabel("Segment")
    ax.set_ylabel("Average Predicted CLV")
    ax.set_title("Average CLV by Segment (Darker = Larger Segment)")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=15, ha="right")

    # Add value labels on bars
    for bar, value in zip(bars, avg_clv["predicted_clv"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01, 
                f"${value:,.0f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.warning("Cannot plot CLV by segment. Missing 'segment_name' or 'predicted_clv' columns.")

# --- Summary Statistics ---
st.subheader("📈 Summary Statistics")
if "predicted_clv" in df.columns:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        avg_clv = df["predicted_clv"].mean()
        st.metric("Average CLV", f"${avg_clv:,.0f}")
    
    with col3:
        total_clv = df["predicted_clv"].sum()
        st.metric("Total CLV", f"${total_clv:,.0f}")
    
    with col4:
        if "segment_name" in df.columns:
            segments = df["segment_name"].nunique()
            st.metric("Segments", segments)

# --- Footer ---
st.markdown("---")
st.markdown("*CLV Dashboard powered by XGBoost, SHAP, and Streamlit*")
st.markdown(f"*Deployment mode: {'GCS Cloud Storage' if (FORCE_GCS_MODE or IS_GCS_DEPLOYMENT) else 'Local Files'}*")
