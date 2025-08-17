import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

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
    st.warning("Google Cloud Storage client not available. Install with: pip install google-cloud-storage")

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
        except:
            return False
    else:
        return os.path.exists(file_path)

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value Dashboard with Explainability")

# Check deployment mode
if IS_GCS_DEPLOYMENT:
    st.info("🌐 Running in GCS mode - loading artifacts from cloud storage")
else:
    st.info("💻 Running in local mode - loading artifacts from local files")

# Check if training has been run
if not file_exists(CUSTOMER_FEATS):
    st.error("❌ No customer features found!")
    st.markdown(f"""
    **Steps to fix:**
    1. Make sure you have training data in the `data/` directory
    2. Update column names in `src/config.py` to match your CSV
    3. Run the training pipeline: `python -m src.train_pipeline`
    4. For GCS deployment, make sure artifacts are uploaded to: `{CUSTOMER_FEATS}`
    5. Redeploy the service
    """)
    st.stop()

# Load data
try:
    df = load_from_gcs_or_local(CUSTOMER_FEATS)
    if df is not None:
        st.success(f"✅ Loaded {len(df)} customer records")
    else:
        st.error("Failed to load customer data")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
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

# --- Global SHAP Importance ---
if SHAP_AVAILABLE:
    st.subheader("🌍 Global Feature Importance (SHAP)")
    required_files = [SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F]
    if all(file_exists(f) for f in required_files):
        try:
            with st.spinner("Loading model and generating SHAP explanations..."):
                model = load_from_gcs_or_local(XGB_MODEL)
                scaler = load_from_gcs_or_local(SCALER_F)
                feature_cols = load_from_gcs_or_local(FEATURES_F)
                explainer = load_from_gcs_or_local(SHAP_EXPLAINER)

                if all(x is not None for x in [model, scaler, feature_cols, explainer]):
                    # Check if we have the required feature columns
                    missing_cols = [col for col in feature_cols if col not in df.columns]
                    if missing_cols:
                        st.warning(f"Missing feature columns for SHAP: {missing_cols}")
                    else:
                        X = df[feature_cols].fillna(0)  # Fill any NaN values
                        X_scaled = scaler.transform(X)
                        
                        # Use a subset for performance
                        sample_size = min(100, len(X_scaled))
                        X_sample = X_scaled[:sample_size]
                        X_sample_orig = X.iloc[:sample_size]
                        
                        shap_values = explainer(X_sample)

                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values, features=X_sample_orig, feature_names=feature_cols, 
                                        show=False, ax=ax)
                        st.pyplot(fig)
                else:
                    st.error("Failed to load required model components")
        except Exception as e:
            st.error(f"Error generating SHAP plots: {e}")
            st.exception(e)
    else:
        missing_files = [f for f in required_files if not file_exists(f)]
        st.warning(f"SHAP files not found: {missing_files}. Please retrain the model with: `python -m src.train_pipeline`")

    # --- Local SHAP Explanation ---
    st.subheader("👤 Local Explanation for a Specific Customer")
    required_files = [SHAP_EXPLAINER, SCALER_F, FEATURES_F]
    if all(file_exists(f) for f in required_files):
        try:
            feature_cols = load_from_gcs_or_local(FEATURES_F)
            scaler = load_from_gcs_or_local(SCALER_F)
            explainer = load_from_gcs_or_local(SHAP_EXPLAINER)

            if all(x is not None for x in [feature_cols, scaler, explainer]):
                customer_list = df["customer_id"].astype(str).tolist()
                selected_customer = st.selectbox("Select Customer ID", customer_list[:100])  # Limit for performance

                if selected_customer:
                    cust_row = df[df["customer_id"].astype(str) == selected_customer][feature_cols]
                    if not cust_row.empty:
                        cust_scaled = scaler.transform(cust_row.fillna(0))
                        cust_shap_values = explainer(cust_scaled)

                        # Display customer info
                        customer_info = df[df["customer_id"].astype(str) == selected_customer].iloc[0]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Customer ID", selected_customer)
                            if "predicted_clv" in customer_info:
                                st.metric("Predicted CLV", f"${customer_info['predicted_clv']:,.0f}")
                        with col2:
                            if "segment_name" in customer_info:
                                st.metric("Segment", customer_info["segment_name"])

                        # SHAP waterfall plot
                        fig_local, ax = plt.subplots(figsize=(10, 6))
                        shap.waterfall_plot(cust_shap_values[0], show=False)
                        st.pyplot(fig_local)
                    else:
                        st.error("Customer not found in dataset.")
        except Exception as e:
            st.error(f"Error generating local SHAP explanation: {e}")
            st.exception(e)

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
st.markdown(f"*Deployment mode: {'GCS Cloud Storage' if IS_GCS_DEPLOYMENT else 'Local Files'}*")
