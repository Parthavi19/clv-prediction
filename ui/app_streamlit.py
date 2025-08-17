import sys
import os

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Import with error handling
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.warning("SHAP not available. Install with: pip install shap")

try:
    from src.config import CUSTOMER_FEATS, SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F
except ImportError:
    st.error("Configuration not found. Make sure src/config.py exists.")
    st.stop()

st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("📊 Customer Lifetime Value Dashboard with Explainability")

# Check if training has been run
if not os.path.exists(CUSTOMER_FEATS):
    st.error("❌ No customer features found!")
    st.markdown("""
    **Steps to fix:**
    1. Make sure you have training data in the `data/` directory
    2. Update column names in `src/config.py` to match your CSV
    3. Run the training pipeline: `python -m src.train_pipeline`
    4. Redeploy the service
    """)
    st.stop()

# Load data
try:
    df = pd.read_csv(CUSTOMER_FEATS)
    st.success(f"✅ Loaded {len(df)} customer records")
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
    if all(os.path.exists(f) for f in [SHAP_EXPLAINER, XGB_MODEL, SCALER_F, FEATURES_F]):
        try:
            model = joblib.load(XGB_MODEL)
            scaler = joblib.load(SCALER_F)
            feature_cols = joblib.load(FEATURES_F)
            explainer = joblib.load(SHAP_EXPLAINER)

            # Check if we have the required feature columns
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing feature columns for SHAP: {missing_cols}")
            else:
                X = df[feature_cols].fillna(0)  # Fill any NaN values
                X_scaled = scaler.transform(X)
                shap_values = explainer(X_scaled)

                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, features=X, feature_names=feature_cols, 
                                show=False, ax=ax)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating SHAP plots: {e}")
    else:
        st.warning("SHAP explainer not found. Please retrain the model with: `python -m src.train_pipeline`")

    # --- Local SHAP Explanation ---
    st.subheader("👤 Local Explanation for a Specific Customer")
    if all(os.path.exists(f) for f in [SHAP_EXPLAINER, SCALER_F, FEATURES_F]):
        try:
            feature_cols = joblib.load(FEATURES_F)
            scaler = joblib.load(SCALER_F)
            explainer = joblib.load(SHAP_EXPLAINER)

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

