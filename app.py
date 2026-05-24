import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# Loading Pre Saved Objects
#---------------------------
model = joblib.load(r"voting_classifier_model.joblib")
encoders = joblib.load(r"encoders.joblib")
imputers = joblib.load(r"imputers.joblib")
feature_columns = joblib.load(r"feature_columns.joblib")
background = pd.read_csv(r"shap_background.csv")

# Fixing Data Leakage
#----------------------
tgt_col = "churned"

if tgt_col in feature_columns:
    feature_columns = [c for c in feature_columns if c != tgt_col]

if tgt_col in imputers:
    imputers.pop(tgt_col)

if tgt_col in encoders:
    encoders.pop(tgt_col)

# Using Gradient Boosting for SHAP
#-----------------------------------
gtb_model = model.named_estimators_["gtb"]

explainer = shap.Explainer(
    gtb_model,
    background,
    model_output="probability"
)

# Feature Mapping
# -----------------
def make_feature_readable(feature_name):
    """
    Converts encoded feature names into readable text.
    """
    if "_" not in feature_name:
        return feature_name.replace("_", " ").title()
    else:
        parts = feature_name.split("_", 1)
        base = parts[0].replace("_", " ").title()
        value = parts[1].replace("_", " ").title()
    return f"{base} {value}"

# UI Configuration & Styling
# -------------
st.set_page_config("Netflix Customer Churn Framework", layout="wide")

st.markdown("""
<style>
    /* Global App Background & Font Settings */
    .stApp {
        background: linear-gradient(135deg, #f8fafd 0%, #eef3f9 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Remove Streamlit default top padding */
    .block-container {
        padding-top: 1.5rem !important;
    }

    /* Top Navigation bar simulation */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0px;
        margin-bottom: 25px;
    }
    .brand-title {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
    }
    .brand-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-top: -4px;
    }

    /* Hero Section styling */
    .hero-container {
        text-align: center;
        margin: 20px 0 40px 0;
    }
    .hero-badge {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        color: #475569;
        display: inline-block;
        margin-bottom: 15px;
        font-weight: 500;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    .hero-main-title {
        font-size: 42px;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.5px;
        margin-bottom: 12px;
    }
    .hero-description {
        font-size: 16px;
        color: #475569;
        max-width: 700px;
        margin: 0 auto 20px auto;
        line-height: 1.5;
    }
    .hero-features {
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
    }

    /* Light Theme Main Configuration Layout Panel */
    .parameter-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 25px;
    }
    .parameter-card h3 {
        color: #0f172a !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-bottom: 20px !important;
    }

    /* CRITICAL FIX: Aggressive CSS enforcement to guarantee labels like Age, Gender are black */
    div[data-testid="stWidgetLabel"] p, 
    div[data-testid="stMarkdownContainer"] p,
    .stWidgetFormLabel label, 
    label p,
    .stSelectbox label p,
    .stNumberInput label p {
        color: #0f172a !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Lower Output Block Display Framework */
    .visualization-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 20px;
    }
    .visualization-card h3 {
        color: #0f172a !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        margin-bottom: 15px !important;
    }

    /* Dynamic Metric Displays */
    .metric-display-container {
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
    }
    .metric-display-value {
        font-size: 44px;
        font-weight: 800;
    }
    .bg-red-light {
        background-color: #fef2f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
    }
    .bg-green-light {
        background-color: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
    }

    /* Driver Badge Tags */
    .badge-tag {
        display: inline-block;
        background-color: #f1f5f9;
        color: #334155;
        padding: 6px 14px;
        border-radius: 20px;
        margin: 5px 8px 5px 0;
        font-size: 14px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
    }

    /* Primary Action Trigger Styling */
    .stButton>button {
        background-color: #3b82f6 !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

# Center Hero Frame 
st.markdown("""
<div class="hero-container">
    <div class="hero-main-title">Netflix Customer Churn Prediction</div>
    <div class="hero-description">
        Using a Explainable AI framework to process and compute churn probabilities based on custom physical metrics, 
        usage constraints, and engagement signals mapped from platform interactions.
    </div>
    <div class="hero-features"> Fast Generation &nbsp;•&nbsp; Explainable AI &nbsp;•&nbsp; Scientific Accuracy</div>
</div>
""", unsafe_allow_html=True)

# Two Column Side-by-Side Parameter Setup
col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("""<div class="parameter-card"><h3> General Information</h3>""", unsafe_allow_html=True)
    age = st.number_input("Age", 1, 100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    device = st.selectbox("Device", ["Mobile", "TV", "Laptop", "Tablet", "Desktop"])
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    payment = st.selectbox("Payment Method", ["Gift Card", "Crypto", "Debit Card", "PayPal", "Credit Card"])
    fav_genre = st.selectbox("Favorite Genre", ['Action', 'Sci-Fi', 'Drama', 'Horror', 'Romance', 'Comedy', 'Documentary'])
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""<div class="parameter-card"><h3> Usage & Billing</h3>""", unsafe_allow_html=True)
    watch_hours = st.number_input("Total Watch Hours", 0.0, value=120.0)
    avg_watch_time = st.number_input("Avg Watch Time Per Day", 0.0, value=2.5)
    last_login = st.number_input("Days Since Last Login", 0, value=4)
    profiles = st.number_input("Number of Profiles", 1, 5, value=2)
    monthly_fee = st.number_input("Monthly Fee ($)", 0.0, value=14.99)
    st.markdown("</div>", unsafe_allow_html=True)

# Action button placed below the split parameter panel
st.markdown("<div style='text-align: center; margin-bottom: 35px;'>", unsafe_allow_html=True)
pred_btn = st.button("Generate Diagnostic Output", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Base DataFrame generation for models
raw_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "subscription_type": subscription,
    "watch_hours": watch_hours,
    "last_login_days": last_login,
    "device": device,
    "monthly_fee": monthly_fee,
    "payment_method": payment,
    "number_of_profiles": profiles,
    "avg_watch_time_per_day": avg_watch_time,
    "favorite_genre": fav_genre
}])

def preprocessing_funct(df):
    df = df.copy()
    for col, imputer in imputers.items():
        if col not in df.columns:
            continue
        else:
            df[col] = imputer.transform(df[[col]]).ravel()

    for col, encoder in encoders.items():
        if col not in df.columns:
            continue
        else:
            encoded = encoder.transform(df[[col]])
            encoded_df = pd.DataFrame(
                encoded,
                columns=encoder.get_feature_names_out([col]),
                index=df.index
            )
            df = df.drop(columns=[col])
            df = pd.concat([df, encoded_df], axis=1)

    df = df.reindex(columns=feature_columns, fill_value=0)
    return df

# Output Area: Generates at the low area of the screen post-click
if pred_btn:
    st.markdown("<hr style='border-color: #cbd5e1; margin: 20px 0 40px 0;'>", unsafe_allow_html=True)
    
    X = preprocessing_funct(raw_df)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    cls_labels = model.classes_

    churn_index = list(cls_labels).index(0)
    not_churn_index = list(cls_labels).index(1)

    if pred == 0:
        confidence = float(proba[churn_index])
        status = "HIGH CHURN RISK"
        color_style = "bg-red-light"
    else:
        confidence = float(proba[not_churn_index])
        status = "LOW CHURN RISK"
        color_style = "bg-green-light"

    # Split Output Section into two main blocks for better wide presentation
    out_col1, out_col2 = st.columns([1, 1], gap="medium")
    
    with out_col1:
        # Diagnostic Metric Presentation
        st.markdown(f"""
        <div class="visualization-card">
            <h3>📈 Analysis Metrics</h3>
            <div class="metric-display-container {color_style}">
                <div style="font-size: 13px; font-weight: 700; letter-spacing: 1px; color: inherit !important;">DIAGNOSTIC STATUS</div>
                <div class="metric-display-value">{status}</div>
                <div style="font-size: 15px; font-weight: 500; opacity: 0.9; margin-top: 5px; color: inherit !important;">
                    Prediction Confidence: <strong>{confidence:.2%}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Calculate SHAP Metrics
    shap_values = explainer(X)
    shap_vals = shap_values.values
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "shap_value": shap_vals[0]
    })
    shap_df["abs_value"] = shap_df["shap_value"].abs()

    class_labels = model.classes_
    churn_index = list(class_labels).index(0)

    if churn_index == 0:
        shap_df["impact"] = shap_df["shap_value"].apply(
            lambda x: "Which Increases churn risk" if x < 0 else "Which Reduces churn risk"
        )
    else:
        shap_df["impact"] = shap_df["shap_value"].apply(
            lambda x: "Which Increases churn risk" if x > 0 else "Which Reduces churn risk"
        )

    shap_df["feature_readable"] = shap_df["feature"].apply(make_feature_readable)
    shap_df = shap_df.sort_values("abs_value", ascending=False)

    if pred == 0:
        driver_title = "💥 Primary Churn Drivers"
        driver_df = shap_df[shap_df["impact"].str.contains("Increases")].head(3)
    else:
        driver_title = "✅ Primary Retention Drivers"
        driver_df = shap_df[shap_df["impact"].str.contains("Reduces")].head(3)

    with out_col2:
        # Top Driver Feature Badges
        st.markdown(f"""
        <div class="visualization-card" style="height: 100%;">
            <h3>{driver_title}</h3>
            <div style="margin-top: 25px; margin-bottom: 5px;">
        """, unsafe_allow_html=True)
        
        badge_html = "".join([f"<span class='badge-tag'>{row['feature_readable']}</span>" for _, row in driver_df.iterrows()])
        st.markdown(badge_html, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

    # Full Width Matrix and Human Explanations 
    st.markdown("""
    <div class="visualization-card">
        <h3> Explainability Matrix</h3>
    """, unsafe_allow_html=True)
    display_matrix = shap_df[["feature_readable", "impact", "abs_value", "shap_value"]].head(3).copy()
    display_matrix.columns = ["Feature Description", "Impact Direction", "Absolute Magnitude", "Raw Weight Value"]
    st.dataframe(display_matrix, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    def gen_explanation(shap_df, prediction, probability, top_k=3):
        top_features = shap_df.head(top_k)
        if prediction == 1:
            intro = f"The system analysis confirms that this subscriber profile remains <b>unlikely to churn</b> with a stability assurance likelihood of <b>{probability:.2%}</b>. "
        else:
            intro = f"The system flags this profile as <b>highly susceptible to churn</b> with an estimation probability score of <b>{probability:.2%}</b>. "

        reasons = []
        for _, row in top_features.iterrows():
            reasons.append(f"<b>{row['feature_readable']}</b> ({row['impact'].lower()})")
        
        return intro + "The variance calculation is principally driven by: " + "; ".join(reasons) + "."

    XAI_human_explanation = gen_explanation(
        shap_df=shap_df,
        prediction=pred,
        probability=confidence,
        top_k=3
    )

    st.markdown(f"""
    <div class="visualization-card">
        <h3> XAI Interpretation</h3>
        <div style="font-size: 15px; color: #334155; line-height: 1.6; background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 4px solid #3b82f6;">
            {XAI_human_explanation}
        </div>
    </div>
    """, unsafe_allow_html=True)