# ============================================================
#  NEO Hazard Prediction System — Streamlit App
#  No model files needed — trains on first launch from CSV
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="NEO Hazard Predictor",
    page_icon="☄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2e 50%, #0a0a1a 100%);
        color: #e8eaf6;
    }
    .neo-header {
        background: linear-gradient(90deg, #1a1a3e, #0d2b4e);
        border: 1px solid #2a4a7f;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 28px;
        text-align: center;
    }
    .neo-header h1 { font-size:2.4rem; font-weight:800; color:#ffffff; letter-spacing:1px; margin:0 0 6px 0; }
    .neo-header p  { color:#90caf9; font-size:1rem; margin:0; }
    .input-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(100,149,237,0.25);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .input-card h3 {
        color:#90caf9; font-size:1rem; font-weight:600;
        text-transform:uppercase; letter-spacing:1px;
        margin:0 0 16px 0;
        border-bottom:1px solid rgba(100,149,237,0.2);
        padding-bottom:10px;
    }
    .result-hazardous {
        background: linear-gradient(135deg,#3d0000,#5c1010);
        border: 2px solid #ff4444; border-radius:12px;
        padding:28px; text-align:center; margin:20px 0;
    }
    .result-safe {
        background: linear-gradient(135deg,#003d1a,#0a5c2e);
        border: 2px solid #00c853; border-radius:12px;
        padding:28px; text-align:center; margin:20px 0;
    }
    .result-hazardous h2 { font-size:2rem; font-weight:900; margin:0 0 8px 0; letter-spacing:2px; color:#ff6b6b; }
    .result-safe h2      { font-size:2rem; font-weight:900; margin:0 0 8px 0; letter-spacing:2px; color:#69f0ae; }
    .result-hazardous p, .result-safe p { font-size:1rem; margin:0; opacity:0.85; }
    .metric-row { display:flex; gap:16px; margin-top:20px; }
    .metric-box {
        flex:1; background:rgba(255,255,255,0.05);
        border:1px solid rgba(100,149,237,0.2);
        border-radius:10px; padding:18px; text-align:center;
    }
    .metric-box .metric-label { font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; color:#90caf9; margin-bottom:6px; }
    .metric-box .metric-value { font-size:1.6rem; font-weight:800; color:#ffffff; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0d1b2e,#0a0a1a);
        border-right: 1px solid rgba(100,149,237,0.2);
    }
    .stButton > button {
        width:100%; background:linear-gradient(90deg,#1565c0,#0d47a1);
        color:white; font-weight:700; font-size:1.05rem;
        letter-spacing:1px; padding:14px; border:none;
        border-radius:8px; cursor:pointer; margin-top:10px;
    }
    .stButton > button:hover {
        background:linear-gradient(90deg,#1976d2,#1565c0);
        box-shadow:0 4px 20px rgba(21,101,192,0.5);
    }
    label { color:#b0bec5 !important; font-size:0.88rem !important; }
    hr    { border-color:rgba(100,149,237,0.15) !important; }
    .history-row {
        background:rgba(255,255,255,0.03);
        border:1px solid rgba(100,149,237,0.15);
        border-radius:8px; padding:12px 16px; margin-bottom:8px;
        font-size:0.85rem; display:flex;
        justify-content:space-between; align-items:center;
    }
    .tag-hazardous { color:#ff6b6b; font-weight:700; }
    .tag-safe      { color:#69f0ae; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ── Train & Cache Model (runs once on first launch) ──────────
@st.cache_resource(show_spinner="🚀 Training model on first launch — this takes ~30 seconds...")
def train_and_cache():
    # ── Load data ──────────────────────────────────────────
    csv_path = 'neo.csv' if os.path.exists('neo.csv') else 'sprint1.csv'

    df = pd.read_csv(csv_path)
    df.drop(columns=['Unnamed: 0', 'id', 'name', 'orbiting_body', 'sentry_object'],
            errors='ignore', inplace=True)
    df['hazardous'] = df['hazardous'].astype(int) if df['hazardous'].dtype != int else df['hazardous']
    df.drop_duplicates(inplace=True)

    # Outlier capping
    num_cols = ['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','absolute_magnitude']
    for col in num_cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = df[col].clip(Q1 - 1.5*IQR, Q3 + 1.5*IQR)

    # ── Feature engineering ────────────────────────────────
    df['Avg_diameter']            = (df['est_diameter_min'] + df['est_diameter_max']) / 2
    df['Diameter_range']          = df['est_diameter_max'] - df['est_diameter_min']
    df['Velocity_distance_ratio'] = df['relative_velocity'] / df['miss_distance']

    X_raw = df.drop('hazardous', axis=1)
    y     = df['hazardous']

    # ── Drop highly correlated features ───────────────────
    corr_matrix = X_raw.corr().abs()
    upper  = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.90)]
    X = X_raw.drop(columns=to_drop)

    # ── Train / Test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Scale ──────────────────────────────────────────────
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_test_scaled  = scaler.transform(X_test)

    # ── RFE feature selection ──────────────────────────────
    rfe = RFE(estimator=RandomForestClassifier(random_state=42), n_features_to_select=5)
    rfe.fit(x_train_scaled, y_train)
    selected_features  = X.columns[rfe.support_].tolist()
    x_train_selected   = rfe.transform(x_train_scaled)
    x_test_selected    = rfe.transform(x_test_scaled)

    # ── Final model (best params from Sprint 3 GridSearchCV) ─
    best_params = {
        'n_estimators'    : 200,
        'max_depth'       : 20,
        'min_samples_split': 2,
        'min_samples_leaf' : 1,
        'random_state'    : 42
    }

    # Build pipeline on selected features
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  RandomForestClassifier(**best_params))
    ])
    X_train_sel_df = pd.DataFrame(x_train_selected, columns=selected_features)
    X_test_sel_df  = pd.DataFrame(x_test_selected,  columns=selected_features)
    pipeline.fit(X_train_sel_df, y_train)

    # ── Metrics ────────────────────────────────────────────
    y_pred = pipeline.predict(X_test_sel_df)
    y_prob = pipeline.predict_proba(X_test_sel_df)[:, 1]

    metrics = {
        'accuracy' : round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred), 4),
        'recall'   : round(recall_score(y_test, y_pred), 4),
        'f1_score' : round(f1_score(y_test, y_pred), 4),
        'roc_auc'  : round(roc_auc_score(y_test, y_prob), 4),
    }

    return pipeline, scaler, rfe, selected_features, metrics


# ── Run training (cached after first run) ────────────────────
pipeline, scaler, rfe, sel_features, metrics = train_and_cache()


# ── Predict function ─────────────────────────────────────────
def predict_hazard(est_diameter_min, est_diameter_max,
                   relative_velocity, miss_distance, absolute_magnitude):
    row = pd.DataFrame([{
        'est_diameter_min' : est_diameter_min,
        'est_diameter_max' : est_diameter_max,
        'relative_velocity': relative_velocity,
        'miss_distance'    : miss_distance,
        'absolute_magnitude': absolute_magnitude
    }])

    # Feature engineering
    row['Avg_diameter']            = (row['est_diameter_min'] + row['est_diameter_max']) / 2
    row['Diameter_range']          = row['est_diameter_max'] - row['est_diameter_min']
    row['Velocity_distance_ratio'] = row['relative_velocity'] / row['miss_distance']

    # Match exact columns scaler was fit on
    scaler_cols = scaler.feature_names_in_.tolist()
    row_clean   = row[scaler_cols]

    # Scale → RFE → predict
    row_scaled   = scaler.transform(row_clean)
    row_selected = rfe.transform(row_scaled)
    row_df       = pd.DataFrame(row_selected, columns=sel_features)

    pred = pipeline.predict(row_df)[0]
    prob = pipeline.predict_proba(row_df)[0][1]
    return int(pred), round(float(prob), 4)


# ── Session state ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="neo-header">
    <h1>☄️ NEO Hazard Prediction System</h1>
    <p>Enter asteroid parameters below to classify its threat level </p>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    st.metric("Version",   "1.0.0")

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    col_a, col_b = st.columns(2)
    col_a.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
    col_b.metric("Precision", f"{metrics['precision']:.4f}")
    col_a.metric("Recall",    f"{metrics['recall']:.4f}")
    col_b.metric("F1 Score",  f"{metrics['f1_score']:.4f}")
    st.metric("ROC-AUC",      f"{metrics['roc_auc']:.4f}")

    st.markdown("---")
    st.markdown("### 🎯 Selected Features")
    for feat in sel_features:
        st.markdown(f"• `{feat}`")

    st.markdown("---")
    st.caption("Sprint 4 · NEO Hazard Prediction · Innomatics Research Labs")


# ── MAIN LAYOUT ───────────────────────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<div class="input-card"><h3>🔭 Physical Characteristics</h3>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        est_diameter_min = st.number_input(
            "Estimated Diameter Min (km)",
            min_value=0.0, max_value=50.0, value=0.25, step=0.01,
            help="Minimum estimated diameter of the asteroid in km"
        )
    with c2:
        est_diameter_max = st.number_input(
            "Estimated Diameter Max (km)",
            min_value=0.0, max_value=50.0, value=0.56, step=0.01,
            help="Maximum estimated diameter of the asteroid in km"
        )
    absolute_magnitude = st.number_input(
        "Absolute Magnitude (H)",
        min_value=0.0, max_value=35.0, value=20.5, step=0.1,
        help="Lower value = brighter / larger object"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card"><h3>🚀 Orbital Parameters</h3>', unsafe_allow_html=True)
    relative_velocity = st.number_input(
        "Relative Velocity (km/s)",
        min_value=0.0, max_value=300000.0, value=75000.0, step=500.0,
        help="Speed of the asteroid relative to Earth"
    )
    miss_distance = st.number_input(
        "Miss Distance (km)",
        min_value=0.0, max_value=100000000.0, value=35000000.0, step=100000.0,
        help="Closest distance the asteroid will pass from Earth"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if est_diameter_min > est_diameter_max:
        st.warning("⚠️ Diameter Min should be ≤ Diameter Max.")

    predict_clicked = st.button("🚀  PREDICT HAZARD LEVEL", use_container_width=True)


# ── RESULTS ───────────────────────────────────────────────────
with col_right:
    st.markdown("### 🎯 Prediction Result")

    if predict_clicked:
        if est_diameter_min > est_diameter_max:
            st.error("Please fix Diameter Min / Max before predicting.")
        else:
            with st.spinner("Analysing asteroid..."):
                pred, prob = predict_hazard(
                    est_diameter_min, est_diameter_max,
                    relative_velocity, miss_distance, absolute_magnitude
                )

            if pred == 1:
                st.markdown("""
                <div class="result-hazardous">
                    <h2>⚠️ HAZARDOUS</h2>
                    <p>This asteroid poses a potential threat to Earth.<br>
                    Immediate monitoring is recommended.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-safe">
                    <h2>✅ NON-HAZARDOUS</h2>
                    <p>This asteroid is not an immediate threat to Earth.<br>
                    Continued observation advised.</p>
                </div>""", unsafe_allow_html=True)

            confidence = max(prob, 1 - prob) * 100
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Hazard Probability</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value">{confidence:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Classification</div>
                    <div class="metric-value" style="font-size:1rem">
                        {"Hazardous" if pred == 1 else "Safe"}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Hazard Probability Meter**")
            st.progress(prob, text=f"{prob*100:.1f}% chance of being hazardous")

            avg_d    = (est_diameter_min + est_diameter_max) / 2
            d_range  = est_diameter_max - est_diameter_min
            vd_ratio = relative_velocity / miss_distance if miss_distance > 0 else 0
            with st.expander("🔬 Computed Features Used by Model"):
                st.write(f"**Avg Diameter:** {avg_d:.4f} km")
                st.write(f"**Diameter Range:** {d_range:.4f} km")
                st.write(f"**Velocity/Distance Ratio:** {vd_ratio:.2e}")

            st.session_state.history.append({
                'time'      : datetime.datetime.now().strftime("%H:%M:%S"),
                'vel'       : relative_velocity,
                'dist'      : miss_distance,
                'prob'      : prob,
                'prediction': pred
            })
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.03);border:1px dashed rgba(100,149,237,0.3);
                    border-radius:10px;padding:40px;text-align:center;color:#546e7a;">
            <div style="font-size:2.5rem;margin-bottom:12px;">🔭</div>
            <div style="font-size:1rem;">Fill in the asteroid parameters on the left<br>
            and click <strong style="color:#90caf9">Predict Hazard Level</strong> to classify.</div>
        </div>""", unsafe_allow_html=True)


# ── HISTORY ───────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📋 Prediction History (this session)")
    for entry in reversed(st.session_state.history[-8:]):
        label     = "⚠️ HAZARDOUS" if entry['prediction'] == 1 else "✅ SAFE"
        tag_class = "tag-hazardous" if entry['prediction'] == 1 else "tag-safe"
        st.markdown(f"""
        <div class="history-row">
            <span style="color:#546e7a">{entry['time']}</span>
            <span>Vel: <b>{entry['vel']:,.0f}</b> km/s &nbsp;|&nbsp;
                  Dist: <b>{entry['dist']/1e6:.2f}M</b> km</span>
            <span>Prob: <b>{entry['prob']*100:.1f}%</b></span>
            <span class="{tag_class}">{label}</span>
        </div>""", unsafe_allow_html=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
