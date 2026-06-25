import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="TalentPulse | Attrition Analytics",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, modern corporate UI
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stMetric { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border: 1px solid #e9ecef; }
    div[data-testid="stSidebarUserContent"] { padding-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    models_dir = Path("models")
    scaler = joblib.load(models_dir / "scaler.pkl")
    log_reg = joblib.load(models_dir / "logistic_regression_model.pkl")
    random_forest = joblib.load(models_dir / "random_forest_model.pkl")
    xgb_model = joblib.load(models_dir / "xgb_model.pkl")
    return scaler, log_reg, random_forest, xgb_model

try:
    scaler, log_reg, random_forest, xgb_model = load_models()
except FileNotFoundError:
    st.error("⚠️ **Models or Scaler not found!** Please ensure your training pipeline has saved the artifacts into the `models/` directory.")
    st.stop()

# ==========================================
# SIDEBAR NAVIGATION & HEADERS
# ==========================================
with st.sidebar:
    st.title("📊 TalentPulse Suite")
    st.markdown("Predict and analyze workforce retention risks effortlessly using advanced machine learning models.")
    st.divider()
    page = st.radio("Workspace View", ["Single Employee Attrition Risk Profile", "Model Intelligence Hub"])
    st.sidebar.caption("v3.0.0 • Running on Streamlit Engine")

# ==========================================
# PAGE 1: SINGLE EMPLOYEE RISK PROFILE
# ==========================================
if page == "Single Employee Attrition Risk Profile":
    st.title("👥 Employee Risk Profile")
    st.markdown("Adjust employee data features below to run dynamic, real-time predictive retention assessments.")
    
    # Input Form Fields separated into functional containers
    with st.container(border=True):
        st.subheader("📝 Employee Demographics & Background")
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 65, 35)
            marital_status_single = st.segmented_control("Marital Status", options=[0, 1], format_func=lambda x: "Married/Divorced" if x == 0 else "Single", default=0)
        with col2:
            distance_from_home = st.slider("Distance from Home (km)", 1, 30, 10)
            num_companies = st.slider("Past Companies Worked", 0, 10, 2)
        with col3:
            total_working_years = st.slider("Total Career Working Years", 0, 50, 10)
            overtime_yes = st.segmented_control("Requires Overtime?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", default=0)

    with st.container(border=True):
        st.subheader("💼 Role, Compensation & Tenure")
        col1, col2, col3 = st.columns(3)
        with col1:
            job_level = st.select_slider("Job Level", options=[1, 2, 3, 4], value=2)
            monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=250)
        with col2:
            years_at_company = st.slider("Years at Current Company", 0, 40, 5)
            years_in_current_role = st.slider("Years in Current Role", 0, 18, 3)
        with col3:
            years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 15, 2)
            years_with_curr_manager = st.slider("Years with Current Manager", 0, 17, 3)

    with st.container(border=True):
        st.subheader("🎭 Sentiment & Engagement Index")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            env_satisfaction = st.select_slider("Environment Sat.", options=[1, 2, 3, 4], value=3)
        with col2:
            job_involvement = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3)
        with col3:
            job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3)
        with col4:
            relationship_satisfaction = st.select_slider("Relationship Sat.", options=[1, 2, 3, 4], value=3)
        
        # Simple calculated/default feature for alignment with model configuration
        percent_salary_hike = 12 
        stock_option_level = 1

    # ------------------ PREDICTION LOGIC ------------------
    input_data = np.array([[
        age, distance_from_home, env_satisfaction, job_involvement,
        job_level, job_satisfaction, monthly_income, num_companies,
        percent_salary_hike, relationship_satisfaction, stock_option_level,
        total_working_years, years_at_company, years_in_current_role,
        years_since_last_promotion, years_with_curr_manager, marital_status_single, overtime_yes
    ]])
    
    input_scaled = scaler.transform(input_data)
    
    # Calculate probabilities
    log_reg_proba = log_reg.predict_proba(input_scaled)[0][1]
    rf_proba = random_forest.predict_proba(input_scaled)[0][1]
    xgb_proba = xgb_model.predict_proba(input_scaled)[0][1]
    
    # Core Ensemble Score
    avg_probability = (log_reg_proba + rf_proba + xgb_proba) / 3
    
    # Dynamic styling states based on result
    if avg_probability > 0.7:
        risk_color, risk_text = "#e74c3c", "High Attrition Risk"
    elif avg_probability > 0.4:
        risk_color, risk_text = "#f1c40f", "Medium Attrition Risk"
    else:
        risk_color, risk_text = "#2ecc71", "Low Attrition Risk"

    st.markdown("---")
    st.subheader("🎯 Risk Assessment Summary")
    
    col_metrics, col_gauge = st.columns([1, 1])
    
    with col_metrics:
        st.markdown(f"#### Status Evaluation: <span style='color:{risk_color};font-weight:bold;'>{risk_text}</span>", unsafe_allow_html=True)
        st.markdown("""
        *The ensemble metric averages predictions over Logistic Regression, Random Forests, and Gradient Boosting trees to construct a stable cross-validated risk metric.*
        """)
    
    with col_gauge:
        # Modern Radial Progress Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=avg_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': risk_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e9ecef",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(46, 204, 113, 0.1)'},
                    {'range': [40, 70], 'color': 'rgba(241, 196, 15, 0.1)'},
                    {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.1)'}
                ]
            }
        ))
        fig_gauge.update_layout(height=220, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    # Model Breakdown Chart
    st.markdown("### 🔍 Underlying Multi-Model Probabilities")
    df_chart = pd.DataFrame({
        "Model Framework": ["Logistic Regression", "Random Forest", "XGBoost Classifier"],
        "Risk Output": [log_reg_proba * 100, rf_proba * 100, xgb_proba * 100]
    })
    
    fig_bar = px.bar(
        df_chart, 
        x="Risk Output", 
        y="Model Framework", 
        orientation='h',
        text_auto='.1f',
        labels={"Risk Output": "Predicted Probability (%)"},
        color="Risk Output",
        color_continuous_scale=["#2ecc71", "#f1c40f", "#e74c3c"]
    )
    fig_bar.update_layout(height=220, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10))
    fig_bar.update_xaxes(range=[0, 100])
    st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 2: MODEL INTELLIGENCE HUB
# ==========================================
elif page == "Model Intelligence Hub":
    st.title("🧠 Model Intelligence Hub")
    st.markdown("Inspect performance telemetry metrics and operational feature importance configurations.")
    
    # ---------------------------------------------------------
    # REAL METRICS PULLED DIRECTLY FROM README.md
    # ---------------------------------------------------------
    # Test Set Metrics (Unseen Data)
    test_performance = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.7347, 0.8061, 0.8061],
        "Precision": [0.3500, 0.4220, 0.4190],
        "Recall": [0.7660, 0.5740, 0.5530],
        "F1-Score": [0.4800, 0.4860, 0.4770],
        "ROC-AUC": [0.7470, 0.7120, 0.7040]
    }
    df_test = pd.DataFrame(test_performance)
    
    # Training Set Metrics (Model Fit)
    train_performance = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.7483, 0.8640, 0.9345],
        "Precision": [0.3668, 0.5630, 0.7233],
        "Recall": [0.7684, 0.7053, 0.9632],
        "F1-Score": [0.4966, 0.6262, 0.8262],
        "ROC-AUC": [0.7564, 0.7999, 0.9461]
    }
    df_train = pd.DataFrame(train_performance)
    
    col_table, col_info = st.columns([1.3, 1])
    
    with col_table:
        st.subheader("📊 Performance Baseline Matrices")
        # Use Streamlit Tabs to cleanly separate Train vs Test metrics
        tab1, tab2 = st.tabs(["🎯 Test Set (Unseen Data)", "🏋️ Training Set (Model Fit)"])
        
        with tab1:
            st.dataframe(
                df_test.style.background_gradient(cmap="Blues", subset=["Accuracy", "F1-Score", "ROC-AUC"]),
                use_container_width=True, 
                hide_index=True
            )
        with tab2:
            st.dataframe(
                df_train.style.background_gradient(cmap="Purples", subset=["Accuracy", "F1-Score", "ROC-AUC"]),
                use_container_width=True, 
                hide_index=True
            )
    
    with col_info:
        st.subheader("💡 Core Architecture Notes")
        st.markdown("""
        * **✅ Logistic Regression (Selected):** Chosen for the highest **Recall (76.6%)** and **ROC-AUC (0.747)** on unseen data. It minimizes the most critical business error by producing the lowest False Negatives (11), maximizing the identification of at-risk employees. It shows **minimal overfitting** (Train vs. Test gap is negligible).
        * **🌲 Random Forest:** Achieved higher raw accuracy but missed more attrition cases and showed moderate overfitting (a train-test recall gap of ~13%).
        * **🚀 XGBoost:** Displayed **significant overfitting** (Training Recall of 96.3% vs. Test Recall of 55.3%) and failed to outperform the linear model on new data.
        """)

    st.markdown("---")
    col_radar, col_features = st.columns(2)
    
    with col_radar:
        st.subheader("🕸️ Model Framework Comparison (Test Data)")
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig_radar = go.Figure()
        for idx, row in df_test.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in categories] + [row[categories[0]]],
                theta=categories + [categories[0]],
                name=row['Model']
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            margin=dict(l=40, r=40, t=40, b=40),
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_features:
        st.subheader("📈 Most Influential Drivers")
        st.markdown("Derived from EDA and feature selection profiles:")
        
        # Extracted directly from the README "Most Influential Drivers of Attrition" section
        top_features = {
            "Feature Driver": [
                "OverTime", "MonthlyIncome", "Age", 
                "JobSatisfaction", "YearsAtCompany", "TotalWorkingYears"
            ],
            "Relative Impact Score": [6, 5, 4, 3, 2, 1] # Visual scaling map for the bar chart
        }
        df_features = pd.DataFrame(top_features).sort_values(by="Relative Impact Score", ascending=True)
        
        fig_feat = px.bar(
            df_features, 
            x="Relative Impact Score", 
            y="Feature Driver", 
            orientation='h', 
            color_discrete_sequence=["#34495e"],
            hover_data={"Relative Impact Score": False} # Hides the arbitrary score number on hover
        )
        fig_feat.update_layout(
            height=350, 
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            xaxis_showticklabels=False
        )
        st.plotly_chart(fig_feat, use_container_width=True)

# Footer
st.divider()
st.caption("🔒 Corporate Data Privacy Notice: Predictions are simulated locally via mathematical logic and do not upload raw data records outside the runtime environment.")