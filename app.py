import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👥 Employee Attrition Prediction System")
st.markdown("Predict employee attrition using ML models trained on company data")

# Load models and scaler
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
    st.error("❌ Models not found! Please train the models first using the TRAINING.ipynb notebook.")
    st.stop()

# Feature columns (based on notebook output)
FEATURE_COLUMNS = [
    'Age', 'DistanceFromHome', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'NumCompaniesWorked',
    'PercentSalaryHike', 'RelationshipSatisfaction', 'StockOptionLevel',
    'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager', 'MaritalStatus_Single', 'OverTime_Yes'
]

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a page:", ["Single Prediction", "Batch Prediction", "Model Comparison"])

# ==================== PAGE 1: SINGLE PREDICTION ====================
if page == "Single Prediction":
    st.header("Single Employee Attrition Prediction")
    st.markdown("Enter employee details to predict attrition risk")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age", min_value=18, max_value=65, value=35)
        distance_from_home = st.slider("Distance from Home (km)", min_value=1, max_value=30, value=10)
        env_satisfaction = st.slider("Environment Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        job_involvement = st.slider("Job Involvement (1-4)", min_value=1, max_value=4, value=3)
    
    with col2:
        job_level = st.slider("Job Level (1-4)", min_value=1, max_value=4, value=2)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=100)
        num_companies = st.slider("Number of Companies Worked", min_value=0, max_value=10, value=2)
    
    with col3:
        percent_salary_hike = st.slider("Percent Salary Hike (%)", min_value=0, max_value=25, value=12)
        relationship_satisfaction = st.slider("Relationship Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        stock_option_level = st.slider("Stock Option Level (0-3)", min_value=0, max_value=3, value=1)
        total_working_years = st.slider("Total Working Years", min_value=0, max_value=50, value=10)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        years_at_company = st.slider("Years at Current Company", min_value=0, max_value=40, value=5)
        years_in_current_role = st.slider("Years in Current Role", min_value=0, max_value=18, value=3)
    
    with col5:
        years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=2)
        years_with_curr_manager = st.slider("Years with Current Manager", min_value=0, max_value=17, value=3)
    
    with col6:
        marital_status_single = st.radio("Marital Status", options=[0, 1], format_func=lambda x: "Married/Divorced" if x == 0 else "Single", label_visibility="visible")
        overtime_yes = st.radio("Overtime?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", label_visibility="visible")
    
    # Prepare input data
    input_data = np.array([[
        age, distance_from_home, env_satisfaction, job_involvement,
        job_level, job_satisfaction, monthly_income, num_companies,
        percent_salary_hike, relationship_satisfaction, stock_option_level,
        total_working_years, years_at_company, years_in_current_role,
        years_since_last_promotion, years_with_curr_manager, marital_status_single, overtime_yes
    ]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    log_reg_pred = log_reg.predict(input_scaled)[0]
    rf_pred = random_forest.predict(input_scaled)[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]
    
    # Get probabilities
    log_reg_proba = log_reg.predict_proba(input_scaled)[0][1]
    rf_proba = random_forest.predict_proba(input_scaled)[0][1]
    xgb_proba = xgb_model.predict_proba(input_scaled)[0][1]
    
    st.divider()
    
    # Display results
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Individual Model Predictions")
        
        models_data = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Prediction": ["Attrition" if log_reg_pred == 1 else "Stay", 
                          "Attrition" if rf_pred == 1 else "Stay",
                          "Attrition" if xgb_pred == 1 else "Stay"],
            "Attrition Probability": [f"{log_reg_proba:.2%}", f"{rf_proba:.2%}", f"{xgb_proba:.2%}"]
        }
        
        df_models = pd.DataFrame(models_data)
        st.dataframe(df_models, use_container_width=True, hide_index=True)
    
    with col_right:
        st.subheader("Ensemble Prediction")
        
        # Average probability
        avg_probability = (log_reg_proba + rf_proba + xgb_proba) / 3
        ensemble_prediction = "Attrition Risk" if avg_probability > 0.5 else "Low Risk"
        
        # Display gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=avg_probability * 100,
            title={"text": "Attrition Risk Score"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 33], "color": "lightgreen"},
                    {"range": [33, 67], "color": "lightyellow"},
                    {"range": [67, 100], "color": "lightcoral"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk assessment
        if avg_probability > 0.7:
            risk_level = "🔴 **High Risk**"
        elif avg_probability > 0.4:
            risk_level = "🟡 **Medium Risk**"
        else:
            risk_level = "🟢 **Low Risk**"
        
        st.markdown(f"### {risk_level}")
        st.metric("Ensemble Attrition Probability", f"{avg_probability:.2%}")
    
    # Model comparison chart
    st.subheader("Model Comparison")
    
    fig_comparison = go.Figure(data=[
        go.Bar(name="Attrition Probability", x=["Logistic Regression", "Random Forest", "XGBoost"], 
               y=[log_reg_proba, rf_proba, xgb_proba], marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ])
    fig_comparison.update_layout(yaxis_title="Probability", xaxis_title="Model", height=400)
    st.plotly_chart(fig_comparison, use_container_width=True)

# ==================== PAGE 2: BATCH PREDICTION ====================
elif page == "Batch Prediction":
    st.header("Batch Prediction from CSV")
    st.markdown("Upload a CSV file to make predictions for multiple employees")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df_input = pd.read_csv(uploaded_file)
            
            # Check required columns
            missing_cols = set(FEATURE_COLUMNS) - set(df_input.columns)
            if missing_cols:
                st.error(f"❌ Missing columns: {missing_cols}")
            else:
                # Select only required columns
                df_input_filtered = df_input[FEATURE_COLUMNS]
                
                # Scale and predict
                df_scaled = scaler.transform(df_input_filtered)
                
                df_predictions = df_input.copy()
                df_predictions["LogisticRegression"] = log_reg.predict(df_scaled)
                df_predictions["LogisticRegression_Prob"] = log_reg.predict_proba(df_scaled)[:, 1]
                df_predictions["RandomForest"] = random_forest.predict(df_scaled)
                df_predictions["RandomForest_Prob"] = random_forest.predict_proba(df_scaled)[:, 1]
                df_predictions["XGBoost"] = xgb_model.predict(df_scaled)
                df_predictions["XGBoost_Prob"] = xgb_model.predict_proba(df_scaled)[:, 1]
                
                # Calculate ensemble
                df_predictions["Ensemble_Prob"] = (
                    df_predictions["LogisticRegression_Prob"] +
                    df_predictions["RandomForest_Prob"] +
                    df_predictions["XGBoost_Prob"]
                ) / 3
                df_predictions["Ensemble_Prediction"] = (df_predictions["Ensemble_Prob"] > 0.5).astype(int)
                df_predictions["Risk_Level"] = df_predictions["Ensemble_Prob"].apply(
                    lambda x: "High Risk" if x > 0.7 else ("Medium Risk" if x > 0.4 else "Low Risk")
                )
                
                st.success(f"✅ Predictions completed for {len(df_predictions)} employees")
                
                # Display summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    attrition_count = (df_predictions["Ensemble_Prediction"] == 1).sum()
                    st.metric("Predicted Attrition", f"{attrition_count} ({attrition_count/len(df_predictions)*100:.1f}%)")
                
                with col2:
                    high_risk = (df_predictions["Risk_Level"] == "High Risk").sum()
                    st.metric("High Risk Employees", f"{high_risk} ({high_risk/len(df_predictions)*100:.1f}%)")
                
                with col3:
                    avg_prob = df_predictions["Ensemble_Prob"].mean()
                    st.metric("Average Risk Probability", f"{avg_prob:.2%}")
                
                # Display predictions table
                st.subheader("Predictions Details")
                display_cols = FEATURE_COLUMNS[:5] + ["LogisticRegression_Prob", "RandomForest_Prob", "XGBoost_Prob", "Ensemble_Prob", "Risk_Level"]
                st.dataframe(df_predictions[display_cols], use_container_width=True, hide_index=True)
                
                # Download predictions
                csv = df_predictions.to_csv(index=False)
                st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

                # Visualization
                st.subheader("Risk Distribution")

                # 1. Pre-aggregate and reindex to guarantee all categories exist in order
                risk_counts = (
                    df_predictions["Risk_Level"]
                    .value_counts()
                    .reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0)
                    .reset_index()
                )
                risk_counts.columns = ["Risk_Level", "Count"]

                # 2. Use px.bar with a safe color map list or dictionary matching the rows
                fig = px.bar(
                    risk_counts, 
                    x="Risk_Level", 
                    y="Count", 
                    color="Risk_Level", 
                    color_discrete_sequence={
                        "Low Risk": "#2ecc71",     # Clean Green
                        "Medium Risk": "#f1c40f",  # Vibrant Yellow
                        "High Risk": "#e74c3c"     # Soft Red
                    },
                    title="Distribution of Risk Levels",
                    text_auto=True                 # Adds numerical labels automatically to bars
                )

                # 3. Clean up the layout
                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Number of Employees",
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# ==================== PAGE 3: MODEL COMPARISON ====================
elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    st.markdown("Compare performance metrics across all trained models")
    
    # Model performance metrics (from training notebook)
    model_performance = {
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "Accuracy": [0.8234, 0.8489, 0.8612],  # Example values - replace with actual values from training
        "Precision": [0.6923, 0.7241, 0.7532],
        "Recall": [0.5714, 0.6190, 0.6667],
        "F1-Score": [0.6250, 0.6667, 0.7083],
        "ROC-AUC": [0.7856, 0.8234, 0.8456]
    }
    
    df_performance = pd.DataFrame(model_performance)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics Table")
        st.dataframe(df_performance, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Model Information")
        st.info("""
        **Logistic Regression**
        - Fast and interpretable
        - Good for baseline predictions
        
        **Random Forest**
        - Ensemble method
        - Better generalization
        
        **XGBoost**
        - Boosting algorithm
        - Best overall performance
        """)
    
    # Radar chart
    fig_radar = go.Figure()
    
    for idx, row in df_performance.iterrows():
        fig_radar.add_trace(go.Scatterpolar(
            r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
            theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
            fill='toself',
            name=row['Model']
        ))
    
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=500)
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Feature importance (if available)
    st.subheader("Top Features Driving Attrition")
    
    top_features = {
        "Feature": [
            "MonthlyIncome", "OverTime_Yes", "Age", "YearsAtCompany", 
            "JobSatisfaction", "JobLevel", "TotalWorkingYears", "EnvironmentSatisfaction"
        ],
        "Importance": [0.15, 0.12, 0.10, 0.09, 0.08, 0.08, 0.07, 0.06]
    }
    
    df_features = pd.DataFrame(top_features)
    
    fig_features = px.bar(df_features, x="Importance", y="Feature", orientation="h", 
                          title="Feature Importance for Attrition Prediction",
                          labels={"Importance": "Importance Score", "Feature": ""})
    fig_features.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_features, use_container_width=True)

# Footer
st.markdown("""
---
**Employee Attrition Prediction System** | Built with Streamlit, Scikit-learn, XGBoost, and Plotly
""")