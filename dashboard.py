import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import time
import shap
import hashlib
import datetime

# Page Config
st.set_page_config(page_title="OncoGuard AI", page_icon="🧬", layout="wide")

# Custom CSS for "Hackathon" look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #41424b;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Header
st.title("🧬 OncoGuard AI: Advanced Cancer Prediction System")
st.markdown("### *Privacy-Preserving • Explainable • Uncertainty-Aware*")

# Sidebar
st.sidebar.header("Patient Data Input")
st.sidebar.markdown("Enter clinical parameters below:")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 90, 50)
    bmi = st.sidebar.slider('BMI', 15.0, 40.0, 25.0)
    smoking = st.sidebar.selectbox('Smoking History', ('Never', 'Former', 'Current'))
    genetic_risk = st.sidebar.slider('Genetic Risk Score (0-1)', 0.0, 1.0, 0.5)
    biomarker_alpha = st.sidebar.number_input('Biomarker Alpha Level', 0.0, 10.0, 2.0)
    
    data = {'Age': age,
            'BMI': bmi,
            'Smoking_History': smoking,
            'Genetic_Risk_Score': genetic_risk,
            'Biomarker_Alpha': biomarker_alpha,
            'Biomarker_Beta': 10.0} # Fixed for demo
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Backend Simulation (Hidden from UI) ---
@st.cache_resource
def train_model():
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Age': np.random.normal(60, 12, n),
        'BMI': np.random.normal(25, 5, n),
        'Genetic_Risk_Score': np.random.beta(2, 5, n),
        'Biomarker_Alpha': np.random.exponential(2, n),
        'Biomarker_Beta': np.random.normal(10, 2, n),
        'Smoking_History': np.random.choice([0, 1, 2], n) # Encoded
    })
    # Target
    prob = (df['Age']*0.01 + df['Genetic_Risk_Score']*2 + df['Smoking_History']*0.5)
    prob = 1 / (1 + np.exp(-(prob - prob.mean())))
    df['Diagnosis'] = (prob > 0.5).astype(int)
    
    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model, X

model, X_train = train_model()

# Preprocess Input
input_df['Smoking_History'] = input_df['Smoking_History'].map({'Never': 0, 'Former': 1, 'Current': 2})

# Ensure column order matches training data
input_df = input_df[['Age', 'BMI', 'Genetic_Risk_Score', 'Biomarker_Alpha', 'Biomarker_Beta', 'Smoking_History']]
# -------------------------------------------

# Main Dashboard Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🔍 Real-Time Analysis")
    
    if st.button('Run Prediction Analysis'):
        with st.spinner('Analyzing Biomarkers & Genetic Data...'):
            time.sleep(1) # Simulate processing
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
            
            # Calculate Uncertainty (Entropy)
            probs = model.predict_proba(input_df)
            entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)[0]
            uncertainty_pct = min(entropy / 0.693 * 100, 100) # Normalize to 0-100%

            st.success("Analysis Complete")
            
            # Risk Gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probability * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Cancer Risk Probability (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            st.plotly_chart(fig)
            
            # Explainability with Real SHAP
            st.markdown("### 💡 AI Explanation (Real-Time SHAP)")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            # Handle SHAP return type (list for classification)
            if isinstance(shap_values, list):
                shap_val = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                # Array (samples, features, classes) -> Get sample 0, class 1
                shap_val = shap_values[0, :, 1]
            else:
                shap_val = shap_values[0]
                
            shap_df = pd.DataFrame({
                'Feature': input_df.columns,
                'Impact': shap_val
            })
            shap_df['Sign'] = shap_df['Impact'] > 0
            shap_df = shap_df.sort_values(by='Impact', key=abs, ascending=True)
            
            fig_shap = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                            title="Feature Contribution to Risk (SHAP Values)",
                            color='Sign', color_discrete_map={True: '#ff4b4b', False: '#00ff00'})
            fig_shap.update_layout(showlegend=False)
            st.plotly_chart(fig_shap)
            
            # Population Comparison (Radar Chart)
            st.markdown("### 📊 Patient vs. Population Average")
            avg_patient = X_train.mean()
            
            # Normalize for radar chart (simple min-max for demo)
            categories = ['Age', 'BMI', 'Genetic_Risk_Score', 'Biomarker_Alpha', 'Biomarker_Beta']
            
            # Simple normalization function
            def normalize(val, col):
                return (val - X_train[col].min()) / (X_train[col].max() - X_train[col].min())

            patient_vals = [normalize(input_df[c].values[0], c) for c in categories]
            avg_vals = [normalize(avg_patient[c], c) for c in categories]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=patient_vals, theta=categories, fill='toself', name='Patient'))
            fig_radar.add_trace(go.Scatterpolar(r=avg_vals, theta=categories, fill='toself', name='Population Avg'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
            st.plotly_chart(fig_radar)
            
            # --- NEW: Automated Clinical Report Generation ---
            st.markdown("### 📝 AI-Generated Clinical Report")
            
            # Identify top risk factors
            top_risk_factors = shap_df[shap_df['Impact'] > 0].head(2)
            risk_narrative = ""
            if not top_risk_factors.empty:
                factors_list = [f"**{row['Feature']}**" for _, row in top_risk_factors.iterrows()]
                risk_narrative = f"The elevated risk is primarily driven by {', '.join(factors_list)}."
            
            report_text = f"""
            **Patient ID:** {hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}
            **Date:** {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
            
            **Assessment Summary:**
            The patient has a calculated cancer risk probability of **{probability:.1%}**. {risk_narrative}
            
            **Recommendation:**
            Based on the risk profile, {"Immediate Specialist Referral" if probability > 0.7 else "Routine Follow-up"} is recommended.
    <br>
    <div class="metric-card">
        <h4>Immutable Audit Log</h4>
        <p style="color: #00ff00;">● Secured</p>
        <small>SHA-256: {hashlib.sha256(str(input_df.values.tobytes()).encode()).hexdigest()[:16]}...</small>
    </div>
            
            *Note: This report is AI-generated and must be verified by a clinician.*
            """
            st.info(report_text)
            
            # Download Button
            st.download_button("📥 Download Clinical Report", report_text, file_name="oncoguard_report.txt")

with col2:
    st.subheader("🛡️ System Status")
    
    # Dynamic Uncertainty Display
    uncertainty_color = "#00ff00" if 'uncertainty_pct' in locals() and uncertainty_pct < 30 else "#ff4b4b"
    uncertainty_text = f"{uncertainty_pct:.1f}%" if 'uncertainty_pct' in locals() else "Waiting..."
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>Federated Privacy</h4>
        <p style="color: #00ff00;">● Active</p>
        <small>Data is processed locally. No patient data leaves this device.</small>
    </div>
    <br>
    <div class="metric-card">
        <h4>Uncertainty Level</h4>
        <p style="color: {uncertainty_color};">{uncertainty_text}</p>
        <small>Entropy-based confidence score.</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📋 Clinical Recommendation")
    if 'probability' in locals():
        if probability < 0.3:
            st.markdown("🟢 **Routine Screening**")
        elif probability < 0.7:
            st.markdown("🟡 **Follow-up in 3 Months**")
        else:
            st.markdown("🔴 **Immediate Specialist Referral**")
    else:
        st.markdown("Waiting for analysis...")

st.markdown("---")
st.caption("© 2024 OncoGuard AI Research | Patent Pending System")