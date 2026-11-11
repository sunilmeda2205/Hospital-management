import streamlit as st
import pandas as pd
import joblib
import os
import time

st.set_page_config(page_title="Emergency Admission Prioritization", layout="wide", page_icon="🚑")

# --- Styling ---
st.markdown("""
<style>
    .critical-row {
        background-color: #ffcccc;
        font-weight: bold;
        color: #990000;
    }
    .medium-row {
        background-color: #fff3cd;
        color: #856404;
    }
    .low-row {
        background-color: #d4edda;
        color: #155724;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading & Model Setup ---
@st.cache_resource
def load_ml_resources():
    model_path = os.path.join('models', 'rf_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

model, scaler = load_ml_resources()

# Initialize Session State for patients queue
if 'patients_queue' not in st.session_state:
    st.session_state['patients_queue'] = []
    
if 'patient_id' not in st.session_state:
    st.session_state['patient_id'] = 1

# Mock Hospital Data
hospitals = pd.DataFrame([
    {"Name": "City General Hospital", "Distance (km)": 2.5, "Critical Beds": 5, "Medium Beds": 12, "Low Beds": 30},
    {"Name": "St. Mary's Medical Center", "Distance (km)": 4.1, "Critical Beds": 0, "Medium Beds": 8, "Low Beds": 15},
    {"Name": "Sunrise Community Hospital", "Distance (km)": 6.8, "Critical Beds": 2, "Medium Beds": 20, "Low Beds": 50},
    {"Name": "Mercy Trauma Center", "Distance (km)": 8.0, "Critical Beds": 12, "Medium Beds": 5, "Low Beds": 10}
])


# --- App Layout ---
st.title("🚑 Emergency Care Admission Prioritization")
st.markdown("Machine Learning based triage and hospital bed allocation system.")

tab1, tab2, tab3 = st.tabs(["📝 Intake Patient", "📋 Prioritized Queue", "🏥 Nearby Hospitals"])

# --- Tab 1: Intake Patient ---
with tab1:
    st.header("New Patient Intake Form")
    
    if not model or not scaler:
        st.error("⚠️ Machine Learning model not loaded. Please ensure the model is trained.")
    else:
        with st.form("intake_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Patient Name")
                age = st.number_input("Age", min_value=0, max_value=120, value=30)
                heart_rate = st.number_input("Heart Rate (bpm)", min_value=0, max_value=300, value=80)
                sys_bp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=120)
                dia_bp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=80)
                
            with col2:
                resp_rate = st.number_input("Respiratory Rate (breaths/min)", min_value=0, max_value=100, value=16)
                temp = st.number_input("Temperature (°C)", min_value=20.0, max_value=45.0, value=37.0, format="%.1f")
                spo2 = st.number_input("SpO2 (%)", min_value=0, max_value=100, value=98)
                symptoms = st.text_area("Symptoms & Notes")
                
            submit_button = st.form_submit_button("Submit & Assess Priority")
            
            if submit_button:
                if not name:
                    st.warning("Please enter a patient name.")
                else:
                    # ML Prediction
                    features = pd.DataFrame([[age, heart_rate, sys_bp, dia_bp, resp_rate, temp, spo2]], 
                                            columns=['Age', 'Heart_Rate', 'Systolic_BP', 'Diastolic_BP', 'Respiratory_Rate', 'Temperature', 'SpO2'])
                    
                    scaled_features = scaler.transform(features)
                    prediction = model.predict(scaled_features)[0]
                    
                    patient_record = {
                        "ID": st.session_state['patient_id'],
                        "Name": name,
                        "Age": age,
                        "Vitals": f"HR:{heart_rate}, BP:{sys_bp}/{dia_bp}, SpO2:{spo2}%",
                        "Symptoms": symptoms if symptoms else "None reported",
                        "Priority": prediction,
                        "Status": "Waiting"
                    }
                    
                    st.session_state['patients_queue'].append(patient_record)
                    st.session_state['patient_id'] += 1
                    
                    st.success(f"Patient {name} admitted successfully!")
                    
                    # Highlight priority
                    if prediction == "Critical":
                        st.error(f"⚠️ Assessed Priority: **{prediction}**")
                    elif prediction == "Medium":
                        st.warning(f"⚠️ Assessed Priority: **{prediction}**")
                    else:
                        st.success(f"✅ Assessed Priority: **{prediction}**")

# --- Tab 2: Prioritized Queue ---
with tab2:
    st.header("Emergency Department Decision Support")
    st.markdown("Patients sorted dynamically by Machine Learning risk assessment.")
    
    if not st.session_state['patients_queue']:
        st.info("The queue is currently empty.")
    else:
        # Convert to DataFrame and sort
        df_queue = pd.DataFrame(st.session_state['patients_queue'])
        
        # Priority mapping for sorting
        priority_map = {"Critical": 0, "Medium": 1, "Low": 2}
        df_queue['Sort_Index'] = df_queue['Priority'].map(priority_map)
        df_queue = df_queue.sort_values(by=['Sort_Index', 'ID']).drop('Sort_Index', axis=1)
        
        # Styling the dataframe
        def highlight_priority(row):
            if row['Priority'] == 'Critical':
                return ['background-color: #ffcccc'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #fff3cd'] * len(row)
            else:
                return ['background-color: #d4edda'] * len(row)
                
        st.dataframe(
            df_queue.style.apply(highlight_priority, axis=1), 
            use_container_width=True, 
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", format="%d"),
                "Priority": st.column_config.TextColumn("Priority", width="medium")
            }
        )
        
        # Actions
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Clear Queue"):
                st.session_state['patients_queue'] = []
                st.session_state['patient_id'] = 1
                st.rerun()

# --- Tab 3: Nearby Hospitals ---
with tab3:
    st.header("Nearby Hospital Availability")
    
    st.dataframe(
        hospitals,
        use_container_width=True,
        hide_index=True
    )
    
    st.info("💡 In a production system, this data would be fetched via real-time APIs from regional hospital networks.")
