# file: app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model CoxPH
with open("cox_model.pkl", "rb") as f:
    cox_model = pickle.load(f)

# Fungsi perhitungan premi
def calculate_premium(patient_data, horizon_days=1825,
                      expected_claim=80_000_000, loading=0.25):
    # Validasi kolom
    required_cols = cox_model.params_.index.tolist()
    for col in required_cols:
        if col not in patient_data:
            raise ValueError(f"Missing column: {col}")
    
    df = pd.DataFrame([patient_data])
    surv_func = cox_model.predict_survival_function(df)
    closest_idx = np.argmin(np.abs(surv_func.index - horizon_days))
    survival_prob = float(surv_func.iloc[closest_idx, 0])
    death_risk = 1 - survival_prob
    expected_loss = death_risk * expected_claim
    premium = expected_loss * (1 + loading)
    return survival_prob, death_risk, premium

# Streamlit layout
st.title("🩺 Survival Premium Calculator Hemodialisa Patient")
st.write("Aplikasi ini menghitung probabilitas bertahan hidup dan premi asuransi berdasarkan data pasien.")

# Input pasien
age = st.number_input("Umur", min_value=0, max_value=120, value=60)
sex = st.selectbox("Jenis Kelamin", options=[("Laki-laki",1), ("Perempuan",0)])
diabetes = st.selectbox("Diabetes", options=[("Ya",1), ("Tidak",0)])
hypertension = st.selectbox("Hipertensi", options=[("Ya",1), ("Tidak",0)])
hb = st.number_input("Hemoglobin (g/dL)", value=12.0)
albumin = st.number_input("Albumin (g/dL)", value=3.5)
k = st.number_input("Kalium (mmol/L)", value=4.5)
phosphate = st.number_input("Fosfat (mg/dL)", value=4.5)
dialysis_freq = st.number_input("Frekuensi Dialisis per minggu", min_value=0, max_value=7, value=3)
access_catheter = st.selectbox("Akses Kateter", options=[("Ya",1), ("Tidak",0)])

# Tombol prediksi
if st.button("Hitung Premi dan Risiko"):
    patient_data = {
        "age": age,
        "sex": sex[1],
        "diabetes": diabetes[1],
        "hypertension": hypertension[1],
        "hb": hb,
        "albumin": albumin,
        "k": k,
        "phosphate": phosphate,
        "dialysis_freq": dialysis_freq,
        "access_catheter": access_catheter[1]
    }
    
    try:
        survival_prob, death_risk, premium = calculate_premium(patient_data)
        
        # Interpretasi
        if survival_prob >= 0.8:
            risk_desc = "Risiko rendah, prognosis pasien baik."
        elif survival_prob >= 0.5:
            risk_desc = "Risiko sedang, perlu monitoring lebih lanjut."
        else:
            risk_desc = "Risiko tinggi, prognosis buruk."

        if premium < 5_000_000:
            premi_desc = "Premi rendah, risiko klaim kecil."
        elif premium < 15_000_000:
            premi_desc = "Premi sedang, risiko klaim moderat."
        else:
            premi_desc = "Premi tinggi, risiko klaim besar."
        
        # Output
        st.success("✅ Hasil Prediksi:")
        st.write(f"- Probabilitas Bertahan Hidup: {survival_prob:.2f} ({risk_desc})")
        st.write(f"- Premi Asuransi: Rp {premium:,.2f} ({premi_desc})")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
