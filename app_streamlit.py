# file: app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

# Load model CoxPH
with open("cox_model.pkl", "rb") as f:
    cox_model = pickle.load(f)

# Fungsi perhitungan premi
def calculate_premium(patient_data, horizon_days=1825,
                      expected_claim=80_000_000, loading=0.25):
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
    return survival_prob, death_risk, premium, surv_func

# Fungsi interpretasi faktor risiko + saran
def interpret_factors_and_suggestions(patient_data):
    factors = []
    suggestions = []

    if patient_data["age"] >= 65:
        factors.append("usia lanjut (≥65 thn)")
        suggestions.append("monitoring ketat diperlukan untuk pasien lansia.")
    if patient_data["hb"] < 10:
        factors.append("hemoglobin rendah (<10 g/dL / anemia)")
        suggestions.append("perlukan koreksi anemia, misalnya EPO atau transfusi.")
    if patient_data["albumin"] < 3.5:
        factors.append("albumin rendah (<3.5 g/dL / malnutrisi/inflamasi)")
        suggestions.append("perlu perbaikan status nutrisi dan pengendalian inflamasi.")
    if patient_data["k"] < 3.5 or patient_data["k"] > 5.0:
        factors.append("kadar kalium abnormal")
        suggestions.append("perlu koreksi kadar kalium untuk mencegah aritmia.")
    if patient_data["phosphate"] > 5.5:
        factors.append("fosfat tinggi")
        suggestions.append("disarankan kontrol fosfat melalui diet/obat pengikat fosfat.")
    if patient_data["access_catheter"] == 1:
        factors.append("akses kateter (risiko infeksi/trombosis)")
        suggestions.append("pertimbangkan fistula AV untuk menurunkan risiko.")
    if patient_data["dialysis_freq"] < 3:
        factors.append("frekuensi dialisis <3 kali/minggu")
        suggestions.append("perlu peningkatan dialisis ≥3 kali/minggu untuk prognosis lebih baik.")

    if not factors:
        factors.append("tidak ada faktor risiko berat yang menonjol")
        suggestions.append("lakukan monitoring rutin sesuai protokol standar.")

    return factors, suggestions

# Fungsi rekomendasi underwriting
def underwriting_recommendation(survival_prob, premium):
    if survival_prob >= 0.8 and premium < 10_000_000:
        return "Polis dapat diterima tanpa loading tambahan."
    elif survival_prob >= 0.5:
        return "Polis dapat diterima dengan loading premi tambahan karena risiko sedang."
    else:
        return "Polis berisiko tinggi, sebaiknya ditinjau ulang atau memerlukan evaluasi medis lebih lanjut."

# Fungsi untuk membuat interpretasi naratif
def narrative_interpretation(survival_prob, premium, factors, suggestions, rekomendasi, loading, expected_claim):
    prognosis = ""
    if survival_prob >= 0.8:
        prognosis = "prognosis pasien secara umum baik dengan risiko kematian rendah."
    elif survival_prob >= 0.5:
        prognosis = "prognosis pasien sedang dengan risiko kematian moderat."
    else:
        prognosis = "prognosis pasien kurang baik dengan risiko kematian tinggi."

    premi_text = ""
    if premium < 5_000_000:
        premi_text = "Premi yang dihasilkan relatif rendah, mencerminkan risiko klaim yang kecil."
    elif premium < 15_000_000:
        premi_text = "Premi berada pada kategori sedang, menunjukkan adanya risiko klaim moderat."
    else:
        premi_text = "Premi cukup tinggi, mencerminkan risiko klaim yang besar."

    factor_text = ", ".join(factors)
    suggestion_text = " ".join(suggestions)

    return (f"Berdasarkan hasil perhitungan, probabilitas bertahan hidup pasien dalam 5 tahun adalah {survival_prob:.2f}, "
            f"sehingga {prognosis} {premi_text} Faktor risiko utama yang teridentifikasi meliputi {factor_text}. "
            f"Oleh karena itu, {suggestion_text} Dengan loading {loading*100:.0f}% dan nilai klaim yang diasumsikan sebesar Rp {expected_claim:,.0f}, "
            f"nilai premi yang dihasilkan adalah Rp {premium:,.2f}. Dari sisi aktuaria, {rekomendasi}")

# Fungsi Monte Carlo Simulation
def monte_carlo_premium(survival_prob, expected_claim, loading, n_sim=1000, variation=0.1):
    """
    Simulasi distribusi premi dengan variasi survival probability ±variation.
    """
    simulated_probs = np.random.normal(loc=survival_prob, scale=variation*survival_prob, size=n_sim)
    simulated_probs = np.clip(simulated_probs, 0, 1)  # pastikan antara 0-1
    simulated_deaths = 1 - simulated_probs
    simulated_premiums = simulated_deaths * expected_claim * (1 + loading)
    return simulated_premiums

def interpret_montecarlo(simulated_premiums):
    mean_val = np.mean(simulated_premiums)
    median_val = np.median(simulated_premiums)
    ci_lower, ci_upper = np.percentile(simulated_premiums, [2.5, 97.5])
    
    # Interpretasi level premi
    if mean_val < 5_000_000:
        kategori = "rendah"
    elif mean_val < 15_000_000:
        kategori = "sedang"
    else:
        kategori = "tinggi"
    
    # Narasi
    interpretasi = (
        f"Hasil simulasi Monte Carlo menunjukkan bahwa rata-rata premi berada pada kisaran Rp {mean_val:,.0f}, "
        f"dengan median sebesar Rp {median_val:,.0f}. Interval kepercayaan 95% berada pada rentang Rp {ci_lower:,.0f} "
        f"sampai Rp {ci_upper:,.0f}, yang menunjukkan variasi premi {kategori}. "
    )
    
    # Apakah distribusi lebar?
    if (ci_upper - ci_lower) / mean_val > 1:
        interpretasi += "Rentang ini relatif lebar, menandakan adanya ketidakpastian tinggi dalam estimasi premi. "
    else:
        interpretasi += "Rentang ini relatif sempit, sehingga estimasi premi cukup stabil. "
    
    # Outlier check
    if np.max(simulated_premiums) > mean_val * 2:
        interpretasi += "Terdapat kemungkinan outlier premi yang jauh lebih tinggi dari rata-rata, sehingga perlu kehati-hatian dalam penentuan premi final."
    else:
        interpretasi += "Distribusi premi tidak menunjukkan adanya outlier ekstrem."
    
    return interpretasi

# Streamlit layout
st.title("🩺 Survival Premium Calculator Hemodialysis Patients")
st.write("Aplikasi ini menghitung probabilitas bertahan hidup, risiko klaim, dan premi asuransi berdasarkan data pasien, dengan interpretasi naratif & simulasi Monte Carlo.")

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

# Input loading & expected claim
loading = st.slider("Loading (%)", min_value=0, max_value=100, value=25, step=5) / 100
expected_claim = st.number_input("Expected Claim (Rp)", min_value=1_000_000, max_value=1_000_000_000, value=80_000_000, step=1_000_000)

# Input Monte Carlo config
st.subheader("⚙️ Pengaturan Simulasi Monte Carlo")
n_sim = st.number_input("Jumlah Simulasi", min_value=500, max_value=10000, value=2000, step=500)
variation = st.slider("Variasi Survival Probability (%)", min_value=1, max_value=30, value=10, step=1) / 100

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
        survival_prob, death_risk, premium, surv_func = calculate_premium(
            patient_data, loading=loading, expected_claim=expected_claim
        )

        # Interpretasi faktor + saran
        faktor_desc, saran_desc = interpret_factors_and_suggestions(patient_data)

        # Rekomendasi underwriting
        rekomendasi = underwriting_recommendation(survival_prob, premium)

        # Interpretasi naratif
        narrative = narrative_interpretation(survival_prob, premium, faktor_desc, saran_desc, rekomendasi, loading, expected_claim)

        # ✅ Bagian output utama
        st.success("✅ Hasil Prediksi Utama:")
        st.write(f"- Probabilitas Bertahan Hidup (5 tahun): {survival_prob:.2f}")
        st.write(f"- Premi Asuransi (Loading {loading*100:.0f}%, Expected Claim Rp {expected_claim:,.0f}): Rp {premium:,.2f}")

        st.subheader("📌 Interpretasi Naratif")
        st.write(narrative)

        # ✅ Output tambahan survival horizons
        st.subheader("📊 Survival Probability pada Horizon Berbeda")
        horizons = {"1 Tahun":365, "5 Tahun":1825, "10 Tahun":3650}
        for label, days in horizons.items():
            prob, _, prem, _ = calculate_premium(patient_data, horizon_days=days, loading=loading, expected_claim=expected_claim)
            st.write(f"{label}: Probabilitas {prob:.2f}, Premi ≈ Rp {prem:,.2f}")

        # ✅ Grafik survival curve
        st.subheader("📈 Kurva Survival Pasien")
        fig, ax = plt.subplots()
        surv_func.plot(ax=ax)
        ax.set_xlabel("Hari")
        ax.set_ylabel("Probabilitas Bertahan Hidup")
        ax.set_title("Kurva Survival Prediksi")
        st.pyplot(fig)

        # ✅ Monte Carlo Simulation
        st.subheader("🎲 Simulasi Monte Carlo Premi")
        simulated_premiums = monte_carlo_premium(survival_prob, expected_claim, loading, n_sim=n_sim, variation=variation)
        # Histogram
        fig2, ax2 = plt.subplots()
        ax2.hist(simulated_premiums, bins=30, edgecolor="black")
        ax2.set_xlabel("Premi (Rp)")
        ax2.set_ylabel("Frekuensi")
        ax2.set_title(f"Distribusi Premi (Monte Carlo, N={n_sim}, Variasi={variation*100:.0f}%)")
        st.pyplot(fig2)
        
        # Statistik ringkasan
        st.write("📊 Ringkasan Statistik Premi (Monte Carlo):")
        st.write(f"- Mean: Rp {np.mean(simulated_premiums):,.2f}")
        st.write(f"- Median: Rp {np.median(simulated_premiums):,.2f}")
        st.write(f"- Minimum: Rp {np.min(simulated_premiums):,.2f}")
        st.write(f"- Maximum: Rp {np.max(simulated_premiums):,.2f}")
        st.write(f"- 95% CI: Rp {np.percentile(simulated_premiums, 2.5):,.2f} – Rp {np.percentile(simulated_premiums, 97.5):,.2f}")

        # ✅ Interpretasi Monte Carlo
        st.subheader("📌 Interpretasi Hasil Simulasi Monte Carlo")
        st.write(interpret_montecarlo(simulated_premiums))

        # =========================
        # Ekspor hasil Survival
        # =========================
        st.subheader("⬇️ Ekspor Data Survival")

        # Buat DataFrame hasil survival function
        result_df = pd.DataFrame({
            "Horizon (hari)": surv_func.index,
            "Survival Probability": surv_func.values.flatten()
        })

        # Export CSV
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Unduh Data Survival (CSV)",
            data=csv,
            file_name="hasil_survival.csv",
            mime="text/csv"
        )

        # Export XLSX
        import io
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, index=False, sheet_name="SurvivalData")
        xlsx_data = output.getvalue()

        st.download_button(
            label="Unduh Data Survival (XLSX)",
            data=xlsx_data,
            file_name="hasil_survival.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )



        montecarlo_df = pd.DataFrame({
            "Simulasi ke-": np.arange(1, len(simulated_premiums)+1),
            "Premi (Rp)": simulated_premiums
        })
        csv2 = montecarlo_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Unduh Hasil Monte Carlo (CSV)",
            data=csv2,
            file_name="hasil_montecarlo.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Terjadi error: {e}")



