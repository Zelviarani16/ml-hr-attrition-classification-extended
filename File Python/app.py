import os

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Attrition Karyawan", layout="wide")

script_dir = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(script_dir, ".."))
MODEL_PATH = os.path.join(ROOT_DIR, "Model", "model_terbaik.joblib")
HASIL_PATH = os.path.join(ROOT_DIR, "hasil_perbandingan_semua_model.csv")

LABEL = {
    "Age": "Usia (tahun)",
    "Gender": "Jenis kelamin",
    "MaritalStatus": "Status pernikahan",
    "Education": "Tingkat pendidikan",
    "EducationField": "Bidang pendidikan",
    "DistanceFromHome": "Jarak rumah ke kantor",
    "Department": "Departemen",
    "JobRole": "Jabatan",
    "JobLevel": "Level jabatan",
    "BusinessTravel": "Perjalanan dinas",
    "OverTime": "Lembur",
    "JobInvolvement": "Keterlibatan dalam pekerjaan",
    "PerformanceRating": "Penilaian kinerja",
    "TrainingTimesLastYear": "Jumlah pelatihan tahun lalu",
    "MonthlyIncome": "Pendapatan bulanan",
    "DailyRate": "Daily rate",
    "HourlyRate": "Hourly rate",
    "MonthlyRate": "Monthly rate",
    "PercentSalaryHike": "Kenaikan gaji terakhir (%)",
    "StockOptionLevel": "Level opsi saham",
    "EnvironmentSatisfaction": "Kepuasan terhadap lingkungan kerja",
    "JobSatisfaction": "Kepuasan terhadap pekerjaan",
    "RelationshipSatisfaction": "Kepuasan terhadap hubungan kerja",
    "WorkLifeBalance": "Keseimbangan kerja dan kehidupan",
    "NumCompaniesWorked": "Jumlah perusahaan sebelumnya",
    "TotalWorkingYears": "Total lama bekerja (tahun)",
    "YearsAtCompany": "Lama di perusahaan ini (tahun)",
    "YearsInCurrentRole": "Lama di jabatan sekarang (tahun)",
    "YearsSinceLastPromotion": "Tahun sejak promosi terakhir",
    "YearsWithCurrManager": "Lama bersama manajer sekarang (tahun)",
}

SKALA_4 = {1: "1 - Rendah", 2: "2 - Sedang", 3: "3 - Tinggi", 4: "4 - Sangat tinggi"}
OPSI_LABEL = {
    "Education": {1: "1 - Di bawah kuliah", 2: "2 - Diploma", 3: "3 - Sarjana", 4: "4 - Magister", 5: "5 - Doktor"},
    "JobInvolvement": SKALA_4,
    "EnvironmentSatisfaction": SKALA_4,
    "JobSatisfaction": SKALA_4,
    "RelationshipSatisfaction": SKALA_4,
    "WorkLifeBalance": {1: "1 - Buruk", 2: "2 - Baik", 3: "3 - Lebih baik", 4: "4 - Terbaik"},
    "PerformanceRating": {3: "3 - Sangat baik", 4: "4 - Luar biasa"},
    "Gender": {"Male": "Laki-laki", "Female": "Perempuan"},
    "MaritalStatus": {"Single": "Lajang", "Married": "Menikah", "Divorced": "Cerai"},
    "OverTime": {"Yes": "Ya", "No": "Tidak"},
    "BusinessTravel": {"Non-Travel": "Tidak pernah", "Travel_Rarely": "Jarang", "Travel_Frequently": "Sering"},
}

KELOMPOK = {
    "Profil karyawan": ["Age", "Gender", "MaritalStatus", "Education", "EducationField", "DistanceFromHome"],
    "Pekerjaan": ["Department", "JobRole", "JobLevel", "BusinessTravel", "OverTime",
                  "JobInvolvement", "PerformanceRating", "TrainingTimesLastYear"],
    "Kompensasi": ["MonthlyIncome", "DailyRate", "HourlyRate", "MonthlyRate",
                   "PercentSalaryHike", "StockOptionLevel"],
    "Kepuasan dan keseimbangan": ["EnvironmentSatisfaction", "JobSatisfaction",
                                  "RelationshipSatisfaction", "WorkLifeBalance"],
    "Riwayat karier": ["NumCompaniesWorked", "TotalWorkingYears", "YearsAtCompany",
                       "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"],
}


@st.cache_resource
def muat_artefak():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.title("Prediksi Attrition Karyawan")
    st.error("Berkas model belum ditemukan. Jalankan dulu program `klasifikasi.py` "
             "agar folder `Model/` terbentuk, lalu buka kembali aplikasi ini.")
    st.stop()

art = muat_artefak()
model = art["model"]
metrik = art["metrik"]

with st.sidebar:
    st.header("Model yang dipakai")
    st.write(f"**{art['nama_model']}** dengan skenario **{art['skenario']}**")
    st.caption("SVM adalah metode dengan akurasi tertinggi dari tiga metode yang dibandingkan.")
    c1, c2 = st.columns(2)
    c1.metric("Akurasi", f"{metrik['Akurasi']:.2%}")
    c2.metric("F1-Score", f"{metrik['F1-Score']:.2%}")
    c1.metric("Presisi", f"{metrik['Presisi']:.2%}")
    c2.metric("Recall", f"{metrik['Recall']:.2%}")
    st.caption("Metrik dihitung pada data testing (20%) untuk kelas Yes (karyawan berhenti).")

st.title("Prediksi Attrition Karyawan")
st.write("Aplikasi ini memperkirakan apakah seorang karyawan **berisiko berhenti bekerja** "
         "berdasarkan dataset IBM HR Attrition.")

tab_prediksi, tab_banding, tab_tentang = st.tabs(["Prediksi", "Perbandingan model", "Tentang"])

with tab_prediksi:
    nilai = {}
    with st.form("form_prediksi"):
        st.caption("Isi data karyawan di bawah ini. Nilai awal adalah nilai tipikal dari data training.")
        for nama_kelompok, kolom_kelompok in KELOMPOK.items():
            st.subheader(nama_kelompok)
            kolom_ui = st.columns(3)
            for i, kolom in enumerate(kolom_kelompok):
                wadah = kolom_ui[i % 3]
                label = LABEL[kolom]
                if kolom in art["categorical_columns"]:
                    opsi = art["opsi_kategorik"][kolom]
                    peta = OPSI_LABEL.get(kolom, {})
                    awal = opsi.index(art["default_kategorik"][kolom])
                    nilai[kolom] = wadah.selectbox(label, opsi, index=awal,
                                                   format_func=lambda x, p=peta: p.get(x, x), key=kolom)
                elif kolom in art["nilai_diskret"] and kolom in OPSI_LABEL:
                    opsi = [int(x) for x in art["nilai_diskret"][kolom]]
                    awal = min(opsi, key=lambda x: abs(x - art["default_numerik"][kolom]))
                    peta = OPSI_LABEL[kolom]
                    nilai[kolom] = wadah.selectbox(label, opsi, index=opsi.index(awal),
                                                   format_func=lambda x, p=peta: p.get(x, str(x)), key=kolom)
                else:
                    batas_bawah, batas_atas = art["rentang_numerik"][kolom]
                    nilai[kolom] = wadah.number_input(
                        label, min_value=int(batas_bawah), max_value=int(batas_atas),
                        value=int(round(art["default_numerik"][kolom])), step=1, key=kolom)
        kirim = st.form_submit_button("Prediksi", type="primary")

    if kirim:
        galat = []
        if nilai["YearsAtCompany"] > nilai["TotalWorkingYears"]:
            galat.append("Lama di perusahaan ini tidak boleh melebihi total lama bekerja.")
        for kolom in ["YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]:
            if nilai[kolom] > nilai["YearsAtCompany"]:
                galat.append(f"{LABEL[kolom]} tidak boleh melebihi lama di perusahaan ini.")
        if galat:
            for pesan in galat:
                st.error(pesan)
        else:
            data = pd.DataFrame([nilai])[art["kolom_fitur"]]
            for kolom, (bawah, atas) in art["bounds"].items():
                data[kolom] = data[kolom].astype(float).clip(bawah, atas)
            data_transform = pd.DataFrame(
                art["preprocessor"].transform(data),
                columns=art["preprocessor"].get_feature_names_out())
            hasil = int(model.predict(data_transform)[0])
            st.divider()
            st.subheader("Hasil prediksi")
            if hasil == 1:
                st.error("Karyawan ini **berisiko berhenti bekerja** (Attrition = Yes).")
            else:
                st.success("Karyawan ini **cenderung bertahan** (Attrition = No).")
            if hasattr(model, "predict_proba"):
                peluang = model.predict_proba(data_transform)[0]
                kelas = list(model.classes_)
                p_yes = float(peluang[kelas.index(1)])
                teks_peluang = "< 0,1%" if p_yes < 0.001 else f"{p_yes:.1%}"
                st.write(f"Perkiraan peluang berhenti: **{teks_peluang}**")
                st.progress(min(max(p_yes, 0.0), 1.0))
                st.caption("Peluang adalah perkiraan model dan tidak dikalibrasi, karena model dilatih "
                           "pada data hasil resampling.")
            st.warning(
                f"Model ini baru mendeteksi sekitar {metrik['Recall']:.0%} karyawan yang benar-benar berhenti "
                f"(recall) dan sebagian peringatannya salah (presisi {metrik['Presisi']:.0%}). "
                "Gunakan hasil ini sebagai alat bantu penyaringan awal, bukan sebagai keputusan akhir.")
            with st.expander("Lihat data yang dimasukkan"):
                st.dataframe(pd.DataFrame({"Fitur": [LABEL[k] for k in nilai],
                                           "Nilai": [str(v) for v in nilai.values()]}),
                             hide_index=True)

with tab_banding:
    st.subheader("Perbandingan Naïve Bayes, SVM, dan Decision Tree")
    if os.path.exists(HASIL_PATH):
        tabel = pd.read_csv(HASIL_PATH)
        st.dataframe(
            tabel.style.format({"Akurasi": "{:.4f}", "Presisi": "{:.4f}", "Recall": "{:.4f}", "F1-Score": "{:.4f}"}),
            hide_index=True)
        rata = tabel.groupby("Model")[["Akurasi", "Presisi", "Recall", "F1-Score"]].mean().round(4)
        st.caption("Rata-rata tiga skenario (baseline, ROS, SMOTE) per metode")
        st.dataframe(rata)
    gambar = os.path.join(ROOT_DIR, "05_perbandingan_metrik_semua_model.png")
    if os.path.exists(gambar):
        st.image(gambar, caption="Perbandingan metrik evaluasi pada data testing")
    for nama in ["naive_bayes", "svm", "decision_tree"]:
        gambar_cm = os.path.join(ROOT_DIR, f"04_confusion_matrix_{nama}.png")
        if os.path.exists(gambar_cm):
            st.image(gambar_cm, caption=f"Confusion matrix {nama.replace('_', ' ').title()}")

with tab_tentang:
    st.subheader("Tentang aplikasi")
    st.markdown(
        """
        **Alur data sebelum prediksi** (sama dengan saat model dilatih):
        1. Nilai yang dimasukkan dibatasi (capping IQR) pada kolom yang rentan outlier.
        2. Fitur numerik distandardisasi dengan Z-Score dan fitur kategorikal diubah dengan one-hot encoding.
        3. Model SVM memprediksi kelas: `Yes` (berisiko berhenti) atau `No` (cenderung bertahan).

        **Model:** dipilih dari metode dengan akurasi tertinggi, lalu skenario resampling dengan F1-Score terbaik
        pada metode tersebut. Data training diseimbangkan dengan resampling agar kelas minoritas (karyawan berhenti)
        lebih mudah dikenali.

        **Keterbatasan:** data testing hanya memuat 47 karyawan kelas Yes, dan model masih melewatkan sebagian
        karyawan yang benar-benar berhenti. Aplikasi ini dibuat untuk keperluan tugas kuliah.
        """
    )
