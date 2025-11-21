import streamlit as st
import joblib
import re
import numpy as np # Ditambahkan untuk operasi array probabilitas

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Sentimen Film",
    page_icon="🎬",
    layout="centered"
)

# Load model & tools
@st.cache_resource
def load_model_objects():
    try:
        # PENTING: Pastikan file-file ini ada di direktori yang sama
        model_bnb = joblib.load("model_bernoulli_nb.pkl")
        model_svm = joblib.load("model_linear_svm.pkl")
        model_ensemble = joblib.load("model_ensemble_voting.pkl")
        vectorizer = joblib.load("vectorizer_tfidf.pkl")
        tools = joblib.load("preprocessing_tools.pkl")
        return model_bnb, model_svm, model_ensemble, vectorizer, tools
    except Exception as e:
        # Menangkap error saat load file
        st.error(f"Gagal memuat file model. Pastikan semua file .pkl tersedia. Error: {e}")
        return None, None, None, None, None

model_bnb, model_svm, model_ensemble, vectorizer, tools = load_model_objects()

# Preprocessing teks
def preprocess_text(text, stopword_remover, stemmer):
    # 1. Cleaning: hapus non-huruf, ubah ke lowercase
    text = re.sub('[^A-Za-z]+', ' ', text).lower().strip()
    # 2. Hapus multiple spasi
    text = re.sub('\s+', ' ', text)
    # 3. Stopword Removal
    text = stopword_remover.remove(text)
    # 4. Stemming (asumsi stemmer bekerja pada kalimat)
    # Catatan: Jika stemmer hanya menerima kata, loop harus ditambahkan di sini.
    # Namun, mengikuti struktur kode Anda:
    text = stemmer.stem(text) 
    return text

# Confidence label
def get_confidence_badge(prob):
    # Asumsi prob adalah nilai 0-100
    if prob >= 80:
        return "🟢 Tinggi", "success"
    elif prob >= 60:
        return "🟡 Sedang", "warning"
    else:
        return "🔴 Rendah", "error"

# UI Utama
st.title("🎬 Analisis Sentimen Film")
st.markdown("### Ensemble Model (BernoulliNB + SVM)")

models_loaded = all([model_bnb, model_svm, model_ensemble, vectorizer, tools])

if not models_loaded:
    st.error("⚠️ File model tidak ditemukan atau gagal dimuat. Cek error di atas.")
else:
    st.subheader("✍️ Masukkan Ulasan Film")

    example_texts = [
        "Filmnya bagus banget, alurnya tidak ketebak!",
        "Film jelek, buang waktu saja",
        "Keren, aktingnya mantap sekali",
        "Goblok banget filmnya tidak bermutu",
        "Biasa aja sih, tidak terlalu bagus",
        "Luar biasa, sangat recommended!"
    ]

    selected_example = st.selectbox(
        "Pilih contoh ulasan:",
        ["-- Ketik manual --"] + example_texts
    )

    default_text = "" if selected_example == "-- Ketik manual --" else selected_example

    input_text = st.text_area("Masukkan ulasan film:", value=default_text, height=100)

    col1, col2, col3 = st.columns(3)
    with col1:
        predict_btn = st.button("🔍 Analisis", type="primary")
    with col2:
        show_comparison = st.checkbox("Bandingkan model", value=True)
    with col3:
        show_details = st.checkbox("Detail preprocessing", value=False)

    if predict_btn:
        if input_text.strip() == "":
            st.warning("⚠️ Masukkan teks terlebih dahulu.")
        else:
            with st.spinner('Menganalisis...'):
                try:
                    # Ambil tools
                    stopword_remover = tools['stopword']
                    stemmer = tools['stemmer']
                    
                    # Preprocessing
                    processed = preprocess_text(input_text, stopword_remover, stemmer)
                    vec = vectorizer.transform([processed])

                    # Prediksi Label
                    pred_bnb = model_bnb.predict(vec)[0]
                    pred_svm = model_svm.predict(vec)[0]
                    pred_ensemble = model_ensemble.predict(vec)[0]

                    # Probabilitas (asumsi output probabilitas berurutan: [negatif, positif])
                    prob_bnb = model_bnb.predict_proba(vec)[0]
                    prob_svm = model_svm.predict_proba(vec)[0]
                    prob_ensemble = model_ensemble.predict_proba(vec)[0]
                    
                    # Tentukan sentimen final
                    # Asumsi: prob_ensemble[0] = Negatif, prob_ensemble[1] = Positif
                    if prob_ensemble[0] > prob_ensemble[1]:
                        final_pred = "negative"
                        max_prob = prob_ensemble[0] * 100
                    else:
                        final_pred = "positive"
                        max_prob = prob_ensemble[1] * 100
                    
                    # === HASIL ENSEMBLE ===
                    st.subheader("🎯 Hasil Analisis (Ensemble)")
                    
                    conf_text, conf_type = get_confidence_badge(max_prob)

                    if final_pred == "positive":
                        st.success("### ✅ Sentimen: POSITIF")
                    else:
                        st.error("### ❌ Sentimen: NEGATIF")

                    st.info(f"**Tingkat Keyakinan:** {conf_text} ({max_prob:.1f}%)")

                    # Probabilitas
                    st.write("**📊 Probabilitas:**")
                    col_neg, col_pos = st.columns(2)
                    with col_neg:
                        # Baris 116 dari error sebelumnya
                        st.metric("Negatif", f"{prob_ensemble[0]*100:.1f}%")
                    with col_pos:
                        st.metric("Positif", f"{prob_ensemble[1]*100:.1f}%") # <--- Baris ini yang hilang!
                    
                    # === PERBANDINGAN MODEL ===
                    if show_comparison:
                        st.markdown("---")
                        st.subheader("Perbandingan Prediksi Model Individu")
                        
                        col_bnb, col_svm = st.columns(2)
                        
                        with col_bnb:
                            st.markdown("**Bernoulli Naive Bayes**")
                            prob_bnb_max = np.max(prob_bnb) * 100
                            bnb_text, _ = get_confidence_badge(prob_bnb_max)
                            
                            if pred_bnb == "positive":
                                st.success(f"**{pred_bnb.upper()}** ({prob_bnb_max:.1f}%)")
                            else:
                                st.warning(f"**{pred_bnb.upper()}** ({prob_bnb_max:.1f}%)")
                            st.caption(f"Negatif: {prob_bnb[0]*100:.1f}%, Positif: {prob_bnb[1]*100:.1f}%")

                        with col_svm:
                            st.markdown("**Linear SVM**")
                            # Untuk SVM yang tidak selalu punya predict_proba, 
                            # kita asumsikan prob_svm adalah output yang sesuai dari VotingClassifier
                            prob_svm_max = np.max(prob_svm) * 100
                            svm_text, _ = get_confidence_badge(prob_svm_max)
                            
                            if pred_svm == "positive":
                                st.success(f"**{pred_svm.upper()}** ({prob_svm_max:.1f}%)")
                            else:
                                st.warning(f"**{pred_svm.upper()}** ({prob_svm_max:.1f}%)")
                            st.caption(f"Negatif: {prob_svm[0]*100:.1f}%, Positif: {prob_svm[1]*100:.1f}%")

                    # === DETAIL PREPROCESSING ===
                    if show_details:
                        st.markdown("---")
                        st.subheader("Detail Preprocessing")
                        st.markdown(f"**Teks Asli:** `{input_text}`")
                        st.markdown(f"**Teks Setelah Preprocessing:** `{processed}`")
                        st.code(f"Teks divetorisasi menjadi: {vec}")


                except Exception as e:
                    # BLOCK 'except' WAJIB UNTUK MENUTUP 'try'
                    st.error(f"❌ Terjadi kesalahan saat memprediksi atau mentransformasi data: {e}")
                    st.exception(e)
