import streamlit as st
import pandas as pd
import joblib

cluster_scaler = joblib.load('cluster_scaler.pkl')
classification_scaler = joblib.load('classification_scaler.pkl')
encoder_gender = joblib.load('gender_encoder.pkl')
encoder_location = joblib.load('location_encoder.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
classifier_model = joblib.load('logistic_regression.pkl')

stress_mapping = {0: "High", 1: "Low", 2: "Medium"}

st.title("Tech Use & Stress Prediction")

with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(["Demografi", "Penggunaan Teknologi", "Gaya Hidup & Fisik"])
    
    with tab1:
        age = st.number_input("Age", min_value=10, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        location_type = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])
        
    with tab2:
        daily_screen_time_hours = st.number_input("Daily Screen Time (Jam)", min_value=0, step=1)
        social_media_hours = st.number_input("Social Media (Jam)", min_value=0, step=1)
        phone_usage_hours = st.number_input("Phone Usage (Jam)", min_value=0, step=1)
        laptop_usage_hours = st.number_input("Laptop Usage (Jam)", min_value=0, step=1)
        tablet_usage_hours = st.number_input("Tablet Usage (Jam)", min_value=0, step=1)
        tv_usage_hours = st.number_input("TV Usage (Jam)", min_value=0, step=1)
        work_related_hours = st.number_input("Work Related (Jam)", min_value=0, step=1)
        entertainment_hours = st.number_input("Entertainment (Jam)", min_value=0, step=1)
        gaming_hours = st.number_input("Gaming (Jam)", min_value=0, step=1)
        
    with tab3:
        sleep_duration_hours = st.number_input("Sleep Duration (Jam)", min_value=0, step=1)
        sleep_quality = st.slider("Sleep Quality", 1, 10)
        physical_activity_hours_per_week = st.number_input("Physical Activity (Jam/Minggu)", min_value=0, step=1)
        mindfulness_minutes_per_day = st.number_input("Mindfulness (Menit)", min_value=0, step=5)

    submitted = st.form_submit_button("Prediksi")

if submitted:
    gender_encoded = encoder_gender.transform([gender])[0]
    location_encoded = encoder_location.transform([location_type])[0]
    
    input_data = pd.DataFrame([[
        age, gender_encoded, location_encoded, daily_screen_time_hours, 
        phone_usage_hours, laptop_usage_hours, tablet_usage_hours, tv_usage_hours, 
        social_media_hours, work_related_hours, entertainment_hours, gaming_hours, 
        sleep_duration_hours, sleep_quality, physical_activity_hours_per_week, 
        mindfulness_minutes_per_day
    ]], columns=[
        'age', 'gender', 'location_type', 'daily_screen_time_hours', 
        'phone_usage_hours', 'laptop_usage_hours', 'tablet_usage_hours', 
        'tv_usage_hours', 'social_media_hours', 'work_related_hours', 
        'entertainment_hours', 'gaming_hours', 'sleep_duration_hours', 
        'sleep_quality', 'physical_activity_hours_per_week', 
        'mindfulness_minutes_per_day'
    ])
    
    input_cluster_scaled = cluster_scaler.transform(input_data)
    cluster_result = kmeans_model.predict(input_cluster_scaled)[0]
    
    classification_data = input_data.copy()
    classification_data['cluster'] = cluster_result
    
    input_class_scaled = classification_scaler.transform(classification_data)
    classification_result = classifier_model.predict(input_class_scaled)[0]
    
    stress_level = stress_mapping[classification_result]
    
    st.subheader("Hasil Analisis")
    if stress_level == "High":
        st.error(f"**Prediksi Tingkat Stres:** {stress_level}")
    elif stress_level == "Medium":
        st.warning(f"**Prediksi Tingkat Stres:** {stress_level}")
    else:
        st.success(f"**Prediksi Tingkat Stres:** {stress_level}")

    st.subheader("Rekomendasi Gaya Hidup")
    rekomendasi = []
    
    if stress_level == "High":
        rekomendasi.append("⚠️ **Kelola Stres:** Cobalah teknik pernapasan dalam atau meditasi 10 menit setiap hari.")
        rekomendasi.append("🛌 **Perbaiki Tidur:** Prioritaskan kualitas tidur di atas hiburan malam.")
    elif stress_level == "Medium":
        rekomendasi.append("⚖️ **Jaga Keseimbangan:** Pertahankan aktivitas fisik agar stres tidak meningkat.")
    else:
        rekomendasi.append("✅ **Pertahankan:** Pola hidup Anda saat ini sudah sangat baik untuk menjaga kesehatan mental.")
        
    if daily_screen_time_hours > 8:
        rekomendasi.append("📱 **Digital Detox:** Screen time Anda cukup tinggi, cobalah beristirahat setiap 60 menit bekerja.")
        
    if physical_activity_hours_per_week < 2:
        rekomendasi.append("🏃 **Bergerak Lebih:** Cobalah berjalan kaki atau olahraga ringan minimal 150 menit per minggu.")
        
    if sleep_duration_hours < 6:
        rekomendasi.append("😴 **Waktu Istirahat:** Usahakan tidur 7-8 jam per hari untuk pemulihan otak yang optimal.")

    for i, rec in enumerate(rekomendasi, 1):
        st.write(f"{i}. {rec}")