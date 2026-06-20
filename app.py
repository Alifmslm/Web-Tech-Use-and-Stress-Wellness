import streamlit as st
import pandas as pd
import joblib

scaler = joblib.load('cluster_scaler.pkl')
encoder_gender = joblib.load('gender_encoder.pkl')
encoder_location = joblib.load('location_encoder.pkl')
kmeans_model = joblib.load('kmeans_model.pkl')
classifier_model = joblib.load('logistic_regression.pkl')

st.title("Tech Use & Wellness Prediction")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, step=1, help="Usia pengguna saat ini.")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], help="Identitas gender pengguna.")
        daily_screen_time_hours = st.number_input("Daily Screen Time (Jam)", min_value=0, step=1, help="Total waktu menatap layar perangkat dalam sehari.")
        phone_usage_hours = st.number_input("Phone Usage (Jam)", min_value=0, step=1, help="Total waktu penggunaan ponsel dalam sehari.")
        laptop_usage_hours = st.number_input("Laptop Usage (Jam)", min_value=0, step=1, help="Total waktu penggunaan laptop dalam sehari.")
        tablet_usage_hours = st.number_input("Tablet Usage (Jam)", min_value=0, step=1, help="Total waktu penggunaan tablet dalam sehari.")
        tv_usage_hours = st.number_input("TV Usage (Jam)", min_value=0, step=1, help="Total waktu menonton TV dalam sehari.")
        social_media_hours = st.number_input("Social Media (Jam)", min_value=0, step=1, help="Total waktu bermain media sosial dalam sehari.")
        
    with col2:
        work_related_hours = st.number_input("Work Related (Jam)", min_value=0, step=1, help="Waktu layar yang digunakan khusus untuk bekerja/belajar.")
        entertainment_hours = st.number_input("Entertainment (Jam)", min_value=0, step=1, help="Waktu layar untuk menonton film, video, atau hiburan lainnya.")
        gaming_hours = st.number_input("Gaming (Jam)", min_value=0, step=1, help="Waktu layar untuk bermain game.")
        sleep_duration_hours = st.number_input("Sleep Duration (Jam)", min_value=0, step=1, help="Durasi rata-rata waktu tidur harian.")
        sleep_quality = st.slider("Sleep Quality", 1, 10, help="Penilaian subjektif kualitas tidur (1 = Sangat Buruk, 10 = Sangat Baik).")
        mood_rating = st.slider("Mood Rating", 1, 10, help="Penilaian rata-rata suasana hati (1 = Sangat Buruk, 10 = Sangat Baik).")
        stress_level = st.slider("Stress Level", 1, 10, help="Tingkat stres yang dirasakan (1 = Sangat Rendah, 10 = Sangat Tinggi).")
        physical_activity_hours_per_week = st.number_input("Physical Activity (Jam/Minggu)", min_value=0, step=1, help="Total waktu aktivitas fisik atau olahraga dalam seminggu.")
        
    with col3:
        location_type = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"], help="Jenis lingkungan tempat tinggal pengguna.")
        mental_health_score = st.slider("Mental Health Score", 1, 100, help="Skor kesehatan mental keseluruhan berdasarkan asesmen (1-100).")
        eats_healthy = st.selectbox("Eats Healthy", ["Yes", "No"], help="Apakah pengguna menerapkan pola makan sehat?")
        caffeine_intake_mg_per_day = st.number_input("Caffeine Intake (mg)", min_value=0, step=10, help="Estimasi asupan kafein harian dalam miligram.")
        weekly_anxiety_score = st.number_input("Weekly Anxiety Score", min_value=0, step=1, help="Skor tingkat kecemasan mingguan.")
        weekly_depression_score = st.number_input("Weekly Depression Score", min_value=0, step=1, help="Skor tingkat depresi mingguan.")
        mindfulness_minutes_per_day = st.number_input("Mindfulness (Menit)", min_value=0, step=5, help="Durasi aktivitas relaksasi atau meditasi per hari.")

    submitted = st.form_submit_button("Prediksi")

if submitted:
    gender_encoded = encoder_gender.transform([gender])[0]
    location_encoded = encoder_location.transform([location_type])[0]
    eats_healthy_encoded = 1 if eats_healthy == "Yes" else 0
    
    input_data = pd.DataFrame([[
        age, gender_encoded, daily_screen_time_hours, phone_usage_hours, laptop_usage_hours,
        tablet_usage_hours, tv_usage_hours, social_media_hours, work_related_hours,
        entertainment_hours, gaming_hours, sleep_duration_hours, sleep_quality,
        mood_rating, stress_level, physical_activity_hours_per_week, location_encoded,
        mental_health_score, eats_healthy_encoded, caffeine_intake_mg_per_day,
        weekly_anxiety_score, weekly_depression_score, mindfulness_minutes_per_day
    ]], columns=[
        'age', 'gender', 'daily_screen_time_hours', 'phone_usage_hours',
        'laptop_usage_hours', 'tablet_usage_hours', 'tv_usage_hours',
        'social_media_hours', 'work_related_hours', 'entertainment_hours',
        'gaming_hours', 'sleep_duration_hours', 'sleep_quality', 'mood_rating',
        'stress_level', 'physical_activity_hours_per_week', 'location_type',
        'mental_health_score', 'eats_healthy', 'caffeine_intake_mg_per_day',
        'weekly_anxiety_score', 'weekly_depression_score', 'mindfulness_minutes_per_day'
    ])
    
    input_scaled = scaler.transform(input_data)
    
    cluster_result = kmeans_model.predict(input_scaled)[0]
    classification_result = classifier_model.predict(input_scaled)[0]
    
    st.subheader("Hasil Analisis")
    st.write(f"**Segmentasi Pengguna (Cluster K-Means):** {cluster_result}")
    
    if classification_result == 1:
        st.success("**Prediksi:** Pengguna cenderung AKTIF menggunakan Wellness Apps.")
    else:
        st.warning("**Prediksi:** Pengguna cenderung TIDAK menggunakan Wellness Apps.")