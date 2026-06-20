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
        age = st.number_input("Age", 10, 100)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        daily_screen_time_hours = st.number_input("Daily Screen Time", 0.0)
        phone_usage_hours = st.number_input("Phone Usage", 0.0)
        laptop_usage_hours = st.number_input("Laptop Usage", 0.0)
        tablet_usage_hours = st.number_input("Tablet Usage", 0.0)
        tv_usage_hours = st.number_input("TV Usage", 0.0)
        social_media_hours = st.number_input("Social Media", 0.0)
        
    with col2:
        work_related_hours = st.number_input("Work Related Hours", 0.0)
        entertainment_hours = st.number_input("Entertainment Hours", 0.0)
        gaming_hours = st.number_input("Gaming Hours", 0.0)
        sleep_duration_hours = st.number_input("Sleep Duration", 0.0)
        sleep_quality = st.slider("Sleep Quality", 1, 10)
        mood_rating = st.slider("Mood Rating", 1, 10)
        stress_level = st.slider("Stress Level", 1, 10)
        physical_activity_hours_per_week = st.number_input("Physical Activity", 0.0)
        
    with col3:
        location_type = st.selectbox("Location Type", ["Urban", "Suburban", "Rural"])
        mental_health_score = st.slider("Mental Health Score", 1, 100)
        eats_healthy = st.selectbox("Eats Healthy", ["Yes", "No"])
        caffeine_intake_mg_per_day = st.number_input("Caffeine Intake (mg)", 0.0)
        weekly_anxiety_score = st.number_input("Weekly Anxiety Score", 0.0)
        weekly_depression_score = st.number_input("Weekly Depression Score", 0.0)
        mindfulness_minutes_per_day = st.number_input("Mindfulness Minutes", 0.0)

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