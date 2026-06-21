import streamlit as st
import pandas as pd
import joblib

# ==========================
# Load Model
# ==========================
cluster_scaler = joblib.load("cluster_scaler.pkl")
classification_scaler = joblib.load("classification_scaler.pkl")
encoder_gender = joblib.load("gender_encoder.pkl")
encoder_location = joblib.load("location_encoder.pkl")
kmeans_model = joblib.load("kmeans_model.pkl")
classifier_model = joblib.load("logistic_regression.pkl")

stress_mapping = {
    0: "High",
    1: "Low",
    2: "Medium"
}

# ==========================
# Session State Initialization
# ==========================
defaults = {
    "step": 1,
    "age": 18,
    "gender": "Male",
    "location_type": "Urban",
    "daily_screen_time_hours": 0,
    "social_media_hours": 0,
    "phone_usage_hours": 0,
    "laptop_usage_hours": 0,
    "tablet_usage_hours": 0,
    "tv_usage_hours": 0,
    "work_related_hours": 0,
    "entertainment_hours": 0,
    "gaming_hours": 0,
    "sleep_duration_hours": 8,
    "sleep_quality": 5,
    "physical_activity_hours_per_week": 0,
    "mindfulness_minutes_per_day": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ==========================
# Header
# ==========================
st.title("📱 Tech Use & Stress Prediction")

step_titles = [
    "Demografi",
    "Penggunaan Teknologi",
    "Gaya Hidup & Fisik"
]

total_steps = len(step_titles)

st.progress(st.session_state.step / total_steps)
st.subheader(
    f"Langkah {st.session_state.step}/{total_steps}: {step_titles[st.session_state.step-1]}"
)

# ==========================
# STEP 1
# ==========================
if st.session_state.step == 1:

    st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        key="age"
    )

    st.selectbox(
        "Gender",
        ["Male", "Female", "Other"],
        key="gender"
    )

    st.selectbox(
        "Location Type",
        ["Urban", "Suburban", "Rural"],
        key="location_type"
    )

# ==========================
# STEP 2
# ==========================
elif st.session_state.step == 2:

    st.number_input(
        "Daily Screen Time (Hours)",
        min_value=0,
        key="daily_screen_time_hours"
    )

    st.number_input(
        "Social Media (Hours)",
        min_value=0,
        key="social_media_hours"
    )

    st.number_input(
        "Phone Usage (Hours)",
        min_value=0,
        key="phone_usage_hours"
    )

    st.number_input(
        "Laptop Usage (Hours)",
        min_value=0,
        key="laptop_usage_hours"
    )

    st.number_input(
        "Tablet Usage (Hours)",
        min_value=0,
        key="tablet_usage_hours"
    )

    st.number_input(
        "TV Usage (Hours)",
        min_value=0,
        key="tv_usage_hours"
    )

    st.number_input(
        "Work Related (Hours)",
        min_value=0,
        key="work_related_hours"
    )

    st.number_input(
        "Entertainment (Hours)",
        min_value=0,
        key="entertainment_hours"
    )

    st.number_input(
        "Gaming (Hours)",
        min_value=0,
        key="gaming_hours"
    )

# ==========================
# STEP 3
# ==========================
elif st.session_state.step == 3:

    st.number_input(
        "Sleep Duration (Hours)",
        min_value=0,
        key="sleep_duration_hours"
    )

    st.slider(
        "Sleep Quality",
        1,
        10,
        key="sleep_quality"
    )

    st.number_input(
        "Physical Activity (Hours/Week)",
        min_value=0,
        key="physical_activity_hours_per_week"
    )

    st.number_input(
        "Mindfulness (Minutes/Day)",
        min_value=0,
        step=5,
        key="mindfulness_minutes_per_day"
    )

# ==========================
# Navigation
# ==========================
st.divider()

col1, col2, col3 = st.columns([1,1,1])

with col1:

    if st.session_state.step > 1:

        if st.button("Previous", use_container_width=True):
            st.session_state.step -= 1
            st.rerun()

with col3:

    if st.session_state.step < total_steps:

        if st.button("Next", use_container_width=True):
            st.session_state.step += 1
            st.rerun()

submitted = False

if st.session_state.step == total_steps:
    submitted = st.button(
        "🔍 Prediksi",
        type="primary",
        use_container_width=True
    )

# ==========================
# Prediction
# ==========================
if submitted:

    gender_encoded = encoder_gender.transform(
        [st.session_state.get("gender", "Male")]
    )[0]

    location_encoded = encoder_location.transform(
        [st.session_state.get("location_type", "Urban")]
    )[0]

    input_data = pd.DataFrame([[
        st.session_state.age,
        gender_encoded,
        location_encoded,
        st.session_state.daily_screen_time_hours,
        st.session_state.phone_usage_hours,
        st.session_state.laptop_usage_hours,
        st.session_state.tablet_usage_hours,
        st.session_state.tv_usage_hours,
        st.session_state.social_media_hours,
        st.session_state.work_related_hours,
        st.session_state.entertainment_hours,
        st.session_state.gaming_hours,
        st.session_state.sleep_duration_hours,
        st.session_state.sleep_quality,
        st.session_state.physical_activity_hours_per_week,
        st.session_state.mindfulness_minutes_per_day
    ]],
    columns=[
        "age",
        "gender",
        "location_type",
        "daily_screen_time_hours",
        "phone_usage_hours",
        "laptop_usage_hours",
        "tablet_usage_hours",
        "tv_usage_hours",
        "social_media_hours",
        "work_related_hours",
        "entertainment_hours",
        "gaming_hours",
        "sleep_duration_hours",
        "sleep_quality",
        "physical_activity_hours_per_week",
        "mindfulness_minutes_per_day"
    ])

    # Clustering
    cluster_scaled = cluster_scaler.transform(input_data)
    cluster = kmeans_model.predict(cluster_scaled)[0]

    # Classification
    classification_data = input_data.copy()
    classification_data["cluster"] = cluster

    class_scaled = classification_scaler.transform(classification_data)

    prediction = classifier_model.predict(class_scaled)[0]

    stress_level = stress_mapping[prediction]

    # ==========================
    # Result
    # ==========================
    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if stress_level == "High":
        st.error(f"Prediksi Tingkat Stres : **{stress_level}**")

    elif stress_level == "Medium":
        st.warning(f"Prediksi Tingkat Stres : **{stress_level}**")

    else:
        st.success(f"Prediksi Tingkat Stres : **{stress_level}**")

    st.write(f"Cluster Pengguna : **{cluster}**")

    st.divider()
    st.subheader("💡 Rekomendasi")

    rekomendasi = []

    if stress_level == "High":
        rekomendasi.append(
            "⚠ Kelola stres dengan latihan pernapasan atau meditasi 10–15 menit setiap hari."
        )
        rekomendasi.append(
            "🛌 Tingkatkan kualitas tidur dan hindari penggunaan gadget sebelum tidur."
        )

    elif stress_level == "Medium":
        rekomendasi.append(
            "⚖ Pertahankan keseimbangan antara aktivitas digital dan aktivitas fisik."
        )

    else:
        rekomendasi.append(
            "✅ Pertahankan gaya hidup sehat yang sudah Anda jalankan."
        )

    if st.session_state.daily_screen_time_hours > 8:
        rekomendasi.append(
            "📱 Screen time cukup tinggi. Cobalah metode 20-20-20 atau istirahat setiap 60 menit."
        )

    if st.session_state.physical_activity_hours_per_week < 2:
        rekomendasi.append(
            "🏃 Tingkatkan aktivitas fisik minimal 150 menit per minggu."
        )

    if st.session_state.sleep_duration_hours < 6:
        rekomendasi.append(
            "😴 Durasi tidur masih kurang. Disarankan tidur 7–9 jam setiap malam."
        )

    for i, item in enumerate(rekomendasi, start=1):
        st.write(f"{i}. {item}")