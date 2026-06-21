import streamlit as st
import pandas as pd
import joblib

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

step_titles = [
    "Demografi",
    "Penggunaan Teknologi",
    "Gaya Hidup & Fisik"
]
total_steps = len(step_titles)

STEP_FIELDS = {
    1: ["age", "gender", "location_type"],
    2: [
        "daily_screen_time_hours", "social_media_hours", "phone_usage_hours",
        "laptop_usage_hours", "tablet_usage_hours", "tv_usage_hours",
        "work_related_hours", "entertainment_hours", "gaming_hours"
    ],
    3: [
        "sleep_duration_hours", "sleep_quality",
        "physical_activity_hours_per_week", "mindfulness_minutes_per_day"
    ],
}

FIELD_LABELS = {
    "age": "Age",
    "gender": "Gender",
    "location_type": "Location Type",
    "daily_screen_time_hours": "Daily Screen Time",
    "social_media_hours": "Social Media",
    "phone_usage_hours": "Phone Usage",
    "laptop_usage_hours": "Laptop Usage",
    "tablet_usage_hours": "Tablet Usage",
    "tv_usage_hours": "TV Usage",
    "work_related_hours": "Work Related",
    "entertainment_hours": "Entertainment",
    "gaming_hours": "Gaming",
    "sleep_duration_hours": "Sleep Duration",
    "sleep_quality": "Sleep Quality",
    "physical_activity_hours_per_week": "Physical Activity",
    "mindfulness_minutes_per_day": "Mindfulness",
}

if "step" not in st.session_state:
    st.session_state.step = 1

if "form_data" not in st.session_state:
    st.session_state.form_data = {}

if "predicted" not in st.session_state:
    st.session_state.predicted = False

def get_missing_fields(step: int):
    missing = []
    for key in STEP_FIELDS[step]:
        if st.session_state.get(key) is None:
            missing.append(FIELD_LABELS[key])
    return missing

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.title("📱 Tech Use & Stress Prediction")

if not st.session_state.predicted:
    st.progress(st.session_state.step / total_steps)
    st.subheader(
        f"Langkah {st.session_state.step}/{total_steps}: {step_titles[st.session_state.step - 1]}"
    )

    if st.session_state.step == 1:
        st.number_input(
            "Age", min_value=10, max_value=100,
            value=st.session_state.form_data.get("age", None),
            placeholder="Masukkan usia", key="age"
        )
        st.selectbox(
            "Gender", ["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(st.session_state.form_data["gender"]) if "gender" in st.session_state.form_data else None,
            placeholder="Pilih gender", key="gender"
        )
        st.selectbox(
            "Location Type", ["Urban", "Suburban", "Rural"],
            index=["Urban", "Suburban", "Rural"].index(st.session_state.form_data["location_type"]) if "location_type" in st.session_state.form_data else None,
            placeholder="Pilih lokasi", key="location_type"
        )

    elif st.session_state.step == 2:
        st.number_input("Daily Screen Time (Hours)", min_value=0, value=st.session_state.form_data.get("daily_screen_time_hours", None), placeholder="Masukkan jam", key="daily_screen_time_hours")
        st.number_input("Social Media (Hours)", min_value=0, value=st.session_state.form_data.get("social_media_hours", None), placeholder="Masukkan jam", key="social_media_hours")
        st.number_input("Phone Usage (Hours)", min_value=0, value=st.session_state.form_data.get("phone_usage_hours", None), placeholder="Masukkan jam", key="phone_usage_hours")
        st.number_input("Laptop Usage (Hours)", min_value=0, value=st.session_state.form_data.get("laptop_usage_hours", None), placeholder="Masukkan jam", key="laptop_usage_hours")
        st.number_input("Tablet Usage (Hours)", min_value=0, value=st.session_state.form_data.get("tablet_usage_hours", None), placeholder="Masukkan jam", key="tablet_usage_hours")
        st.number_input("TV Usage (Hours)", min_value=0, value=st.session_state.form_data.get("tv_usage_hours", None), placeholder="Masukkan jam", key="tv_usage_hours")
        st.number_input("Work Related (Hours)", min_value=0, value=st.session_state.form_data.get("work_related_hours", None), placeholder="Masukkan jam", key="work_related_hours")
        st.number_input("Entertainment (Hours)", min_value=0, value=st.session_state.form_data.get("entertainment_hours", None), placeholder="Masukkan jam", key="entertainment_hours")
        st.number_input("Gaming (Hours)", min_value=0, value=st.session_state.form_data.get("gaming_hours", None), placeholder="Masukkan jam", key="gaming_hours")

    elif st.session_state.step == 3:
        st.number_input("Sleep Duration (Hours)", min_value=0, value=st.session_state.form_data.get("sleep_duration_hours", None), placeholder="Masukkan jam", key="sleep_duration_hours")
        st.slider("Sleep Quality", 1, 10, value=st.session_state.form_data.get("sleep_quality", 1), key="sleep_quality")
        st.number_input("Physical Activity (Hours/Week)", min_value=0, value=st.session_state.form_data.get("physical_activity_hours_per_week", None), placeholder="Masukkan jam", key="physical_activity_hours_per_week")
        st.number_input("Mindfulness (Minutes/Day)", min_value=0, step=5, value=st.session_state.form_data.get("mindfulness_minutes_per_day", None), placeholder="Masukkan menit", key="mindfulness_minutes_per_day")

    st.divider()

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.session_state.step > 1:
            if st.button("Previous", use_container_width=True):
                for key in STEP_FIELDS[st.session_state.step]:
                    if st.session_state.get(key) is not None:
                        st.session_state.form_data[key] = st.session_state[key]
                st.session_state.step -= 1
                st.rerun()

    with col3:
        if st.session_state.step < total_steps:
            if st.button("Next", use_container_width=True):
                missing = get_missing_fields(st.session_state.step)
                if missing:
                    st.warning(f"⚠️ Mohon lengkapi field berikut sebelum melanjutkan: {', '.join(missing)}")
                else:
                    for key in STEP_FIELDS[st.session_state.step]:
                        st.session_state.form_data[key] = st.session_state[key]
                    st.session_state.step += 1
                    st.rerun()

    if st.session_state.step == total_steps:
        if st.button("🔍 Prediksi", type="primary", use_container_width=True):
            missing = get_missing_fields(total_steps)
            if missing:
                st.warning(f"⚠️ Mohon lengkapi field berikut sebelum memprediksi: {', '.join(missing)}")
            else:
                for key in STEP_FIELDS[total_steps]:
                    st.session_state.form_data[key] = st.session_state[key]
                st.session_state.predicted = True
                st.rerun()

if st.session_state.predicted:
    gender_encoded = encoder_gender.transform([st.session_state.form_data["gender"]])[0]
    location_encoded = encoder_location.transform([st.session_state.form_data["location_type"]])[0]

    input_data = pd.DataFrame([[
        st.session_state.form_data["age"],
        gender_encoded,
        location_encoded,
        st.session_state.form_data["daily_screen_time_hours"],
        st.session_state.form_data["phone_usage_hours"],
        st.session_state.form_data["laptop_usage_hours"],
        st.session_state.form_data["tablet_usage_hours"],
        st.session_state.form_data["tv_usage_hours"],
        st.session_state.form_data["social_media_hours"],
        st.session_state.form_data["work_related_hours"],
        st.session_state.form_data["entertainment_hours"],
        st.session_state.form_data["gaming_hours"],
        st.session_state.form_data["sleep_duration_hours"],
        st.session_state.form_data["sleep_quality"],
        st.session_state.form_data["physical_activity_hours_per_week"],
        st.session_state.form_data["mindfulness_minutes_per_day"]
    ]],
    columns=[
        "age", "gender", "location_type", "daily_screen_time_hours",
        "phone_usage_hours", "laptop_usage_hours", "tablet_usage_hours",
        "tv_usage_hours", "social_media_hours", "work_related_hours",
        "entertainment_hours", "gaming_hours", "sleep_duration_hours",
        "sleep_quality", "physical_activity_hours_per_week",
        "mindfulness_minutes_per_day"
    ])

    cluster_scaled = cluster_scaler.transform(input_data)
    cluster = kmeans_model.predict(cluster_scaled)[0]

    classification_data = input_data.copy()
    classification_data["cluster"] = cluster

    class_scaled = classification_scaler.transform(classification_data)

    prediction = classifier_model.predict(class_scaled)[0]
    stress_level = stress_mapping[prediction]

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
        rekomendasi.append("⚠ Kelola stres dengan latihan pernapasan atau meditasi 10–15 menit setiap hari.")
        rekomendasi.append("🛌 Tingkatkan kualitas tidur dan hindari penggunaan gadget sebelum tidur.")
    elif stress_level == "Medium":
        rekomendasi.append("⚖ Pertahankan keseimbangan antara aktivitas digital dan aktivitas fisik.")
    else:
        rekomendasi.append("✅ Pertahankan gaya hidup sehat yang sudah Anda jalankan.")

    if st.session_state.form_data["daily_screen_time_hours"] > 8:
        rekomendasi.append("📱 Screen time cukup tinggi. Cobalah metode 20-20-20 atau istirahat setiap 60 menit.")
    if st.session_state.form_data["physical_activity_hours_per_week"] < 2:
        rekomendasi.append("🏃 Tingkatkan aktivitas fisik minimal 150 menit per minggu.")
    if st.session_state.form_data["sleep_duration_hours"] < 6:
        rekomendasi.append("😴 Durasi tidur masih kurang. Disarankan tidur 7–9 jam setiap malam.")

    for i, item in enumerate(rekomendasi, start=1):
        st.write(f"{i}. {item}")

    st.divider()
    if st.button("🔄 Selesai & Reset Data", type="primary", use_container_width=True):
        reset_app()