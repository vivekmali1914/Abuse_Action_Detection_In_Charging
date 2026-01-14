import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================================
# 1️⃣ PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="EV Charging Abuse Detection",
    page_icon="🔋",
    layout="centered"
)

st.title("🔋 EV Charging Abuse Detection System")
st.markdown(
    "This system detects **abusive EV charging behavior**, identifies the **exact abusive action**, "
    "and provides **eco-friendly recommendations** to protect battery health."
)

# =====================================================
# 2️⃣ LOAD MODEL, SCALER, FEATURES
# =====================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("charging_abuse_model123.pkl")     # update path if needed
    scaler = joblib.load("scaler123.pkl")
    features = joblib.load("features.pkl")
    return model, scaler, features

model, scaler, features = load_artifacts()

# =====================================================
# 3️⃣ RULE-BASED ABUSE ACTION DETECTION
# =====================================================
def detect_abuse_actions(
   # soc_start, 
    soc_end, soc_diff,
    charging_rate, c_rate,
    temperature, thermal_load,
    charging_duration,
    battery_capacity,
    energy_stress
):
    actions = []

    if temperature > 45:
        actions.append("High Temperature Charging")

    if soc_end > 80 and charging_rate > 50:
        actions.append("Fast Charging at High SoC")

    if soc_end >= 95:
        actions.append("Overcharging")

    if charging_duration > 6:
        actions.append("Overnight / Long Duration Charging")

    if charging_rate > battery_capacity:
        actions.append("High Power Charging on Low Capacity Battery")

    if energy_stress > 3:
        actions.append("High Energy Stress Charging")

    if c_rate > 0.8:
        actions.append("Excessive C-Rate Charging")

    if thermal_load > 300:
        actions.append("Thermal Stress Due to Long Charging")

    return actions

# =====================================================
# 4️⃣ ACTION → ECO-FRIENDLY SUGGESTIONS
# =====================================================
ACTION_SUGGESTIONS = {
    "High Temperature Charging":
        "Pause charging until the battery temperature drops below 45°C.",

    "Fast Charging at High SoC":
        "Switch to slow charging above 80% to reduce battery stress.",

    "Overcharging":
        "Avoid charging beyond 90% for daily usage.",

    "Overnight / Long Duration Charging":
        "Unplug the charger once charging is complete.",

    "High Power Charging on Low Capacity Battery":
        "Use a charger compatible with your battery capacity.",

    "High Energy Stress Charging":
        "Reduce charging power to minimize energy stress.",

    "Excessive C-Rate Charging":
        "Decrease the Voltage to improve long-term battery health.",

    "Thermal Stress Due to Long Charging":
        "Avoid long continuous charging sessions at high temperature."
}

# =====================================================
# 5️⃣ USER INPUTS
# =====================================================
st.subheader("🔧 Enter Charging Parameters")

soc_end = st.slider("SOC End (%)", 0, 100, 55)
energy_stress = st.slider("Energy Stress", 0.0, 5.0, 1.0, 0.1)
c_rate = st.slider("C-Rate", 0.0, 2.0, 0.6, 0.05)
charging_rate = st.slider("Charging Rate (kW)", 1, 120, 40)
soc_diff = st.slider("SOC Difference", -100, 130, 30)
#soc_start = st.slider("SOC Start (%)", 0, 100, 40)
battery_capacity = st.slider("Battery Capacity (kWh)", 20, 200, 80)
energy_consumed = st.slider("Energy Consumed (kWh)", 0, 150, 20)
temperature = st.slider("Battery Temperature (°C)", -10, 80, 25)
thermal_load = st.slider("Thermal Load (Temp × Duration)", 0, 500, 150)
charging_duration = st.slider("Charging Duration (hours)", 0.1, 8.0, 2.0, 0.1)
#distance_driven = st.slider("Distance Driven since last charge (km)", 0, 400, 50)
#vehicle_age = st.slider("Vehicle Age (years)", 0, 15, 3)

# =====================================================
# 6️⃣ PREDICTION BUTTON
# =====================================================
if st.button("🔍 Analyze Charging Behavior"):

    input_df = pd.DataFrame([[
        battery_capacity,
        energy_consumed,
        charging_duration,
        charging_rate,
       # soc_start,
        soc_end,
        #distance_driven,
        temperature,
       # vehicle_age,
        soc_diff,
        energy_stress,
        c_rate,
        thermal_load
    ]], columns=features)

    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    abuse_actions = detect_abuse_actions(
        #soc_start,
        soc_end, soc_diff,
        charging_rate, c_rate,
        temperature, thermal_load,
        charging_duration,
        battery_capacity,
        energy_stress
    )

    st.markdown("---")

    # =================================================
    # 7️⃣ OUTPUT DISPLAY
    # =================================================
    if pred == 1:
        st.error(f"⚠️ **ABUSIVE CHARGING DETECTED**  \nConfidence: **{prob:.2f}**")

        if abuse_actions:
            st.subheader("🚨 Abusive Action(s)")
            for action in abuse_actions:
                st.write(f"• {action}")

            st.subheader("🌱 Eco-Friendly Suggestions")
            for action in abuse_actions:
                if action in ACTION_SUGGESTIONS:
                    st.write(f"• {ACTION_SUGGESTIONS[action]}")
        else:
            st.warning("Abnormal charging pattern detected by ML model.")
            st.info(
                "Reduce charging power, avoid extreme temperatures, "
                "and follow recommended SOC limits."
            )

    else:
        st.success(f"✅ **NORMAL CHARGING BEHAVIOR**  \nConfidence: **{1 - prob:.2f}**")
        st.info("Charging behavior is eco-friendly. Keep following best practices.")

