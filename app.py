
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.hero {
    padding: 30px;
    border-radius: 18px;
    background: linear-gradient(135deg, #fff7d6, #eaf7ff);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 40px;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #555;
}

.result-card {
    padding: 25px;
    border-radius: 16px;
    background: #f5f7fa;
    border: 1px solid #ddd;
    text-align: center;
}

.result-value {
    font-size: 38px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = Path("/content/solar_power_prediction_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error("Model could not be loaded.")
    st.exception(e)
    st.stop()

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>☀️ Solar Power Output Prediction</h1>
    <p>
        Predict solar power generation using environmental
        and weather parameters with Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("⚙️ Prediction Settings")
    st.write("Enter the environmental parameters.")
    st.info("Use realistic weather values for better predictions.")

# -----------------------------
# Input Parameters
# -----------------------------
st.subheader("📊 Environmental Parameters")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input(
        "🌡️ Temperature",
        min_value=-50.0,
        max_value=80.0,
        value=25.0,
        step=0.1
    )

    solar_irradiance = st.number_input(
        "☀️ Solar Irradiance (W/m²)",
        min_value=0.0,
        max_value=2000.0,
        value=500.0,
        step=1.0
    )

with col2:
    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

    wind_speed = st.number_input(
        "🌬️ Wind Speed (m/s)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=0.1
    )

# -----------------------------
# Prediction Button
# -----------------------------
if st.button(
    "🔮 Predict Solar Power",
    type="primary",
    use_container_width=True
):

    input_data = pd.DataFrame({
        "temperature": [temperature],
        "humidity": [humidity],
        "solar_irradiance": [solar_irradiance],
        "wind_speed": [wind_speed]
    })

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction completed successfully!")

        st.markdown(f"""
        <div class="result-card">
            <h3>⚡ Predicted Solar Power Output</h3>
            <div class="result-value">
                {prediction:,.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("📌 Input Summary")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Temperature", f"{temperature:.1f} °C")
        m2.metric("Humidity", f"{humidity:.1f} %")
        m3.metric("Irradiance", f"{solar_irradiance:.1f} W/m²")
        m4.metric("Wind Speed", f"{wind_speed:.1f} m/s")

        with st.expander("🔍 View Model Input"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)

# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown("""
<div style="text-align:center;">
    <small>
        ☀️ Solar Power Prediction System |
        Python • Pandas • Machine Learning • Streamlit
    </small>
</div>
""", unsafe_allow_html=True)
