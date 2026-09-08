
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
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
    font-weight: 700;
}

.hero p {
    font-size: 18px;
    color: #555;
}

.result-card {
    padding: 30px;
    border-radius: 18px;
    background: #f8fafc;
    border: 1px solid #dbe3ea;
    text-align: center;
}

.result-value {
    font-size: 42px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL PATH
# =====================================================
MODEL_PATH = Path(__file__).parent / "solar_power_prediction_model.pkl"

# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "❌ Model file not found!\n\n"
        "Make sure 'solar_power_prediction_model.pkl' "
        "is in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model()
except Exception as e:
    st.error("❌ Error while loading the model.")
    st.exception(e)
    st.stop()

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="hero">
    <h1>☀️ Solar Power Output Prediction</h1>
    <p>
        Machine Learning based system for predicting
        solar power output from environmental parameters.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.header("☀️ Solar Prediction")

    st.write(
        "Enter the environmental conditions "
        "to predict solar power output."
    )

    st.divider()

    st.info(
        "This application uses a trained "
        "Machine Learning model."
    )

# =====================================================
# INPUT SECTION
# =====================================================
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

st.divider()

# =====================================================
# PREDICTION
# =====================================================
if st.button(
    "🔮 Predict Solar Power",
    type="primary",
    use_container_width=True
):

    try:

        # Features used by the trained model
        input_data = pd.DataFrame({
            "temperature": [temperature],
            "humidity": [humidity],
            "solar_irradiance": [solar_irradiance],
            "wind_speed": [wind_speed]
        })

        # Make prediction
        prediction = model.predict(input_data)

        predicted_value = float(prediction[0])

        # =================================================
        # RESULT
        # =================================================
        st.success("✅ Prediction completed successfully!")

        st.markdown(
            f"""
            <div class="result-card">

                <h3>⚡ Predicted Solar Power Output</h3>

                <div class="result-value">
                    {predicted_value:,.2f}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        # =================================================
        # INPUT SUMMARY
        # =================================================
        st.subheader("📋 Input Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Temperature",
            f"{temperature:.1f} °C"
        )

        c2.metric(
            "Humidity",
            f"{humidity:.1f} %"
        )

        c3.metric(
            "Solar Irradiance",
            f"{solar_irradiance:.1f} W/m²"
        )

        c4.metric(
            "Wind Speed",
            f"{wind_speed:.1f} m/s"
        )

        # =================================================
        # MODEL INPUT
        # =================================================
        with st.expander("🔍 View Model Input"):

            st.dataframe(
                input_data,
                use_container_width=True
            )

    except Exception as e:

        st.error("❌ Prediction failed.")

        st.warning(
            "Please check that your model was trained "
            "with the same four input features."
        )

        st.exception(e)

# =====================================================
# FOOTER
# =====================================================
st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b;">
        <small>
            ☀️ Solar Power Prediction System<br>
            Python • Pandas • Scikit-learn • Streamlit
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
